import logging
from collections import deque
from datetime import UTC, datetime
from typing import Any

import dspy
from fastapi.concurrency import run_in_threadpool

from app.core.config import settings
from app.services.eval_service import DatasetOptimizer, EvalMetric, Sample
from app.services.lm_manager import LMManager
from app.services.progress import ProgressCallback, no_progress

logger = logging.getLogger(__name__)


class PromptOptimizationService:
    """
    Service for optimizing prompts using DSPy and meta-prompt techniques.
    Adapted from Promptomatix core functionality.
    """

    def __init__(self) -> None:
        self.optimized_prompt = None
        # Bounded: this lives on a module-level singleton for the process
        # lifetime, so an unbounded list would grow without limit.
        self.optimization_history: deque[dict[str, Any]] = deque(
            maxlen=settings.optimization_history_size
        )

    async def optimize_prompt(
        self,
        original_prompt: str,
        provider: str,
        model: str,
        task_type: str = "general",
        optimization_method: str = "meta_prompt",
        temperature: float | None = None,
        max_tokens: int | None = None,
        output_format: str = "auto",
        target_length: str = "auto",
        preserve_wording: bool = False,
        dataset_samples: list[Sample] | None = None,
        eval_metric: EvalMetric = "auto",
        max_demos: int = 4,
        progress: ProgressCallback = no_progress,
    ) -> dict[str, Any]:
        """
        Optimize a prompt using the specified method and provider.

        Args:
            original_prompt: The prompt to optimize
            provider: AI provider to use
            model: Model name
            task_type: Type of task (general, code, creative, etc.)
            optimization_method: Method to use (meta_prompt, dspy, simple)
            dataset_samples: When given, the rewrite is measured against these
                (input, expected output) pairs and few-shot variants are
                compiled with DSPy; the best candidate on held-out samples is
                returned and ``improvement_score`` is that measured score.
            eval_metric: How a model answer is compared with the expected one.
            max_demos: Cap on few-shot examples per compiled candidate.

        Returns:
            Dictionary containing optimization results. ``success`` is False when
            no method produced a prompt that differs from the original.
        """
        start_time = datetime.now(UTC)
        run_settings: dict[str, Any] = {
            "temperature": (
                settings.default_temperature if temperature is None else temperature
            ),
            "max_tokens": 2000 if max_tokens is None else max_tokens,
            "output_format": output_format,
            "target_length": target_length,
            "preserve_wording": preserve_wording,
        }
        constraints = self._build_constraints(
            output_format, target_length, preserve_wording
        )

        try:
            lm = LMManager.get_lm(
                provider=provider,
                model_name=model,
                temperature=run_settings["temperature"],
                max_tokens=run_settings["max_tokens"],
            )

            # Model calls are synchronous and can run for minutes against a
            # local model, so they must not run on the event loop.
            result = await run_in_threadpool(
                self._run_optimization,
                lm,
                original_prompt,
                task_type,
                optimization_method,
                constraints,
                dataset_samples,
                eval_metric,
                max_demos,
                progress,
            )
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return self._failure_result(
                original_prompt,
                provider,
                model,
                task_type,
                optimization_method,
                str(e),
                start_time,
            )

        error = result.get("metadata", {}).get("error")
        if error:
            return self._failure_result(
                original_prompt,
                provider,
                model,
                task_type,
                optimization_method,
                error,
                start_time,
            )

        processing_time = (datetime.now(UTC) - start_time).total_seconds()
        evaluation = result.get("metadata", {}).get("eval")

        if evaluation:
            # Measured: percentage of held-out samples the metric accepted.
            improvement_score = float(evaluation["eval_score"])
            score_breakdown = [
                {
                    "label": (
                        f"Measured on {evaluation['dev_size']} held-out samples "
                        f"({evaluation['metric']} metric)"
                    ),
                    "points": improvement_score,
                    "applied": True,
                }
            ]
            score_type = "measured"
        else:
            improvement_score = self._calculate_improvement_score(
                original_prompt, result["optimized_prompt"]
            )
            score_breakdown = self._score_breakdown(
                original_prompt, result["optimized_prompt"]
            )
            score_type = "heuristic"

        optimization_result = {
            "original_prompt": original_prompt,
            "optimized_prompt": result["optimized_prompt"],
            "method": optimization_method,
            "provider": provider,
            "model": model,
            "task_type": task_type,
            "improvement_score": improvement_score,
            "score_type": score_type,
            "baseline_score": evaluation["baseline_score"] if evaluation else None,
            "eval_score": evaluation["eval_score"] if evaluation else None,
            "eval_metric": evaluation["metric"] if evaluation else None,
            "eval_sample_count": evaluation["dev_size"] if evaluation else None,
            "processing_time": processing_time,
            "metadata": {
                **result.get("metadata", {}),
                "settings": run_settings,
                "score_type": score_type,
                "score_breakdown": score_breakdown,
            },
            "success": True,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        self.optimization_history.append(optimization_result)
        self.optimized_prompt = result["optimized_prompt"]

        return optimization_result

    def _failure_result(
        self,
        original_prompt: str,
        provider: str,
        model: str,
        task_type: str,
        optimization_method: str,
        error: str,
        start_time: datetime,
    ) -> dict[str, Any]:
        """Build the response for an optimization that produced nothing usable."""
        return {
            "original_prompt": original_prompt,
            "optimized_prompt": original_prompt,  # Fallback
            "method": optimization_method,
            "provider": provider,
            "model": model,
            "task_type": task_type,
            "improvement_score": 0.0,
            "processing_time": (datetime.now(UTC) - start_time).total_seconds(),
            "error": error,
            "success": False,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def _run_optimization(
        self,
        lm: dspy.LM,
        original_prompt: str,
        task_type: str,
        optimization_method: str,
        constraints: str = "",
        dataset_samples: list[Sample] | None = None,
        eval_metric: EvalMetric = "auto",
        max_demos: int = 4,
        progress: ProgressCallback = no_progress,
    ) -> dict[str, Any]:
        """Run the selected optimization strategy. Executed in a worker thread.

        The DSPy context is entered here rather than in the caller so that the
        configured LM is bound to the thread that actually makes the calls.
        """
        with dspy.context(lm=lm):
            progress("rewrite", f"Rewriting the prompt ({optimization_method})")
            if optimization_method == "meta_prompt":
                result = self._optimize_with_meta_prompt(
                    original_prompt, task_type, lm, constraints
                )
            elif optimization_method == "dspy":
                result = self._optimize_with_dspy(
                    original_prompt, task_type, lm, constraints
                )
            else:
                result = self._simple_optimization(
                    original_prompt, task_type, lm, constraints
                )

            if not dataset_samples or result.get("metadata", {}).get("error"):
                return result

            return self._optimize_against_dataset(
                original_prompt,
                result,
                dataset_samples,
                eval_metric,
                max_demos,
                progress,
            )

    @staticmethod
    def _optimize_against_dataset(
        original_prompt: str,
        rewrite: dict[str, Any],
        samples: list[Sample],
        eval_metric: EvalMetric,
        max_demos: int,
        progress: ProgressCallback = no_progress,
    ) -> dict[str, Any]:
        """Measure the rewrite on the dataset and return the best candidate.

        The rewrite is kept in ``metadata["rewrite"]`` so the client can still
        show it even when a few-shot variant of the original prompt scored
        higher.
        """
        optimizer = DatasetOptimizer(
            samples, metric=eval_metric, max_demos=max_demos, progress=progress
        )
        report = optimizer.run(original_prompt, rewrite["optimized_prompt"])

        return {
            "optimized_prompt": report.pop("optimized_prompt"),
            "metadata": {
                **rewrite.get("metadata", {}),
                "rewrite": rewrite["optimized_prompt"],
                "eval": report,
            },
        }

    def _optimize_with_meta_prompt(
        self, original_prompt: str, task_type: str, lm: dspy.LM, constraints: str = ""
    ) -> dict[str, Any]:
        """Optimize using meta-prompt technique from Promptomatix."""

        meta_prompt = self._generate_meta_prompt(original_prompt, task_type)

        try:
            # Use DSPy to generate the optimized prompt
            predictor = dspy.Predict("meta_prompt -> optimized_prompt")
            result = predictor(meta_prompt=meta_prompt)

            return {
                "optimized_prompt": result.optimized_prompt.strip(),
                "metadata": {
                    "method": "meta_prompt",
                    "meta_prompt_used": (
                        meta_prompt[:200] + "..."
                        if len(meta_prompt) > 200
                        else meta_prompt
                    ),
                },
            }
        except Exception as e:
            logger.error(f"Meta-prompt optimization failed: {e}")
            # Fallback to simple optimization
            return self._simple_optimization(
                original_prompt, task_type, lm, constraints
            )

    def _optimize_with_dspy(
        self, original_prompt: str, task_type: str, lm: dspy.LM, constraints: str = ""
    ) -> dict[str, Any]:
        """Optimize using DSPy's ChainOfThought over an explicit rewrite signature."""

        class PromptRewrite(dspy.Signature):
            """Rewrite a prompt so a language model answers it more reliably."""

            original_prompt = dspy.InputField(desc="The prompt to improve")
            task_type = dspy.InputField(desc="The kind of task the prompt is for")
            constraints = dspy.InputField(
                desc="Requirements the rewritten prompt must satisfy ('none' if empty)"
            )
            optimized_prompt = dspy.OutputField(
                desc="An improved prompt: specific, structured, unambiguous"
            )

        try:
            predictor = dspy.ChainOfThought(PromptRewrite)
            result = predictor(
                original_prompt=original_prompt,
                task_type=task_type,
                constraints=constraints or "none",
            )
            optimized = result.optimized_prompt.strip()

            if optimized:
                return {
                    "optimized_prompt": optimized,
                    "metadata": {
                        "method": "dspy",
                        "signature": "PromptRewrite",
                        "predictor": "ChainOfThought",
                        "reasoning": getattr(result, "reasoning", None),
                    },
                }
            raise ValueError("DSPy returned an empty prompt")
        except Exception as e:
            # Falling back to a deterministic template keeps the endpoint usable
            # when the model cannot satisfy the structured DSPy output format.
            logger.warning(f"DSPy optimization fell back to template: {e}")
            return {
                "optimized_prompt": (
                    f"Task: {task_type.title()} Task\n\n"
                    f"{original_prompt}\n\n"
                    "Please provide a clear, detailed response that addresses all aspects "
                    "of the request. Think step by step and ensure your response is "
                    "comprehensive and accurate."
                ),
                "metadata": {
                    "method": "dspy",
                    "predictor": "template_fallback",
                    "fallback_reason": str(e),
                },
            }

    def _simple_optimization(
        self, original_prompt: str, task_type: str, lm: dspy.LM, constraints: str = ""
    ) -> dict[str, Any]:
        """Simple optimization using direct LM completion."""

        optimization_prompt = f"""You are an expert prompt engineer. Analyze and improve the following prompt to make it more effective, clear, and likely to produce better results.

Original prompt:
{original_prompt}

Task type: {task_type}

Please provide an improved version that:
1. Is more specific and clear
2. Includes better instructions
3. Has appropriate context
4. Will produce more consistent results
5. Follows prompt engineering best practices
{constraints}
Improved prompt:"""

        try:
            response = lm(optimization_prompt, max_tokens=1000)
            # Handle both string and list responses
            if isinstance(response, list):
                optimized_prompt = response[0] if response else original_prompt
            else:
                optimized_prompt = str(response)
            optimized_prompt = optimized_prompt.strip()

            # Clean up the response if it includes extra text
            if "Improved prompt:" in optimized_prompt:
                optimized_prompt = optimized_prompt.split("Improved prompt:")[
                    -1
                ].strip()

            if not optimized_prompt:
                raise ValueError("Model returned an empty prompt")

            return {
                "optimized_prompt": optimized_prompt,
                "metadata": {
                    "method": "simple",
                    "optimization_prompt_length": len(optimization_prompt),
                },
            }
        except Exception as e:
            logger.error(f"Simple optimization failed: {e}")
            # Reported as an error rather than a "successful" no-op so callers
            # are not handed the original prompt with a passing score.
            return {
                "optimized_prompt": original_prompt,
                "metadata": {
                    "method": "simple",
                    "error": str(e),
                    "fallback": True,
                },
            }

    @staticmethod
    def _build_constraints(
        output_format: str, target_length: str, preserve_wording: bool
    ) -> str:
        """Turn the advanced settings into instructions the strategies can embed."""
        lines = []
        if output_format == "markdown":
            lines.append("The prompt must ask for the answer in Markdown.")
        elif output_format == "plain":
            lines.append(
                "The prompt must ask for plain text with no Markdown formatting."
            )
        elif output_format == "json":
            lines.append(
                "The prompt must ask for a JSON response and describe its schema."
            )
        if target_length == "concise":
            lines.append(
                "Keep the rewritten prompt concise: no longer than the original plus a few clarifying lines."
            )
        elif target_length == "balanced":
            lines.append(
                "Keep the rewritten prompt to a moderate length: roughly 1.5-2x the original."
            )
        elif target_length == "detailed":
            lines.append(
                "Make the rewritten prompt thorough: add context, constraints and an example if useful."
            )
        if preserve_wording:
            lines.append(
                "Preserve the original wording of the request; improve structure and add instructions around it rather than rephrasing it."
            )
        return "\n".join(f"- {line}" for line in lines)

    def _generate_meta_prompt(
        self, original_prompt: str, task_type: str, constraints: str = ""
    ) -> str:
        """Generate meta-prompt for optimization (adapted from Promptomatix)."""
        constraints_section = (
            f"\n\n## Constraints:\n{constraints}" if constraints else ""
        )

        return f"""You are an expert prompt engineer specializing in {task_type} tasks. Your goal is to analyze and dramatically improve the following prompt to make it more effective, specific, and reliable.

## Original Prompt Analysis:
{original_prompt}

## Task Type: {task_type}

## Optimization Guidelines:
1. **Clarity & Specificity**: Make instructions crystal clear and unambiguous
2. **Context**: Add relevant context that helps the AI understand the task better
3. **Structure**: Organize the prompt with clear sections and formatting
4. **Examples**: Include examples if they would improve understanding
5. **Constraints**: Add appropriate constraints to guide the output format
6. **Best Practices**: Apply proven prompt engineering techniques

## Your Task:
Rewrite the prompt to be significantly more effective. Focus on:
- Clear, actionable instructions
- Proper formatting and structure
- Relevant context and constraints
- Better specification of desired output
- Elimination of ambiguity{constraints_section}

## Optimized Prompt:"""

    def _score_breakdown(self, original: str, optimized: str) -> list[dict[str, Any]]:
        """Itemised rubric behind the heuristic score.

        Every criterion is listed with whether it applied, so a client can show
        the user exactly what the number rewards. This is a structural check
        (length, formatting, sectioning), not a measured performance gain -- it
        does not evaluate the prompt against any task.
        """
        if optimized.strip() == original.strip():
            return [{"label": "Prompt unchanged", "points": 0, "applied": True}]

        length_ratio = len(optimized) / len(original) if len(original) > 0 else 1.0
        lowered = optimized.lower()

        return [
            {"label": "Base", "points": 50, "applied": True},
            {
                "label": "Length 1.2-3x the original",
                "points": 20,
                "applied": 1.2 <= length_ratio <= 3.0,
            },
            {
                "label": "Longer than 3x (verbose)",
                "points": 10,
                "applied": length_ratio > 3.0,
            },
            {
                "label": "Markdown formatting (## or **)",
                "points": 10,
                "applied": "##" in optimized or "**" in optimized,
            },
            {
                "label": "Mentions examples or a format",
                "points": 10,
                "applied": "example" in lowered or "format" in lowered,
            },
            {
                "label": "More lines than the original",
                "points": 10,
                "applied": len(optimized.split("\n")) > len(original.split("\n")),
            },
        ]

    def _calculate_improvement_score(self, original: str, optimized: str) -> float:
        """Heuristic 0-100 quality signal for the rewrite; see _score_breakdown."""
        total = sum(
            item["points"]
            for item in self._score_breakdown(original, optimized)
            if item["applied"]
        )
        return float(min(total, 100))

    def get_optimization_history(self) -> list[dict[str, Any]]:
        """Get the history of optimizations performed."""
        return list(self.optimization_history)

    def clear_history(self) -> None:
        """Clear optimization history."""
        self.optimization_history.clear()
        self.optimized_prompt = None


# Global service instance
optimization_service = PromptOptimizationService()
