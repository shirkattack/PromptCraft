import logging
from collections import deque
from datetime import UTC, datetime
from typing import Any

import dspy
from fastapi.concurrency import run_in_threadpool

from app.core.config import settings
from app.services.lm_manager import LMManager

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
        self.optimization_history: deque = deque(
            maxlen=settings.optimization_history_size
        )

    async def optimize_prompt(
        self,
        original_prompt: str,
        provider: str,
        model: str,
        task_type: str = "general",
        optimization_method: str = "meta_prompt",
        **kwargs,
    ) -> dict[str, Any]:
        """
        Optimize a prompt using the specified method and provider.

        Args:
            original_prompt: The prompt to optimize
            provider: AI provider to use
            model: Model name
            task_type: Type of task (general, code, creative, etc.)
            optimization_method: Method to use (meta_prompt, dspy, simple)

        Returns:
            Dictionary containing optimization results. ``success`` is False when
            no method produced a prompt that differs from the original.
        """
        start_time = datetime.now(UTC)

        try:
            lm = LMManager.get_lm(
                provider=provider,
                model_name=model,
                temperature=settings.default_temperature,
                max_tokens=2000,
            )

            # Model calls are synchronous and can run for minutes against a
            # local model, so they must not run on the event loop.
            result = await run_in_threadpool(
                self._run_optimization,
                lm,
                original_prompt,
                task_type,
                optimization_method,
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
        improvement_score = self._calculate_improvement_score(
            original_prompt, result["optimized_prompt"]
        )

        optimization_result = {
            "original_prompt": original_prompt,
            "optimized_prompt": result["optimized_prompt"],
            "method": optimization_method,
            "provider": provider,
            "model": model,
            "task_type": task_type,
            "improvement_score": improvement_score,
            "processing_time": processing_time,
            "metadata": result.get("metadata", {}),
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
        lm,
        original_prompt: str,
        task_type: str,
        optimization_method: str,
    ) -> dict[str, Any]:
        """Run the selected optimization strategy. Executed in a worker thread.

        The DSPy context is entered here rather than in the caller so that the
        configured LM is bound to the thread that actually makes the calls.
        """
        with dspy.context(lm=lm):
            if optimization_method == "meta_prompt":
                return self._optimize_with_meta_prompt(original_prompt, task_type, lm)
            if optimization_method == "dspy":
                return self._optimize_with_dspy(original_prompt, task_type, lm)
            return self._simple_optimization(original_prompt, task_type, lm)

    def _optimize_with_meta_prompt(
        self, original_prompt: str, task_type: str, lm
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
            return self._simple_optimization(original_prompt, task_type, lm)

    def _optimize_with_dspy(
        self, original_prompt: str, task_type: str, lm
    ) -> dict[str, Any]:
        """Optimize using DSPy's ChainOfThought over an explicit rewrite signature."""

        class PromptRewrite(dspy.Signature):
            """Rewrite a prompt so a language model answers it more reliably."""

            original_prompt = dspy.InputField(desc="The prompt to improve")
            task_type = dspy.InputField(desc="The kind of task the prompt is for")
            optimized_prompt = dspy.OutputField(
                desc="An improved prompt: specific, structured, unambiguous"
            )

        try:
            predictor = dspy.ChainOfThought(PromptRewrite)
            result = predictor(original_prompt=original_prompt, task_type=task_type)
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
        self, original_prompt: str, task_type: str, lm
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

    def _generate_meta_prompt(self, original_prompt: str, task_type: str) -> str:
        """Generate meta-prompt for optimization (adapted from Promptomatix)."""

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
- Elimination of ambiguity

## Optimized Prompt:"""

    def _calculate_improvement_score(self, original: str, optimized: str) -> float:
        """Heuristic 0-100 quality signal for the rewrite.

        This is a structural heuristic (length, formatting, sectioning), not a
        measured performance gain -- it does not evaluate the prompt against any
        task. Treat it as a sanity check, not a benchmark.
        """

        if optimized.strip() == original.strip():
            # Nothing changed, so there is nothing to score.
            return 0.0

        score = 50.0  # Base score

        # Length improvement (more detailed prompts are often better)
        length_ratio = len(optimized) / len(original) if len(original) > 0 else 1.0
        if 1.2 <= length_ratio <= 3.0:  # Good length increase
            score += 20.0
        elif length_ratio > 3.0:  # Too verbose
            score += 10.0

        # Structure improvements (simple heuristics)
        if "##" in optimized or "**" in optimized:  # Has formatting
            score += 10.0

        if "example" in optimized.lower() or "format" in optimized.lower():
            score += 10.0

        if len(optimized.split("\n")) > len(original.split("\n")):  # Better structure
            score += 10.0

        # Cap the score
        return min(score, 100.0)

    def get_optimization_history(self) -> list[dict[str, Any]]:
        """Get the history of optimizations performed."""
        return list(self.optimization_history)

    def clear_history(self) -> None:
        """Clear optimization history."""
        self.optimization_history.clear()
        self.optimized_prompt = None


# Global service instance
optimization_service = PromptOptimizationService()
