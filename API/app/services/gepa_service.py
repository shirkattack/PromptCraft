"""GEPA: reflective prompt evolution against a dataset.

GEPA (Agrawal et al., 2025) runs the prompt on a few training samples, asks a
metric for *written feedback* on each miss, has a reflection model rewrite the
instructions to address that feedback, and keeps a Pareto front of candidates
that each win on different samples. This module wraps ``dspy.teleprompt.GEPA``
so that:

* the metric produces feedback a small local model can act on (for example
  "the right label is buried in a 60-word answer"),
* every iteration is reported as progress while the job runs, and
* the run returns a timeline: each candidate's instructions, score, parent
  and the feedback that led to it, so the client can show why each edit
  happened.
"""

import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import dspy
from dspy.teleprompt import GEPA

from app.core.config import settings
from app.services.eval_service import (
    SHORT_ANSWER_CHARS,
    EvalError,
    EvalMetric,
    Sample,
    choose_metric,
    normalize,
    render_prompt,
    split_samples,
)
from app.services.progress import ProgressCallback, no_progress

logger = logging.getLogger(__name__)

GEPA_LOGGER_NAME = "dspy.teleprompt.gepa.gepa"

# Feedback metric signature GEPA calls: (gold, pred, trace, pred_name, pred_trace)
FeedbackMetric = Callable[..., dspy.Prediction]


# -- feedback metrics ---------------------------------------------------------


class JudgeWithReason(dspy.Signature):
    """Decide whether a response satisfies the expected output, and say why.

    Judge meaning, not wording. The reason must be specific enough that a
    prompt author could act on it: name what is missing, wrong or extra.
    """

    input = dspy.InputField(desc="The input the response was written for")
    expected_output = dspy.InputField(desc="A reference answer known to be correct")
    actual_output = dspy.InputField(desc="The response being judged")
    verdict = dspy.OutputField(desc="'yes' if the response is correct, otherwise 'no'")
    reason = dspy.OutputField(
        desc="One or two sentences on what made it right or wrong"
    )


def label_feedback_metric(exact: bool) -> FeedbackMetric:
    """Feedback for label-style outputs (classes, short answers).

    Full credit for the bare label, half credit when the label is present but
    wrapped in extra text (with feedback telling the model to stop), none
    otherwise.
    """

    def metric(
        gold: dspy.Example,
        pred: Any,
        trace: Any = None,
        pred_name: str | None = None,
        pred_trace: Any = None,
    ) -> dspy.Prediction:
        expected = normalize(gold.output)
        raw = str(getattr(pred, "output", "") or "")
        actual = normalize(raw)
        words = len(raw.split())

        if actual == expected:
            return dspy.Prediction(
                score=1.0, feedback=f"Correct: answered exactly '{gold.output}'."
            )
        if not exact and expected and expected in actual:
            return dspy.Prediction(
                score=0.5,
                feedback=(
                    f"The right answer '{gold.output}' is in the response but buried "
                    f"in {words} words. Respond with '{gold.output}' alone, no explanation."
                ),
            )
        return dspy.Prediction(
            score=0.0,
            feedback=(
                f"Wrong. Expected '{gold.output}' but got '{raw[:120]}'. "
                f"Input was: '{gold.input[:160]}'."
            ),
        )

    return metric


def judge_feedback_metric() -> FeedbackMetric:
    """Feedback for free-text outputs: a model judge's verdict and reason."""
    judge = dspy.Predict(JudgeWithReason)

    def metric(
        gold: dspy.Example,
        pred: Any,
        trace: Any = None,
        pred_name: str | None = None,
        pred_trace: Any = None,
    ) -> dspy.Prediction:
        raw = str(getattr(pred, "output", "") or "").strip()
        if not raw:
            return dspy.Prediction(score=0.0, feedback="The response was empty.")
        verdict = judge(
            input=gold.input, expected_output=gold.output, actual_output=raw
        )
        ok = normalize(verdict.verdict).startswith("yes")
        reason = str(getattr(verdict, "reason", "") or "").strip()
        return dspy.Prediction(
            score=1.0 if ok else 0.0,
            feedback=(
                f"{'Correct' if ok else 'Incorrect'}: {reason}"
                if reason
                else (
                    "Correct." if ok else f"Incorrect. Expected: '{gold.output[:160]}'."
                )
            ),
        )

    return metric


def build_feedback_metric(metric_name: str) -> FeedbackMetric:
    if metric_name == "exact":
        return label_feedback_metric(exact=True)
    if metric_name == "contains":
        return label_feedback_metric(exact=False)
    if metric_name == "llm_judge":
        return judge_feedback_metric()
    raise EvalError(f"Unknown metric: {metric_name}")


# -- instruction clean-up -----------------------------------------------------

_FENCE = re.compile(r"^\s*```[\w-]*\s*\n|\n\s*```\s*$")


def clean_instructions(text: str) -> str:
    """Undo formatting a small reflection model wraps its proposal in.

    llama3.2 tends to return the new instructions as a fenced code block of
    Python comments ("python\\n# Classify ..."). The content is fine; the
    markup would end up in the user's prompt.
    """
    cleaned = _FENCE.sub("", text.strip())
    lines = cleaned.splitlines()
    if lines and lines[0].strip().lower() in {"python", "text", "markdown", "md"}:
        lines = lines[1:]
    stripped = [ln.strip() for ln in lines if ln.strip()]
    if stripped and all(ln.startswith("#") for ln in stripped):
        lines = [re.sub(r"^\s*#\s?", "", ln) for ln in lines]
    return "\n".join(lines).strip()


# -- iteration tracking --------------------------------------------------------

_ITERATION = re.compile(r"^Iteration (\d+): (.*)$", re.S)


@dataclass
class IterationEvent:
    iteration: int
    kind: str  # selected | proposed | accepted | skipped | scored | other
    message: str
    value: float | None = None


class GepaTracker(logging.Handler):
    """Turns GEPA's log lines into progress updates and a per-iteration record.

    Also collects the feedback strings the metric emits, keyed by the
    iteration in flight, so each proposal can be shown with the feedback that
    triggered it.
    """

    def __init__(self, progress: ProgressCallback) -> None:
        super().__init__(level=logging.INFO)
        self.progress = progress
        self.events: list[IterationEvent] = []
        self.feedback_by_iteration: dict[int, list[str]] = {}
        self.current_iteration = 0
        self.best_score: float | None = None
        self._lock = threading.Lock()

    def record_feedback(self, feedback: str, score: float) -> None:
        if score >= 1.0:
            return  # only misses carry information for the reflection step
        with self._lock:
            bucket = self.feedback_by_iteration.setdefault(self.current_iteration, [])
            if feedback not in bucket and len(bucket) < 12:
                bucket.append(feedback)

    def emit(self, record: logging.LogRecord) -> None:
        match = _ITERATION.match(record.getMessage())
        if not match:
            return
        iteration, rest = int(match.group(1)), match.group(2).strip()
        with self._lock:
            self.current_iteration = iteration

        kind, value = "other", None
        if rest.startswith("Selected program"):
            kind = "selected"
        elif rest.startswith("Proposed new text"):
            kind = "proposed"
            rest = "Reflection proposed new instructions"
        elif "New subsample score is not better" in rest or "worse than both" in rest:
            kind = "skipped"
            rest = "Proposal did not beat its parent on the sample; skipped"
        elif rest.startswith("New program candidate index"):
            kind = "accepted"
        elif rest.startswith("Full valset score for new program"):
            kind = "scored"
            value = _trailing_float(rest)
        elif rest.startswith("Best valset aggregate score so far"):
            value = _trailing_float(rest)
            if value is not None:
                self.best_score = value
        elif rest.startswith("Base program full valset score"):
            value = _trailing_float(rest)
            if value is not None:
                self.best_score = value

        self.events.append(IterationEvent(iteration, kind, rest[:200], value))
        if kind in {"selected", "proposed", "skipped", "accepted", "scored"}:
            self.progress(
                "evolve",
                f"Generation {iteration}: {rest[:120]}",
                current=iteration,
                total=None,
                best_score=(
                    round(self.best_score * 100, 1)
                    if self.best_score is not None
                    else None
                ),
            )

    def accepted_iterations(self) -> list[int]:
        return [e.iteration for e in self.events if e.kind == "accepted"]


def _trailing_float(text: str) -> float | None:
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*$", text)
    return float(match.group(1)) if match else None


# -- the optimizer ---------------------------------------------------------------


@dataclass
class GepaCandidate:
    index: int
    parent: int | None
    generation: int
    instructions: str
    score: float | None
    iteration: int | None = None
    feedback: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "parent": self.parent,
            "generation": self.generation,
            "instructions": self.instructions,
            "score": self.score,
            "iteration": self.iteration,
            "feedback": self.feedback,
        }


class GepaOptimizer:
    """Evolve a prompt's instructions with GEPA and measure the result.

    Must be called inside ``dspy.context(lm=...)`` on the thread that runs it.
    """

    def __init__(
        self,
        samples: list[Sample],
        metric: EvalMetric = "auto",
        budget: int = 60,
        reflection_lm: dspy.LM | None = None,
        train_ratio: float | None = None,
        seed: int = 13,
        progress: ProgressCallback = no_progress,
    ) -> None:
        if len(samples) < 2:
            raise EvalError("GEPA needs a dataset with at least 2 samples")
        self.samples = samples
        self.metric_name = choose_metric(metric, samples)
        self.budget = max(10, budget)
        self.reflection_lm = reflection_lm
        self.seed = seed
        self.progress = progress
        self.train, self.dev = split_samples(
            samples,
            settings.default_train_ratio if train_ratio is None else train_ratio,
            settings.eval_max_train_samples,
            settings.eval_max_dev_samples,
            seed,
        )
        self.metric = build_feedback_metric(self.metric_name)

    # -- evaluation helpers

    def _evaluate(self, program: dspy.Module) -> tuple[float, list[dict[str, Any]]]:
        """Score a program on the held-out split; returns (percent, rows)."""
        rows = []
        total = 0.0
        for sample in self.dev:
            example = sample.to_example()
            try:
                pred = program(input=sample.input_text)
            except Exception as exc:  # a single failed call scores zero
                logger.warning(f"Held-out call failed: {exc}")
                pred = dspy.Prediction(output="")
            verdict = self.metric(example, pred)
            score = float(verdict.score)
            total += score
            rows.append(
                {
                    "input": sample.input_text,
                    "expected": sample.expected_output,
                    "actual": str(getattr(pred, "output", "") or ""),
                    "passed": score >= 1.0,
                    "score": score,
                    "feedback": str(getattr(verdict, "feedback", "") or ""),
                }
            )
        percent = round(total / len(self.dev) * 100, 2) if self.dev else 0.0
        return percent, rows

    @staticmethod
    def _program(instructions: str) -> dspy.Predict:
        return dspy.Predict(dspy.Signature("input -> output", instructions.strip()))

    # -- main entry

    def run(self, original: str) -> dict[str, Any]:
        started = time.time()
        self.progress(
            "evaluate",
            f"Scoring the original prompt on {len(self.dev)} held-out samples",
            current=0,
            total=None,
        )
        baseline_score, baseline_rows = self._evaluate(self._program(original))
        self.progress(
            "evolve",
            f"Original scored {baseline_score:.0f}%. Evolving with a budget of "
            f"{self.budget} scored calls",
            current=0,
            total=None,
            best_score=baseline_score,
        )

        tracker = GepaTracker(self.progress)
        tracker.best_score = baseline_score / 100

        base_metric = self.metric

        def tracked_metric(
            gold: dspy.Example,
            pred: Any,
            trace: Any = None,
            pred_name: str | None = None,
            pred_trace: Any = None,
        ) -> dspy.Prediction:
            verdict = base_metric(gold, pred, trace, pred_name, pred_trace)
            tracker.record_feedback(
                str(getattr(verdict, "feedback", "") or ""), float(verdict.score)
            )
            return verdict

        optimizer = GEPA(
            metric=tracked_metric,
            max_metric_calls=self.budget,
            reflection_minibatch_size=min(3, len(self.train)),
            reflection_lm=self.reflection_lm,
            num_threads=1,  # one local model; parallel calls just queue
            track_stats=True,
            skip_perfect_score=True,
            seed=self.seed,
        )

        gepa_logger = logging.getLogger(GEPA_LOGGER_NAME)
        previous_level = gepa_logger.level
        gepa_logger.addHandler(tracker)
        if gepa_logger.level > logging.INFO or gepa_logger.level == logging.NOTSET:
            gepa_logger.setLevel(logging.INFO)
        try:
            compiled = optimizer.compile(
                self._program(original),
                trainset=[s.to_example() for s in self.train],
                valset=[s.to_example() for s in self.dev],
            )
        finally:
            gepa_logger.removeHandler(tracker)
            gepa_logger.setLevel(previous_level)

        self.progress(
            "evaluate",
            f"Scoring the evolved prompt on {len(self.dev)} held-out samples",
            current=tracker.current_iteration,
            total=None,
        )
        evolved_instructions = clean_instructions(
            str(getattr(compiled.signature, "instructions", "") or original)
        )
        evolved_program = self._program(evolved_instructions)
        final_score, final_rows = self._evaluate(evolved_program)

        timeline = self._timeline(compiled, original, tracker)
        improved = final_score > baseline_score
        chosen_instructions = evolved_instructions if improved else original
        chosen_score = final_score if improved else baseline_score

        elapsed = round(time.time() - started, 1)
        detailed = getattr(compiled, "detailed_results", None)
        metric_calls = getattr(detailed, "total_metric_calls", None)

        report = {
            "budget": self.budget,
            "metric_calls": metric_calls,
            "iterations": tracker.current_iteration,
            "reflection_model": getattr(self.reflection_lm, "model", None),
            "baseline_score": baseline_score,
            "final_score": final_score,
            "improved": improved,
            "best_index": (
                getattr(detailed, "best_idx", None) if detailed is not None else None
            ),
            "timeline": [c.as_dict() for c in timeline],
            "instructions": chosen_instructions,
            "elapsed_seconds": elapsed,
        }

        evaluation = {
            "metric": self.metric_name,
            "train_size": len(self.train),
            "dev_size": len(self.dev),
            "total_samples": len(self.samples),
            "max_demos": 0,
            "baseline_score": baseline_score,
            "eval_score": chosen_score,
            "best": "gepa" if improved else "original",
            "improved": improved,
            "candidates": [
                {
                    "name": "original",
                    "score": baseline_score,
                    "demo_count": 0,
                    "bootstrapped_demos": 0,
                    "error": None,
                },
                {
                    "name": "gepa",
                    "score": final_score,
                    "demo_count": 0,
                    "bootstrapped_demos": 0,
                    "error": None,
                },
            ],
            "demos": [],
            "baseline_results": baseline_rows,
            "results": final_rows if improved else baseline_rows,
            "instructions": chosen_instructions,
        }

        return {
            "optimized_prompt": render_prompt(chosen_instructions, []),
            "instructions": chosen_instructions,
            "gepa": report,
            "eval": evaluation,
        }

    def _timeline(
        self, compiled: Any, original: str, tracker: GepaTracker
    ) -> list[GepaCandidate]:
        """Assemble the candidate lineage from GEPA's result object.

        Falls back to a two-entry timeline (original, evolved) when the
        detailed results are missing, so the client always has something.
        """
        detailed = getattr(compiled, "detailed_results", None)
        # GEPAResult exposes these as attributes; to_dict() reshapes them.
        candidates = list(getattr(detailed, "candidates", None) or [])
        parents = list(getattr(detailed, "parents", None) or [])
        scores = list(getattr(detailed, "val_aggregate_scores", None) or [])
        if not candidates:
            evolved = clean_instructions(
                str(getattr(compiled.signature, "instructions", "") or original)
            )
            return [
                GepaCandidate(0, None, 0, original, None),
                GepaCandidate(1, 0, 1, evolved, None),
            ]

        accepted = tracker.accepted_iterations()
        timeline: list[GepaCandidate] = []
        for index, candidate in enumerate(candidates):
            instructions: Any = candidate
            if isinstance(candidate, dict):
                instructions = next(iter(candidate.values()), "")
            elif hasattr(candidate, "signature"):  # a compiled dspy program
                instructions = getattr(candidate.signature, "instructions", "")
            elif hasattr(candidate, "predictors"):
                preds = list(candidate.predictors())
                instructions = (
                    getattr(preds[0].signature, "instructions", "") if preds else ""
                )
            parent_list = parents[index] if index < len(parents) else [None]
            parent = next((p for p in (parent_list or [None]) if p is not None), None)
            generation = 0 if parent is None else timeline[parent].generation + 1
            score = scores[index] if index < len(scores) else None
            # Candidate 0 is the seed; candidate k was accepted in the k-th
            # accepted iteration, whose reflection read that iteration's feedback.
            iteration = accepted[index - 1] if 0 < index <= len(accepted) else None
            feedback = (
                tracker.feedback_by_iteration.get(iteration, []) if iteration else []
            )
            timeline.append(
                GepaCandidate(
                    index=index,
                    parent=parent,
                    generation=generation,
                    instructions=clean_instructions(str(instructions)),
                    score=round(float(score) * 100, 2) if score is not None else None,
                    iteration=iteration,
                    feedback=feedback,
                )
            )
        return timeline


__all__ = [
    "SHORT_ANSWER_CHARS",
    "GepaOptimizer",
    "GepaTracker",
    "build_feedback_metric",
    "clean_instructions",
]
