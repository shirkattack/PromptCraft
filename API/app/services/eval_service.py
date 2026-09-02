"""Dataset-driven prompt optimization and evaluation.

Given a prompt, an optional rewrite of it and a dataset of (input, expected
output) pairs, this module:

1. splits the dataset into a train set and a held-out dev set,
2. builds a DSPy program per candidate prompt (the original, the rewrite, and
   few-shot versions of each compiled with ``BootstrapFewShot`` on the train
   set),
3. scores every candidate on the dev set with the chosen metric, and
4. returns the best candidate rendered as a plain-text prompt, with the full
   scoreboard and per-sample results so the client can show what was measured.

The score is the percentage of held-out samples the metric accepted. It is a
measurement of the prompt on this dataset, not a general quality claim.
"""

import logging
import random
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Literal

import dspy
from dspy.teleprompt import BootstrapFewShot

from app.core.config import settings

logger = logging.getLogger(__name__)

EvalMetric = Literal["auto", "exact", "contains", "llm_judge"]

# Expected outputs at or below this length are treated as labels (classes,
# short answers) where a string comparison is meaningful; longer ones are
# free text and need a judge.
SHORT_ANSWER_CHARS = 40

# Candidate names, in the order they are tried and the order ties are broken:
# an earlier candidate wins a tie. The zero-shot rewrite comes first because it
# is the simplest artifact to hand back.
CANDIDATE_ORDER = ["rewritten", "rewritten_fewshot", "original_fewshot"]

BASELINE = "original"


class EvalError(Exception):
    """Raised when a dataset cannot be used for evaluation."""


@dataclass
class Sample:
    input_text: str
    expected_output: str

    def to_example(self) -> dspy.Example:
        return dspy.Example(
            input=self.input_text, output=self.expected_output
        ).with_inputs("input")


@dataclass
class Candidate:
    name: str
    instructions: str
    program: dspy.Module
    demos: list[dict[str, str]] = field(default_factory=list)
    score: float | None = None
    results: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "demo_count": len(self.demos),
            "bootstrapped_demos": sum(1 for d in self.demos if d.get("bootstrapped")),
            "error": self.error,
        }


def normalize(text: Any) -> str:
    """Lower-case, collapse whitespace and strip surrounding punctuation."""
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .!?:;,\"'`*_")


def exact_metric(example: dspy.Example, pred: Any, trace: Any = None) -> bool:
    return normalize(getattr(pred, "output", "")) == normalize(example.output)


def contains_metric(example: dspy.Example, pred: Any, trace: Any = None) -> bool:
    expected = normalize(example.output)
    actual = normalize(getattr(pred, "output", ""))
    if not expected or not actual:
        return False
    # Either direction, but a very short answer must not pass just because
    # its letters happen to appear inside the expected text.
    return expected in actual or (len(actual) >= 3 and actual in expected)


class JudgeOutput(dspy.Signature):
    """Decide whether a response satisfies the expected output for an input.

    Judge meaning, not wording: a response that conveys the same content as the
    expected output in different words is correct. A response that contradicts
    it, omits its key content or answers a different question is incorrect.
    """

    input = dspy.InputField(desc="The input the response was written for")
    expected_output = dspy.InputField(
        desc="A reference answer that is known to be correct"
    )
    actual_output = dspy.InputField(desc="The response being judged")
    verdict = dspy.OutputField(desc="'yes' if the response is correct, otherwise 'no'")


def make_judge_metric():
    judge = dspy.Predict(JudgeOutput)

    def llm_judge_metric(example: dspy.Example, pred: Any, trace: Any = None) -> bool:
        actual = str(getattr(pred, "output", "") or "").strip()
        if not actual:
            return False
        verdict = judge(
            input=example.input, expected_output=example.output, actual_output=actual
        )
        return normalize(verdict.verdict).startswith("yes")

    return llm_judge_metric


def choose_metric(metric: EvalMetric, samples: list[Sample]) -> str:
    if metric != "auto":
        return metric
    median_len = statistics.median(len(s.expected_output.strip()) for s in samples)
    return "contains" if median_len <= SHORT_ANSWER_CHARS else "llm_judge"


def split_samples(
    samples: list[Sample],
    train_ratio: float,
    max_train: int,
    max_dev: int,
    seed: int,
) -> tuple[list[Sample], list[Sample]]:
    """Shuffle deterministically and split; both halves are non-empty."""
    if len(samples) < 2:
        raise EvalError(
            "At least 2 samples are needed: one to learn from, one to test on"
        )

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)

    dev_size = max(1, round(len(shuffled) * (1 - train_ratio)))
    dev_size = min(dev_size, max_dev, len(shuffled) - 1)
    dev = shuffled[:dev_size]
    train = shuffled[dev_size : dev_size + max_train]
    return train, dev


def render_prompt(instructions: str, demos: list[dict[str, str]]) -> str:
    """Turn instructions plus demos into a copy-pasteable prompt.

    ``{input}`` marks where the caller's data goes. This is a rendering of the
    DSPy program that was measured, not the exact bytes it sent: DSPy wraps the
    same instructions and demos in its own field markup.
    """
    parts = [instructions.strip()]
    if demos:
        parts.append("Examples:")
        for demo in demos:
            parts.append(f"Input: {demo['input']}\nOutput: {demo['output']}")
    parts.append("Input: {input}\nOutput:")
    return "\n\n".join(parts)


class DatasetOptimizer:
    """Compile and evaluate prompt candidates against a dataset.

    Must be called inside ``dspy.context(lm=...)`` on the thread that runs it.
    """

    def __init__(
        self,
        samples: list[Sample],
        metric: EvalMetric = "auto",
        max_demos: int = 4,
        train_ratio: float | None = None,
        seed: int = 13,
    ) -> None:
        if not samples:
            raise EvalError("The dataset has no samples")
        self.samples = samples
        self.metric_name = choose_metric(metric, samples)
        self.max_demos = max(1, min(max_demos, settings.eval_max_demos))
        self.train, self.dev = split_samples(
            samples,
            settings.default_train_ratio if train_ratio is None else train_ratio,
            settings.eval_max_train_samples,
            settings.eval_max_dev_samples,
            seed,
        )
        self.metric = self._build_metric()

    def _build_metric(self):
        if self.metric_name == "exact":
            return exact_metric
        if self.metric_name == "contains":
            return contains_metric
        if self.metric_name == "llm_judge":
            return make_judge_metric()
        raise EvalError(f"Unknown metric: {self.metric_name}")

    # -- candidates -------------------------------------------------------

    @staticmethod
    def _program(instructions: str) -> dspy.Predict:
        signature = dspy.Signature("input -> output", instructions.strip())
        return dspy.Predict(signature)

    def _compile(self, instructions: str) -> tuple[dspy.Predict, list[dict[str, str]]]:
        """Bootstrap few-shot demos for ``instructions`` on the train split."""
        demo_cap = min(self.max_demos, len(self.train))
        teleprompter = BootstrapFewShot(
            metric=self.metric,
            max_bootstrapped_demos=demo_cap,
            max_labeled_demos=demo_cap,
            max_rounds=1,
        )
        compiled = teleprompter.compile(
            self._program(instructions),
            trainset=[s.to_example() for s in self.train],
        )
        demos = [
            {
                "input": str(demo.input),
                "output": str(demo.output),
                "bootstrapped": bool(getattr(demo, "augmented", False)),
            }
            for demo in compiled.demos
        ]
        return compiled, demos

    def _evaluate(self, candidate: Candidate) -> None:
        evaluator = dspy.Evaluate(
            devset=[s.to_example() for s in self.dev],
            metric=self.metric,
            num_threads=1,  # one local model; parallel calls just queue up
            display_progress=False,
            display_table=False,
            provide_traceback=False,
        )
        outcome = evaluator(candidate.program)
        candidate.score = round(float(outcome.score), 2)
        candidate.results = [
            {
                "input": str(example.input),
                "expected": str(example.output),
                "actual": str(getattr(prediction, "output", "") or ""),
                "passed": bool(score),
            }
            for example, prediction, score in outcome.results
        ]

    def run(self, original: str, rewritten: str | None = None) -> dict[str, Any]:
        """Measure the original prompt and every candidate; return the report."""
        baseline = Candidate(BASELINE, original, self._program(original))
        self._evaluate(baseline)

        candidates: list[Candidate] = []

        def add(name: str, instructions: str, few_shot: bool) -> None:
            try:
                if few_shot:
                    program, demos = self._compile(instructions)
                    candidate = Candidate(name, instructions, program, demos)
                else:
                    candidate = Candidate(
                        name, instructions, self._program(instructions)
                    )
                self._evaluate(candidate)
            except Exception as exc:  # one bad candidate must not sink the run
                logger.warning(f"Candidate {name} failed: {exc}")
                candidate = Candidate(
                    name, instructions, dspy.Predict("input -> output")
                )
                candidate.error = str(exc)
            candidates.append(candidate)

        has_rewrite = bool(rewritten and rewritten.strip() != original.strip())
        if has_rewrite:
            add("rewritten", rewritten, few_shot=False)  # type: ignore[arg-type]
            add("rewritten_fewshot", rewritten, few_shot=True)  # type: ignore[arg-type]
        add("original_fewshot", original, few_shot=True)

        best = self.pick_best(candidates)
        if best is None:
            raise EvalError(
                "Every candidate failed to evaluate: "
                + "; ".join(f"{c.name}: {c.error}" for c in candidates)
            )

        return {
            "metric": self.metric_name,
            "train_size": len(self.train),
            "dev_size": len(self.dev),
            "total_samples": len(self.samples),
            "max_demos": self.max_demos,
            "baseline_score": baseline.score,
            "eval_score": best.score,
            "best": best.name,
            "improved": best.score is not None
            and baseline.score is not None
            and best.score > baseline.score,
            "candidates": [baseline.summary()] + [c.summary() for c in candidates],
            "demos": best.demos,
            "baseline_results": baseline.results,
            "results": best.results,
            "optimized_prompt": render_prompt(best.instructions, best.demos),
            "instructions": best.instructions,
        }

    @staticmethod
    def pick_best(candidates: list[Candidate]) -> Candidate | None:
        """Highest score wins; ties go to the earliest name in CANDIDATE_ORDER."""
        scored = [c for c in candidates if c.score is not None]
        if not scored:
            return None
        return max(
            scored,
            key=lambda c: (c.score, -CANDIDATE_ORDER.index(c.name)),
        )
