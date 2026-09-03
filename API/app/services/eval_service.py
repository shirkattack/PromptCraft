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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import dspy
from dspy.teleprompt import BootstrapFewShot

from app.core.config import settings
from app.services.embedding_service import EmbeddingUnavailable, coverage_selection
from app.services.progress import ProgressCallback, no_progress

logger = logging.getLogger(__name__)

EvalMetric = Literal["auto", "exact", "contains", "llm_judge"]
EvalStrategy = Literal["holdout", "kfold"]

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
    demos: list[dict[str, Any]] = field(default_factory=list)
    score: float | None = None
    results: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    selection: dict[str, Any] | None = None

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
    normalized: str = str(text or "").strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" .!?:;,\"'`*_")


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


MetricFn = Callable[[dspy.Example, Any, Any], bool]


def make_judge_metric() -> MetricFn:
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


def is_label_dataset(samples: list[Sample]) -> bool:
    """True when expected outputs are a small set of short labels (classes)."""
    outputs = [normalize(s.expected_output) for s in samples]
    if not outputs or any(len(o) > SHORT_ANSWER_CHARS for o in outputs):
        return False
    distinct = len(set(outputs))
    return 1 < distinct <= max(2, len(outputs) // 2)


def label_counts(samples: list[Sample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        key = normalize(sample.expected_output)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def stratified_order(samples: list[Sample], seed: int) -> tuple[list[Sample], bool]:
    """Deterministic shuffle whose every prefix is as class-balanced as possible.

    For label datasets the samples are shuffled within each class and then
    interleaved round-robin, so taking the first k gives each class a turn
    before any class repeats. Other datasets get a plain shuffle. Returns the
    order and whether stratification applied.
    """
    rng = random.Random(seed)
    if not is_label_dataset(samples):
        shuffled = list(samples)
        rng.shuffle(shuffled)
        return shuffled, False

    groups: dict[str, list[Sample]] = {}
    for sample in samples:
        groups.setdefault(normalize(sample.expected_output), []).append(sample)
    # Largest classes first so a tiny dev set still sees the common labels;
    # shuffle within each class for variety across seeds.
    ordered_groups = sorted(groups.values(), key=len, reverse=True)
    for group in ordered_groups:
        rng.shuffle(group)

    order: list[Sample] = []
    position = 0
    while len(order) < len(samples):
        for group in ordered_groups:
            if position < len(group):
                order.append(group[position])
        position += 1
    return order, True


def split_samples(
    samples: list[Sample],
    train_ratio: float,
    max_train: int,
    max_dev: int,
    seed: int,
    stratify: bool = True,
) -> tuple[list[Sample], list[Sample]]:
    """Deterministic hold-out split; both halves are non-empty.

    With ``stratify`` (the default) label datasets get a dev set that covers
    the classes instead of, say, two samples of the same label.
    """
    if len(samples) < 2:
        raise EvalError(
            "At least 2 samples are needed: one to learn from, one to test on"
        )

    if stratify:
        ordered, _ = stratified_order(samples, seed)
    else:
        ordered = list(samples)
        random.Random(seed).shuffle(ordered)

    dev_size = max(1, round(len(ordered) * (1 - train_ratio)))
    dev_size = min(dev_size, max_dev, len(ordered) - 1)
    dev = ordered[:dev_size]
    train = ordered[dev_size : dev_size + max_train]
    return train, dev


def make_folds(samples: list[Sample], folds: int, seed: int) -> list[list[Sample]]:
    """Split into ``folds`` disjoint groups, class-balanced for label datasets."""
    if len(samples) < 2:
        raise EvalError("At least 2 samples are needed for k-fold evaluation")
    folds = max(2, min(folds, len(samples)))
    ordered, _ = stratified_order(samples, seed)
    return [ordered[i::folds] for i in range(folds)]


def describe_split(
    samples: list[Sample],
    train: list[Sample],
    dev: list[Sample],
    strategy: str,
    folds: int | None = None,
) -> dict[str, Any]:
    """What the client needs to explain how the score was obtained."""
    stratified = is_label_dataset(samples)
    return {
        "strategy": strategy,
        "folds": folds,
        "stratified": stratified,
        "labels": label_counts(samples) if stratified else None,
        "dev_labels": label_counts(dev) if stratified else None,
        "train_size": len(train),
        "dev_size": len(dev),
    }


def render_prompt(instructions: str, demos: list[dict[str, Any]]) -> str:
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
        progress: ProgressCallback = no_progress,
        strategy: EvalStrategy = "holdout",
    ) -> None:
        if not samples:
            raise EvalError("The dataset has no samples")
        self.samples = samples
        self.progress = progress
        self.seed = seed
        self.strategy: EvalStrategy = strategy
        self.metric_name = choose_metric(metric, samples)
        self.max_demos = max(1, min(max_demos, settings.eval_max_demos))
        self.train, self.dev = split_samples(
            samples,
            settings.default_train_ratio if train_ratio is None else train_ratio,
            settings.eval_max_train_samples,
            settings.eval_max_dev_samples,
            seed,
        )
        self.folds: list[list[Sample]] = (
            make_folds(samples, settings.eval_max_folds, seed)
            if strategy == "kfold"
            else []
        )
        self.metric = self._build_metric()
        self.last_selection: dict[str, Any] | None = None

    def _build_metric(self) -> MetricFn:
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

    def _compile(
        self, instructions: str, train: list[Sample] | None = None
    ) -> tuple[dspy.Predict, list[dict[str, Any]]]:
        """Bootstrap few-shot demos for ``instructions`` on a train split."""
        train = self.train if train is None else train
        demo_cap = min(self.max_demos, len(train))
        # Validate a larger pool than we keep, then choose the demos that
        # best cover the inputs rather than the first that passed.
        pool_cap = min(max(demo_cap, settings.eval_demo_pool), len(train))
        teleprompter = BootstrapFewShot(
            metric=self.metric,
            max_bootstrapped_demos=pool_cap,
            max_labeled_demos=pool_cap,
            max_rounds=1,
        )
        compiled = teleprompter.compile(
            self._program(instructions),
            trainset=[s.to_example() for s in train],
        )
        pool = list(compiled.demos)
        chosen, covers, selection = self._select_demos(pool, train, demo_cap)
        compiled.demos = [pool[i] for i in chosen]
        self.last_selection = selection
        demos = [
            {
                "input": str(pool[i].input),
                "output": str(pool[i].output),
                "bootstrapped": bool(getattr(pool[i], "augmented", False)),
                "covers": covers.get(i, []),
            }
            for i in chosen
        ]
        return compiled, demos

    def _select_demos(
        self, pool: list[Any], train: list[Sample], k: int
    ) -> tuple[list[int], dict[int, list[str]], dict[str, Any]]:
        """Pick ``k`` demos from the validated pool by input coverage."""
        if len(pool) <= k:
            return (
                list(range(len(pool))),
                {},
                {"method": "bootstrap", "pool": len(pool), "kept": len(pool)},
            )
        labels = (
            [normalize(str(demo.output)) for demo in pool]
            if is_label_dataset(train)
            else None
        )
        try:
            chosen, covers = coverage_selection(
                [str(demo.input) for demo in pool],
                [s.input_text for s in train],
                k,
                labels=labels,
            )
        except EmbeddingUnavailable as exc:
            logger.warning(
                f"Coverage selection unavailable, keeping first demos: {exc}"
            )
            return (
                list(range(k)),
                {},
                {
                    "method": "bootstrap",
                    "pool": len(pool),
                    "kept": k,
                    "reason": str(exc),
                },
            )
        return (
            chosen,
            covers,
            {
                "method": "coverage",
                "pool": len(pool),
                "kept": len(chosen),
                "embedding_model": settings.embedding_model,
                "label_balanced": labels is not None,
            },
        )

    def _evaluate(self, candidate: Candidate, dev: list[Sample] | None = None) -> None:
        dev = self.dev if dev is None else dev
        evaluator = dspy.Evaluate(
            devset=[s.to_example() for s in dev],
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
        if self.strategy == "kfold":
            return self._run_kfold(original, rewritten)

        has_rewrite = bool(rewritten and rewritten.strip() != original.strip())
        total_steps = 1 + (3 if has_rewrite else 1)
        step = 0

        def report(message: str, best: float | None = None) -> None:
            self.progress(
                "evaluate", message, current=step, total=total_steps, best_score=best
            )

        report(f"Scoring the original prompt on {len(self.dev)} held-out samples")
        baseline = Candidate(BASELINE, original, self._program(original))
        self._evaluate(baseline)
        step += 1
        report("Original prompt scored", baseline.score)

        candidates: list[Candidate] = []

        def add(name: str, instructions: str, few_shot: bool) -> None:
            nonlocal step
            label = name.replace("_", " ")
            try:
                if few_shot:
                    report(
                        f"Choosing examples for '{label}' from {len(self.train)} train samples"
                    )
                    program, demos = self._compile(instructions)
                    candidate = Candidate(name, instructions, program, demos)
                    candidate.selection = self.last_selection
                else:
                    candidate = Candidate(
                        name, instructions, self._program(instructions)
                    )
                report(f"Scoring '{label}' on {len(self.dev)} held-out samples")
                self._evaluate(candidate)
            except Exception as exc:  # one bad candidate must not sink the run
                logger.warning(f"Candidate {name} failed: {exc}")
                candidate = Candidate(
                    name, instructions, dspy.Predict("input -> output")
                )
                candidate.error = str(exc)
            candidates.append(candidate)
            step += 1
            best = max(
                (c.score for c in candidates if c.score is not None), default=None
            )
            report(f"'{label}' scored", best)

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
            "split": describe_split(self.samples, self.train, self.dev, "holdout"),
            "demo_selection": best.selection,
        }

    def _run_kfold(self, original: str, rewritten: str | None) -> dict[str, Any]:
        """Cross-validate every candidate type, then fit the winner on all samples.

        Each fold is held out once, so every sample is scored and the score
        moves in steps of 1/N instead of 1/dev_size. The returned prompt is the
        winning candidate type recompiled on the whole dataset.
        """
        has_rewrite = bool(rewritten and rewritten.strip() != original.strip())
        names = [BASELINE]
        if has_rewrite:
            names += ["rewritten", "rewritten_fewshot"]
        names.append("original_fewshot")
        instructions_for = {
            BASELINE: original,
            "rewritten": rewritten or original,
            "rewritten_fewshot": rewritten or original,
            "original_fewshot": original,
        }
        few_shot = {"rewritten_fewshot", "original_fewshot"}

        tally: dict[str, dict[str, Any]] = {
            name: {"passed": 0, "total": 0, "results": [], "errors": [], "demos": 0}
            for name in names
        }
        total_steps = len(self.folds) * len(names) + 1
        step = 0

        def report(message: str, best: float | None = None) -> None:
            self.progress(
                "evaluate", message, current=step, total=total_steps, best_score=best
            )

        def running_best() -> float | None:
            scores = [
                t["passed"] / t["total"] * 100
                for name, t in tally.items()
                if name != BASELINE and t["total"]
            ]
            return round(max(scores), 2) if scores else None

        for index, dev in enumerate(self.folds, start=1):
            train = [s for fold in self.folds if fold is not dev for s in fold][
                : settings.eval_max_train_samples
            ]
            for name in names:
                label = name.replace("_", " ")
                report(
                    f"Fold {index}/{len(self.folds)}: scoring '{label}' on {len(dev)} samples"
                )
                try:
                    if name in few_shot:
                        program, demos = self._compile(instructions_for[name], train)
                        candidate = Candidate(
                            name, instructions_for[name], program, demos
                        )
                    else:
                        candidate = Candidate(
                            name,
                            instructions_for[name],
                            self._program(instructions_for[name]),
                        )
                    self._evaluate(candidate, dev)
                    tally[name]["passed"] += sum(
                        1 for r in candidate.results if r["passed"]
                    )
                    tally[name]["total"] += len(candidate.results)
                    tally[name]["results"].extend(candidate.results)
                    tally[name]["demos"] = max(
                        tally[name]["demos"], len(candidate.demos)
                    )
                except Exception as exc:  # a failed fold counts as zero for it
                    logger.warning(f"Fold {index} candidate {name} failed: {exc}")
                    tally[name]["errors"].append(f"fold {index}: {exc}")
                    tally[name]["total"] += len(dev)
                step += 1
            report(f"Fold {index}/{len(self.folds)} done", running_best())

        def score_of(name: str) -> float | None:
            t = tally[name]
            return round(t["passed"] / t["total"] * 100, 2) if t["total"] else None

        scored: list[Candidate] = []
        for name in names:
            candidate = Candidate(
                name, instructions_for[name], dspy.Predict("input -> output")
            )
            candidate.score = score_of(name)
            candidate.results = tally[name]["results"]
            candidate.error = "; ".join(tally[name]["errors"]) or None
            scored.append(candidate)
        baseline = scored[0]
        best = self.pick_best(scored[1:])
        if best is None:
            raise EvalError(
                "Every candidate failed to evaluate: "
                + "; ".join(f"{c.name}: {c.error}" for c in scored[1:])
            )

        # Fit the winning candidate type on everything for the returned prompt.
        final_demos: list[dict[str, Any]] = []
        final_selection: dict[str, Any] | None = None
        if best.name in few_shot:
            report(
                f"Choosing examples for '{best.name.replace('_', ' ')}' from all {len(self.samples)} samples"
            )
            try:
                _, final_demos = self._compile(
                    best.instructions, self.samples[: settings.eval_max_train_samples]
                )
                final_selection = self.last_selection
            except Exception as exc:
                logger.warning(
                    f"Final compile failed, returning instructions only: {exc}"
                )
        step += 1
        report("Cross-validation complete", best.score)

        summaries = []
        for candidate in scored:
            summary = candidate.summary()
            summary["demo_count"] = (
                len(final_demos)
                if candidate is best
                else tally[candidate.name]["demos"]
            )
            summary["bootstrapped_demos"] = (
                sum(1 for d in final_demos if d.get("bootstrapped"))
                if candidate is best
                else 0
            )
            summaries.append(summary)

        fold_size = len(self.folds[0]) if self.folds else 0
        return {
            "metric": self.metric_name,
            "train_size": len(self.samples) - fold_size,
            "dev_size": len(self.samples),
            "total_samples": len(self.samples),
            "max_demos": self.max_demos,
            "baseline_score": baseline.score,
            "eval_score": best.score,
            "best": best.name,
            "improved": best.score is not None
            and baseline.score is not None
            and best.score > baseline.score,
            "candidates": summaries,
            "demos": final_demos,
            "baseline_results": baseline.results,
            "results": best.results,
            "optimized_prompt": render_prompt(best.instructions, final_demos),
            "instructions": best.instructions,
            "demo_selection": final_selection,
            "split": describe_split(
                self.samples,
                self.samples[fold_size:],  # a typical fold's training share
                self.samples,
                "kfold",
                len(self.folds),
            ),
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
