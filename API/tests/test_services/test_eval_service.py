"""Tests for dataset-driven optimization and evaluation."""

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from app.services.eval_service import (
    CANDIDATE_ORDER,
    Candidate,
    DatasetOptimizer,
    EvalError,
    Sample,
    choose_metric,
    contains_metric,
    exact_metric,
    make_judge_metric,
    normalize,
    render_prompt,
    split_samples,
)


def _pred(output: str) -> dspy.Prediction:
    return dspy.Prediction(output=output)


def _example(expected: str) -> dspy.Example:
    return dspy.Example(input="x", output=expected).with_inputs("input")


SAMPLES = [
    Sample("alpha ticket", "high"),
    Sample("beta ticket", "low"),
    Sample("gamma ticket", "high"),
    Sample("delta ticket", "medium"),
    Sample("epsilon ticket", "low"),
]

# Dict-mode DummyLM answers by the key found in the final message, so each
# sample input maps to a fixed answer regardless of instructions or demos.
ANSWERS = {
    "alpha ticket": {"output": "High"},
    "beta ticket": {"output": "low."},
    "gamma ticket": {"output": "This is high priority"},
    "delta ticket": {"output": "urgent"},  # wrong
    "epsilon ticket": {"output": "low"},
}


class TestMetrics:
    def test_normalize_ignores_case_whitespace_and_punctuation(self):
        assert normalize("  High.  ") == "high"
        assert normalize("**Medium**") == "medium"
        assert normalize("a   b\nc") == "a b c"

    def test_exact(self):
        assert exact_metric(_example("high"), _pred("High."))
        assert not exact_metric(_example("high"), _pred("high priority"))

    def test_contains_either_direction_but_not_trivially(self):
        assert contains_metric(_example("high"), _pred("This is high priority"))
        assert contains_metric(_example("high priority"), _pred("high"))
        assert not contains_metric(_example("high"), _pred("h"))
        assert not contains_metric(_example("high"), _pred(""))

    def test_auto_picks_string_metric_for_labels_and_judge_for_prose(self):
        assert choose_metric("auto", SAMPLES) == "contains"
        prose = [Sample("q", "A long free-text answer " * 5)] * 3
        assert choose_metric("auto", prose) == "llm_judge"
        assert choose_metric("exact", prose) == "exact"

    def test_judge_metric_reads_the_verdict(self):
        with dspy.context(lm=DummyLM([{"verdict": "Yes"}, {"verdict": "no"}])):
            metric = make_judge_metric()
            assert metric(_example("Paris"), _pred("The capital is Paris")) is True
            assert metric(_example("Paris"), _pred("Berlin")) is False
            # An empty answer never reaches the judge.
            assert metric(_example("Paris"), _pred("")) is False


class TestSplit:
    def test_both_halves_non_empty_and_deterministic(self):
        train, dev = split_samples(SAMPLES, 0.8, max_train=40, max_dev=20, seed=1)
        assert len(train) == 4 and len(dev) == 1
        assert {s.input_text for s in train + dev} == {s.input_text for s in SAMPLES}
        again = split_samples(SAMPLES, 0.8, max_train=40, max_dev=20, seed=1)
        assert [s.input_text for s in again[1]] == [s.input_text for s in dev]

    def test_caps_apply(self):
        many = [Sample(f"in {i}", "out") for i in range(100)]
        train, dev = split_samples(many, 0.8, max_train=10, max_dev=5, seed=1)
        assert len(train) == 10 and len(dev) == 5

    def test_two_samples_is_the_minimum(self):
        train, dev = split_samples(SAMPLES[:2], 0.8, max_train=40, max_dev=20, seed=1)
        assert len(train) == 1 and len(dev) == 1
        with pytest.raises(EvalError):
            split_samples(SAMPLES[:1], 0.8, max_train=40, max_dev=20, seed=1)


class TestRendering:
    def test_prompt_with_demos(self):
        text = render_prompt("Classify.", [{"input": "a", "output": "b"}])
        assert text.startswith("Classify.\n\nExamples:\n\nInput: a\nOutput: b")
        assert text.endswith("Input: {input}\nOutput:")

    def test_prompt_without_demos_has_no_examples_section(self):
        text = render_prompt("Classify.", [])
        assert "Examples" not in text
        assert text == "Classify.\n\nInput: {input}\nOutput:"


class TestPickBest:
    def _candidate(self, name, score):
        c = Candidate(name, "i", dspy.Predict("input -> output"))
        c.score = score
        return c

    def test_highest_score_wins(self):
        best = DatasetOptimizer.pick_best(
            [
                self._candidate("rewritten", 50.0),
                self._candidate("original_fewshot", 75.0),
            ]
        )
        assert best.name == "original_fewshot"

    def test_ties_go_to_the_simplest_candidate(self):
        candidates = [self._candidate(name, 60.0) for name in reversed(CANDIDATE_ORDER)]
        assert DatasetOptimizer.pick_best(candidates).name == "rewritten"

    def test_all_failed(self):
        assert DatasetOptimizer.pick_best([self._candidate("rewritten", None)]) is None


class TestDatasetOptimizer:
    def test_run_reports_every_candidate(self):
        with dspy.context(lm=DummyLM(ANSWERS)):
            optimizer = DatasetOptimizer(SAMPLES, metric="contains", max_demos=2)
            report = optimizer.run(
                "Classify priority.", "Classify the ticket priority."
            )

        assert report["metric"] == "contains"
        assert report["train_size"] == 4 and report["dev_size"] == 1
        assert [c["name"] for c in report["candidates"]] == [
            "original",
            "rewritten",
            "rewritten_fewshot",
            "original_fewshot",
        ]
        assert all(c["error"] is None for c in report["candidates"])
        assert all(0.0 <= c["score"] <= 100.0 for c in report["candidates"])
        assert report["best"] in CANDIDATE_ORDER
        assert len(report["results"]) == 1
        assert {"input", "expected", "actual", "passed"} <= report["results"][0].keys()
        assert report["optimized_prompt"].endswith("Input: {input}\nOutput:")
        assert len(report["demos"]) <= 2

    def test_run_without_a_rewrite_only_tries_few_shot_original(self):
        with dspy.context(lm=DummyLM(ANSWERS)):
            report = DatasetOptimizer(SAMPLES, metric="exact").run("Classify.", None)

        assert [c["name"] for c in report["candidates"]] == [
            "original",
            "original_fewshot",
        ]
        assert report["best"] == "original_fewshot"
        assert report["instructions"] == "Classify."

    def test_identical_rewrite_is_not_a_candidate(self):
        with dspy.context(lm=DummyLM(ANSWERS)):
            report = DatasetOptimizer(SAMPLES, metric="exact").run(
                "Classify.", "Classify."
            )

        assert "rewritten" not in [c["name"] for c in report["candidates"]]

    def test_max_demos_is_capped_by_settings(self):
        optimizer = DatasetOptimizer(SAMPLES, max_demos=1000)
        assert optimizer.max_demos <= 8

    def test_empty_dataset_rejected(self):
        with pytest.raises(EvalError):
            DatasetOptimizer([])
