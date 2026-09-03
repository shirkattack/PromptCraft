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


from app.services.eval_service import (  # noqa: E402
    describe_split,
    is_label_dataset,
    label_counts,
    make_folds,
    stratified_order,
)

LABELLED = [
    Sample("a1", "high"),
    Sample("a2", "high"),
    Sample("a3", "high"),
    Sample("a4", "high"),
    Sample("b1", "medium"),
    Sample("b2", "medium"),
    Sample("b3", "medium"),
    Sample("c1", "low"),
    Sample("c2", "low"),
    Sample("c3", "low"),
]


class TestStratification:
    def test_label_detection(self):
        assert is_label_dataset(LABELLED)
        assert not is_label_dataset([Sample("q", "A long free-text answer " * 5)] * 4)
        assert not is_label_dataset(
            [Sample("q", "same")] * 4
        )  # one class is not a label task
        assert not is_label_dataset(
            [Sample(f"q{i}", f"answer {i}") for i in range(6)]
        )  # all distinct

    def test_stratified_order_interleaves_classes(self):
        order, stratified = stratified_order(LABELLED, seed=1)
        assert stratified
        assert {s.input_text for s in order} == {s.input_text for s in LABELLED}
        first_three = [normalize(s.expected_output) for s in order[:3]]
        assert set(first_three) == {"high", "medium", "low"}

    def test_holdout_dev_covers_classes(self):
        train, dev = split_samples(LABELLED, 0.7, max_train=40, max_dev=20, seed=3)
        assert len(dev) == 3
        assert set(label_counts(dev)) == {"high", "medium", "low"}
        assert len(train) + len(dev) == len(LABELLED)

    def test_unstratified_split_is_plain_shuffle(self):
        train, dev = split_samples(LABELLED, 0.7, 40, 20, seed=3, stratify=False)
        assert len(train) + len(dev) == len(LABELLED)

    def test_folds_are_disjoint_balanced_and_cover_everything(self):
        folds = make_folds(LABELLED, folds=5, seed=2)
        assert len(folds) == 5
        seen = [s.input_text for fold in folds for s in fold]
        assert sorted(seen) == sorted(s.input_text for s in LABELLED)
        assert all(len(fold) == 2 for fold in folds)
        # Round-robin over a class-interleaved order keeps folds mixed.
        assert all(
            len({normalize(s.expected_output) for s in fold}) == 2 for fold in folds
        )

    def test_folds_capped_by_sample_count(self):
        assert len(make_folds(SAMPLES[:3], folds=5, seed=1)) == 3
        with pytest.raises(EvalError):
            make_folds(SAMPLES[:1], folds=5, seed=1)

    def test_describe_split(self):
        train, dev = split_samples(LABELLED, 0.7, 40, 20, seed=3)
        info = describe_split(LABELLED, train, dev, "holdout")
        assert info["strategy"] == "holdout" and info["stratified"] is True
        assert info["labels"] == {"high": 4, "low": 3, "medium": 3}
        assert info["dev_size"] == 3 and sum(info["dev_labels"].values()) == 3


KFOLD_ANSWERS = {
    "a1": {"output": "high"},
    "a2": {"output": "high"},
    "a3": {"output": "wrong"},
    "a4": {"output": "high"},
    "b1": {"output": "medium"},
    "b2": {"output": "nope"},
    "b3": {"output": "medium"},
    "c1": {"output": "low"},
    "c2": {"output": "low"},
    "c3": {"output": "low"},
}


class TestKFold:
    def test_every_sample_is_scored_once_and_winner_is_refit(self):
        updates = []

        def progress(stage, message="", *, current=None, total=None, best_score=None):
            updates.append((message, current, total))

        with dspy.context(lm=DummyLM(KFOLD_ANSWERS)):
            optimizer = DatasetOptimizer(
                LABELLED,
                metric="exact",
                max_demos=2,
                strategy="kfold",
                progress=progress,
            )
            report = optimizer.run("Classify.", "Classify the ticket.")

        assert report["split"]["strategy"] == "kfold"
        assert report["split"]["folds"] == 5
        assert report["dev_size"] == len(LABELLED)
        assert len(report["baseline_results"]) == len(LABELLED)
        assert len(report["results"]) == len(LABELLED)
        # 8 of 10 canned answers are right, whatever the prompt: 80% for every candidate.
        assert report["baseline_score"] == 80.0
        assert all(c["score"] == 80.0 for c in report["candidates"])
        assert report["best"] == "rewritten"  # tie -> simplest candidate
        assert report["optimized_prompt"].endswith("Input: {input}\nOutput:")
        assert updates[-1][0] == "Cross-validation complete"
        assert updates[0][2] == 5 * 4 + 1  # folds x candidates + final fit

    def test_kfold_refits_few_shot_winner_on_all_samples(self):
        # The rewrite is identical to the original, so only few-shot candidates compete.
        with dspy.context(lm=DummyLM(KFOLD_ANSWERS)):
            report = DatasetOptimizer(
                LABELLED, metric="exact", max_demos=3, strategy="kfold"
            ).run("Classify.", "Classify.")

        assert report["best"] == "original_fewshot"
        assert 1 <= len(report["demos"]) <= 3
        assert "Examples:" in report["optimized_prompt"]
        best = next(c for c in report["candidates"] if c["name"] == "original_fewshot")
        assert best["demo_count"] == len(report["demos"])

    def test_holdout_report_carries_split_info(self):
        with dspy.context(lm=DummyLM(KFOLD_ANSWERS)):
            report = DatasetOptimizer(LABELLED, metric="exact").run("Classify.", None)
        assert report["split"]["strategy"] == "holdout"
        assert report["split"]["stratified"] is True


class TestDemoCoverage:
    def test_more_validated_demos_than_kept_triggers_coverage_selection(self):
        # Every answer is right, so BootstrapFewShot validates the whole pool.
        answers = {s.input_text: {"output": s.expected_output} for s in LABELLED}
        with dspy.context(lm=DummyLM(answers)):
            optimizer = DatasetOptimizer(LABELLED, metric="exact", max_demos=3)
            report = optimizer.run("Classify.", None)

        assert report["best"] == "original_fewshot"
        assert len(report["demos"]) == 3
        selection = report["demo_selection"]
        assert selection["method"] == "coverage"
        assert selection["pool"] > 3 and selection["kept"] == 3
        assert selection["label_balanced"] is True
        # One example per class before any class repeats.
        assert {normalize(d["output"]) for d in report["demos"]} == {
            "high",
            "medium",
            "low",
        }
        assert all(isinstance(d["covers"], list) for d in report["demos"])

    def test_falls_back_to_bootstrap_order_without_embeddings(self, monkeypatch):
        from app.services import eval_service as module
        from app.services.embedding_service import EmbeddingUnavailable

        def boom(*args, **kwargs):
            raise EmbeddingUnavailable("no model")

        monkeypatch.setattr(module, "coverage_selection", boom)
        answers = {s.input_text: {"output": s.expected_output} for s in LABELLED}
        with dspy.context(lm=DummyLM(answers)):
            report = DatasetOptimizer(LABELLED, metric="exact", max_demos=2).run(
                "Classify.", None
            )

        assert len(report["demos"]) == 2
        assert report["demo_selection"]["method"] == "bootstrap"
        assert "no model" in report["demo_selection"]["reason"]
