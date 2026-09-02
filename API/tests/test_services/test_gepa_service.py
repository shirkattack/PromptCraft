"""Tests for the GEPA wrapper: feedback metrics, clean-up, tracking, reporting."""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from app.services.eval_service import EvalError, Sample
from app.services.gepa_service import (
    GEPA_LOGGER_NAME,
    GepaOptimizer,
    GepaTracker,
    build_feedback_metric,
    clean_instructions,
)


def _gold(inp: str, out: str) -> dspy.Example:
    return dspy.Example(input=inp, output=out).with_inputs("input")


class TestFeedbackMetrics:
    def test_exact_label(self):
        metric = build_feedback_metric("contains")
        good = metric(_gold("x", "high"), dspy.Prediction(output="High."))
        assert good.score == 1.0 and "Correct" in good.feedback

    def test_buried_label_gets_half_credit_and_actionable_feedback(self):
        metric = build_feedback_metric("contains")
        verdict = metric(
            _gold("x", "high"),
            dspy.Prediction(output="This is high priority because the server is down"),
        )
        assert verdict.score == 0.5
        assert "buried" in verdict.feedback
        assert "'high' alone" in verdict.feedback

    def test_exact_metric_gives_no_partial_credit(self):
        metric = build_feedback_metric("exact")
        verdict = metric(_gold("x", "high"), dspy.Prediction(output="high priority"))
        assert verdict.score == 0.0
        assert "Expected 'high'" in verdict.feedback

    def test_wrong_answer_feedback_names_expected_and_input(self):
        metric = build_feedback_metric("contains")
        verdict = metric(_gold("Server down", "high"), dspy.Prediction(output="low"))
        assert verdict.score == 0.0
        assert "Expected 'high'" in verdict.feedback
        assert "Server down" in verdict.feedback

    def test_judge_metric_uses_reason_as_feedback(self):
        with dspy.context(
            lm=DummyLM([{"verdict": "no", "reason": "It omits the deadline."}])
        ):
            metric = build_feedback_metric("llm_judge")
            verdict = metric(_gold("q", "Friday"), dspy.Prediction(output="Soon"))
        assert verdict.score == 0.0
        assert verdict.feedback == "Incorrect: It omits the deadline."

    def test_unknown_metric(self):
        with pytest.raises(EvalError):
            build_feedback_metric("nope")


class TestCleanInstructions:
    def test_strips_python_comment_block(self):
        raw = (
            "python\n# Classify the ticket.\n#\n#   - high: outages\n#   - low: praise"
        )
        assert (
            clean_instructions(raw)
            == "Classify the ticket.\n\n  - high: outages\n  - low: praise"
        )

    def test_strips_code_fence(self):
        assert clean_instructions("```text\nDo the thing.\n```") == "Do the thing."

    def test_plain_text_untouched(self):
        text = "Classify it.\n# Not all lines are comments"
        assert clean_instructions(text) == text


class TestTracker:
    def _emit(self, tracker: GepaTracker, message: str) -> None:
        record = logging.LogRecord(
            GEPA_LOGGER_NAME, logging.INFO, "", 0, message, None, None
        )
        tracker.emit(record)

    def test_parses_iterations_into_progress_and_events(self):
        updates = []

        def progress(stage, message="", *, current=None, total=None, best_score=None):
            updates.append((stage, message, current, best_score))

        tracker = GepaTracker(progress)
        self._emit(tracker, "Iteration 0: Base program full valset score: 0.3333")
        self._emit(tracker, "Iteration 1: Selected program 0 score: 0.3333")
        tracker.record_feedback("Wrong. Expected 'high'", 0.0)
        tracker.record_feedback("Correct", 1.0)
        self._emit(
            tracker, "Iteration 1: Proposed new text for self: python\n# Classify"
        )
        self._emit(tracker, "Iteration 1: Full valset score for new program: 0.6667")
        self._emit(tracker, "Iteration 1: Best valset aggregate score so far: 0.6667")
        self._emit(tracker, "Iteration 1: New program candidate index: 1")
        self._emit(tracker, "Iteration 2: New subsample score is not better, skipping")
        self._emit(tracker, "not an iteration line")

        assert tracker.accepted_iterations() == [1]
        assert tracker.feedback_by_iteration == {1: ["Wrong. Expected 'high'"]}
        assert tracker.best_score == pytest.approx(0.6667)
        kinds = [e.kind for e in tracker.events]
        assert kinds == [
            "other",
            "selected",
            "proposed",
            "scored",
            "other",
            "accepted",
            "skipped",
        ]
        assert updates[0][0] == "evolve"
        assert updates[-1][2] == 2  # current iteration
        assert updates[-1][3] == pytest.approx(66.67, abs=0.1)


SAMPLES = [
    Sample("alpha ticket", "high"),
    Sample("beta ticket", "low"),
    Sample("gamma ticket", "high"),
    Sample("delta ticket", "medium"),
    Sample("epsilon ticket", "low"),
]

ANSWERS = {
    "alpha ticket": {"output": "high"},
    "beta ticket": {"output": "low"},
    "gamma ticket": {"output": "high"},
    "delta ticket": {"output": "urgent"},
    "epsilon ticket": {"output": "This is low priority"},
}


class FakeGEPA:
    """Stands in for dspy.teleprompt.GEPA: returns an evolved program with lineage."""

    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        FakeGEPA.last_kwargs = kwargs
        self.metric = kwargs["metric"]

    def compile(self, student, *, trainset, valset):
        # Exercise the metric so feedback is recorded, like the real optimizer.
        gepa_logger = logging.getLogger(GEPA_LOGGER_NAME)
        gepa_logger.info("Iteration 0: Base program full valset score: 0.5")
        gepa_logger.info("Iteration 1: Selected program 0 score: 0.5")
        for example in trainset[:2]:
            self.metric(example, dspy.Prediction(output="urgent"))
        gepa_logger.info(
            "Iteration 1: Proposed new text for self: python\n# Reply with the label only."
        )
        gepa_logger.info("Iteration 1: Full valset score for new program: 1.0")
        gepa_logger.info("Iteration 1: New program candidate index: 1")
        program = dspy.Predict(
            dspy.Signature("input -> output", "python\n# Reply with the label only.")
        )
        program.detailed_results = SimpleNamespace(
            best_idx=1,
            total_metric_calls=7,
            candidates=[
                {"self": student.signature.instructions},
                {"self": "python\n# Reply with the label only."},
            ],
            parents=[[None], [0]],
            val_aggregate_scores=[0.5, 1.0],
        )
        return program


class TestGepaOptimizer:
    def test_run_reports_timeline_and_scores(self):
        updates = []

        def progress(stage, message="", *, current=None, total=None, best_score=None):
            updates.append(stage)

        with (
            dspy.context(lm=DummyLM(ANSWERS)),
            patch("app.services.gepa_service.GEPA", FakeGEPA),
        ):
            optimizer = GepaOptimizer(
                SAMPLES, metric="contains", budget=40, progress=progress
            )
            outcome = optimizer.run("Classify the ticket.")

        assert FakeGEPA.last_kwargs["max_metric_calls"] == 40
        assert FakeGEPA.last_kwargs["reflection_minibatch_size"] <= 3

        gepa = outcome["gepa"]
        assert gepa["metric_calls"] == 7
        assert gepa["best_index"] == 1
        assert [c["index"] for c in gepa["timeline"]] == [0, 1]
        assert gepa["timeline"][1]["parent"] == 0
        assert gepa["timeline"][1]["generation"] == 1
        assert gepa["timeline"][1]["instructions"] == "Reply with the label only."
        assert gepa["timeline"][1]["score"] == 100.0
        assert gepa["timeline"][1]["iteration"] == 1
        assert any("Expected" in fb for fb in gepa["timeline"][1]["feedback"])
        assert outcome["instructions"] in (
            "Reply with the label only.",
            "Classify the ticket.",
        )

        evaluation = outcome["eval"]
        assert evaluation["metric"] == "contains"
        assert {c["name"] for c in evaluation["candidates"]} == {"original", "gepa"}
        assert len(evaluation["results"]) == evaluation["dev_size"] == 1
        assert {
            "input",
            "expected",
            "actual",
            "passed",
            "score",
            "feedback",
        } <= evaluation["results"][0].keys()
        assert outcome["optimized_prompt"].endswith("Input: {input}\nOutput:")
        assert "evaluate" in updates and "evolve" in updates

    def test_minimum_budget_and_samples(self):
        assert GepaOptimizer(SAMPLES, budget=1).budget == 10
        with pytest.raises(EvalError):
            GepaOptimizer(SAMPLES[:1])
