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
    GuardedInstructionProposer,
    InstructionLeakGuard,
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

    def test_user_feedback_reaches_the_reflection_feedback(self):
        with (
            dspy.context(lm=DummyLM(ANSWERS)),
            patch("app.services.gepa_service.GEPA", FakeGEPA),
        ):
            optimizer = GepaOptimizer(
                SAMPLES,
                metric="contains",
                budget=40,
                user_feedback=["Answer with the label only"],
            )
            outcome = optimizer.run("Classify the ticket.")

        gepa = outcome["gepa"]
        assert gepa["user_feedback"] == ["Answer with the label only"]
        # The fake optimizer scored two misses; each carried the user's note.
        misses = [fb for c in gepa["timeline"] for fb in c["feedback"]]
        assert misses and all("Answer with the label only" in fb for fb in misses)


def test_guard_is_on_by_default_and_can_be_turned_off():
    for guard, expected in ((True, GuardedInstructionProposer), (False, type(None))):
        with (
            dspy.context(lm=DummyLM(ANSWERS)),
            patch("app.services.gepa_service.GEPA", FakeGEPA),
        ):
            outcome = GepaOptimizer(
                SAMPLES, metric="contains", budget=40, guard=guard
            ).run("Classify the ticket.")
        assert isinstance(FakeGEPA.last_kwargs["instruction_proposer"], expected)
        assert outcome["gepa"]["guard"]["mode"] == ("placeholders" if guard else "off")


# -- leak guard ------------------------------------------------------------------

ORIGINAL = "Write a Python function for the task below."
CODE_SAMPLES = [
    Sample(
        "Write a function to find the nth newman–shanks–williams prime number.\n"
        "assert newman_prime(3) == 7",
        "",
        {"tests": ["assert newman_prime(3) == 7"], "entry_point": "newman_prime"},
    ),
    Sample(
        "Write a function to remove uppercase substrings from a given string.\n"
        "assert remove_uppercase('cAstyoUrFavoRitETVshoWs') == 'cstyoravoitshos'",
        "",
        {
            "tests": ["assert remove_uppercase('a') == 'a'"],
            "entry_point": "remove_uppercase",
        },
    ),
    Sample(
        "Write a python function to split a string into characters.\n"
        "assert Split('python') == ['p','y','t','h','o','n']",
        "",
        {"tests": ["assert Split('ab') == ['a', 'b']"], "entry_point": "Split"},
    ),
    Sample(
        "Write a function to find the sum of the numbers in a list.\n"
        "assert sum_list([1, 2]) == 3",
        "",
        {"tests": ["assert sum_list([1, 2]) == 3"], "entry_point": "sum_list"},
    ),
]
SHOWN = [s.input_text for s in CODE_SAMPLES]


class ScriptedLM:
    """A reflection LM that returns canned answers and keeps the prompts."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt=None, messages=None, **kwargs):
        self.prompts.append(prompt)
        return [self.answers.pop(0)]


def _records(samples):
    return [
        {
            "Inputs": {"input": s.input_text},
            "Generated Outputs": {"output": "def f():\n    pass"},
            "Feedback": "The function name is wrong.",
        }
        for s in samples
    ]


class TestLeakGuard:
    def test_flags_names_literals_phrases_and_placeholders(self):
        guard = InstructionLeakGuard(CODE_SAMPLES, ORIGINAL, template=True)
        report = guard.check(
            "Define newman_prime carefully and return 'cstyoravoitshos' for the "
            "example. To find the nth newman shanks williams prime number, use the "
            "recurrence. Task: {task_input}",
            SHOWN[:2],
        )
        assert report.names == ["newman_prime"]
        assert report.literals == ["cstyoravoitshos"]
        assert report.placeholders == ["{task_input}"]
        assert report.phrases
        assert {"newman", "shanks", "williams"} <= set(report.terms)

    def test_generic_guidance_passes(self):
        guard = InstructionLeakGuard(CODE_SAMPLES, ORIGINAL, template=True)
        report = guard.check(
            "Name the function exactly as the assert calls it. Import every module "
            "you use inside the code block. Handle empty inputs, and use sum() or "
            "max() where they fit. Split long logic into helpers. Write a Python "
            "function for the task below.",
            SHOWN,
        )
        assert report.leaked == []

    def test_title_case_names_count_only_as_calls_or_code(self):
        guard = InstructionLeakGuard(CODE_SAMPLES, ORIGINAL, template=True)
        assert guard.check("Split the work into steps.", SHOWN).names == []
        assert guard.check("If the assert calls `Split`, name it so.", SHOWN).names == [
            "Split"
        ]

    def test_label_instructions_are_checked_for_placeholders_only(self):
        guard = InstructionLeakGuard(SAMPLES, "Classify the ticket.", template=False)
        report = guard.check(
            "Label the alpha ticket as high. Input: {input}", ["alpha ticket"]
        )
        assert report.leaked == ["{input}"]


class TestGuardedProposer:
    def test_leaky_proposal_is_regenerated_and_the_clean_one_kept(self):
        lm = ScriptedLM(
            [
                "```\nImplement newman_prime with the recurrence.\n```",
                "```\nName the function as the assert does and handle edge cases.\n```",
            ]
        )
        guard = InstructionLeakGuard(CODE_SAMPLES, ORIGINAL, template=True)
        proposer = GuardedInstructionProposer(guard, lm=lm)
        proposal = proposer(
            {"self": ORIGINAL}, {"self": _records(CODE_SAMPLES[:1])}, ["self"]
        )
        assert proposal == {
            "self": "Name the function as the assert does and handle edge cases."
        }
        assert "none of these examples will come back" in lm.prompts[0]
        assert "niche and domain specific" not in lm.prompts[0]
        assert "'newman_prime'" in lm.prompts[1]
        summary = proposer.summary()
        assert (summary["mode"], summary["regenerated"], summary["rejected"]) == (
            "template",
            1,
            0,
        )

    def test_proposal_still_leaking_after_the_retry_is_dropped(self):
        lm = ScriptedLM(["```\nUse newman_prime.\n```"] * 2)
        guard = InstructionLeakGuard(CODE_SAMPLES, ORIGINAL, template=True)
        proposer = GuardedInstructionProposer(guard, lm=lm)
        proposal = proposer(
            {"self": ORIGINAL}, {"self": _records(CODE_SAMPLES[:1])}, ["self"]
        )
        assert proposal == {}
        assert proposer.events[0]["rejected"]
        assert proposer.events[0]["leaked_after_retry"] == ["newman_prime"]

    def test_label_instructions_keep_gepas_reflection_prompt(self):
        lm = ScriptedLM(["```\nReply with the label only.\n```"])
        guard = InstructionLeakGuard(SAMPLES, "Classify the ticket.", template=False)
        records = [
            {
                "Inputs": {"input": "alpha ticket"},
                "Generated Outputs": {"output": "urgent"},
                "Feedback": "Wrong.",
            }
        ]
        proposal = GuardedInstructionProposer(guard, lm=lm)(
            {"self": "Classify the ticket."}, {"self": records}, ["self"]
        )
        assert proposal == {"self": "Reply with the label only."}
        assert lm.prompts[0].startswith(
            "I provided an assistant with the following instructions"
        )
        assert "niche and domain specific" in lm.prompts[0]
