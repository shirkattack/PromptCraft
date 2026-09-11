"""Tests for the code sandbox, the ``tests`` metric and its GEPA feedback.

No network, no dataset download; every case finishes in well under a second
apart from the deliberate 1s timeout.
"""

import importlib.util
from pathlib import Path

import dspy
import pytest

from app.core.config import settings
from app.services.code_eval_service import (
    CodeEvalDisabled,
    evaluate_response,
    extract_code,
    run_tests,
    to_evalplus_sample,
)
from app.services.eval_service import (
    EvalError,
    Sample,
    build_metric,
    choose_metric,
    is_label_dataset,
    tests_metric,
)
from app.services.gepa_service import build_feedback_metric

ADD_TESTS = ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert add(-1, 1) == 0"]
GOOD = "def add(a, b):\n    return a + b\n"
BAD = "def add(a, b):\n    return a - b\n"


def _run(
    code: str, tests: list[str] = ADD_TESTS, imports: list[str] | None = None, **kw
):
    return run_tests(
        code,
        tests,
        imports or [],
        "add",
        timeout_s=kw.get("timeout_s", 5.0),
        memory_mb=512,
    )


def _extra(**overrides):
    return {
        "task_id": "Mbpp/1",
        "entry_point": "add",
        "test_imports": [],
        "tests": ADD_TESTS,
        **overrides,
    }


class TestExtractCode:
    def test_fenced_block(self):
        assert extract_code(
            "Sure!\n```python\n" + GOOD + "```\nDone.", "add"
        ) == GOOD.strip("\n")

    def test_fence_without_language(self):
        assert extract_code("```\n" + GOOD + "```", "add") == GOOD.strip("\n")

    def test_unfenced_function(self):
        assert extract_code(GOOD, "add") == GOOD.strip("\n")

    def test_prose_then_code(self):
        text = "Here is the function you asked for:\n\n" + GOOD + "\nHope this helps."
        assert extract_code(text, "add").startswith("def add")

    def test_wrong_function_name_is_still_extracted(self):
        text = "Here is my answer: def plus(a, b): return a + b"
        assert extract_code(text, "add") == "def plus(a, b): return a + b"

    def test_empty_or_prose_only(self):
        assert extract_code("", "add") is None
        assert extract_code("I cannot help with that.", "add") is None

    def test_stray_fence_and_dspy_markers_after_the_code_are_cut(self):
        code = GOOD.strip("\n")
        for trailer in ("```", "[[ ## completed ## ]]", "[/## completed ## ]]", "[/]"):
            assert extract_code(f"{code}\n{trailer}", "add") == code
            assert extract_code(f"{code}\n{trailer}\nSome prose.", "add") == code
            assert extract_code(f"```python\n{code}\n{trailer}\n```", "add") == code

    def test_imports_and_helpers_above_an_unfenced_function_are_kept(self):
        code = "import math\n\ndef area(r):\n    return math.pi * r * r"
        assert extract_code(code, "area") == code
        assert extract_code(f"Sure, here it is:\n{code}\n", "area") == code
        helper = "def double(x):\n    return 2 * x\n\ndef add(a, b):\n    return double(a) + b"
        assert extract_code(helper, "add") == helper
        result = _run(helper.replace("double(a) + b", "a + b"))
        assert result.status == "pass"

    def test_prose_between_import_and_function_stops_the_walk(self):
        text = "import math\nThe function below uses it.\ndef area(r):\n    return r"
        assert extract_code(text, "area") == "def area(r):\n    return r"

    def test_trailing_junk_that_breaks_parsing_is_dropped(self):
        code = GOOD.strip("\n")
        for junk in ("}", '"""', "]", "[/output]", "This function adds two numbers."):
            assert extract_code(f"{code}\n{junk}", "add") == code

    def test_code_broken_inside_the_function_stays_broken(self):
        truncated = 'def add(a, b):\n    """Add two numbers.\n    Returns the sum'
        assert extract_code(truncated, "add") == truncated
        assert _run(truncated).status == "syntax_error"
        # A complete helper does not stand in for a truncated entry point.
        text = "def helper(x):\n    return x\n\ndef add(a, b):\n    return (a +"
        assert extract_code(text, "add") == text

    def test_list_literal_lines_are_not_mistaken_for_markers(self):
        code = "def add(a, b):\n    total = [\n        a,\n        b,\n    ]\n    return sum(total)"
        assert extract_code(code, "add") == code


class TestRunTests:
    def test_all_pass(self):
        result = _run(GOOD)
        assert result.status == "pass"
        assert (result.passed, result.total) == (3, 3)
        assert result.feedback == "Correct: all 3 asserts passed."

    def test_one_assert_fails_and_is_quoted(self):
        result = _run(BAD)
        assert result.status == "assert_failed"
        assert (result.passed, result.total) == (1, 3)  # only add(0, 0) survives
        assert "Passed 1/3 asserts" in result.feedback
        assert "`assert add(1, 2) == 3`" in result.feedback

    def test_syntax_error(self):
        result = _run("def add(a, b:\n    return a + b")
        assert result.status == "syntax_error"
        assert result.feedback.startswith("SyntaxError at line 1")
        assert result.passed == 0

    def test_wrong_name(self):
        result = _run("def plus(a, b):\n    return a + b")
        assert result.status == "wrong_name"
        assert "call `add`" in result.feedback and "`plus`" in result.feedback

    def test_no_function_at_all(self):
        result = _run("x = 1")
        assert result.status == "wrong_name"
        assert "`x`" in result.feedback

    def test_exception_inside_function(self):
        result = _run("def add(a, b):\n    return a / 0")
        assert result.status == "exception"
        assert result.feedback.startswith("ZeroDivisionError")
        assert "`assert add(1, 2) == 3`" in result.feedback
        assert "ZeroDivisionError" in result.stderr_tail

    def test_timeout(self):
        result = _run("def add(a, b):\n    while True:\n        pass", timeout_s=1.0)
        assert result.status == "timeout"
        assert "Timed out after 1s" in result.feedback
        assert "`assert add(1, 2) == 3`" in result.feedback

    def test_test_imports_are_honoured_by_the_asserts_only(self):
        # The asserts see the imports; the solution does not, so a solution
        # that forgot its own import fails as it would under EvalPlus.
        tests = ["assert math.isclose(add(1, 2), 3)"]
        assert _run(GOOD, tests, imports=["import math"]).status == "pass"
        assert _run(GOOD, tests).status == "exception"  # NameError in the assert
        forgot = "def add(a, b):\n    return math.fsum([a, b])"
        assert _run(forgot, imports=["import math"]).status == "exception"

    def test_reliability_guard_blocks_os_system(self):
        code = "import os\ndef add(a, b):\n    os.system('echo x')\n    return a + b"
        result = _run(code)
        assert result.status == "exception"
        assert "TypeError" in result.feedback

    def test_empty_code_is_no_code(self):
        result = _run("")
        assert result.status == "no_code"
        assert "fenced Python code block" in result.feedback


class TestEvaluateResponse:
    def test_extracts_then_runs(self):
        assert (
            evaluate_response("```python\n" + GOOD + "```", _extra()).status == "pass"
        )
        assert evaluate_response("no code here", _extra()).status == "no_code"

    def test_disabled_setting(self, monkeypatch):
        monkeypatch.setattr(settings, "code_eval_enabled", False)
        with pytest.raises(CodeEvalDisabled):
            evaluate_response(GOOD, _extra())

    def test_evalplus_sample_shape(self):
        assert to_evalplus_sample("Mbpp/2", GOOD) == {
            "task_id": "Mbpp/2",
            "solution": GOOD,
        }


CODE_SAMPLES = [
    Sample("Write add(a, b).", GOOD, _extra(task_id=f"Mbpp/{i}")) for i in range(4)
]
LABEL_SAMPLES = [
    Sample("alpha", "high"),
    Sample("beta", "low"),
    Sample("gamma", "high"),
]
PROSE_SAMPLES = [Sample("q", "A long free-text answer " * 5)] * 3


class TestMetricWiring:
    def test_sample_example_carries_code_fields(self):
        example = CODE_SAMPLES[0].to_example()
        assert example.entry_point == "add" and example.tests == ADD_TESTS
        assert example.task_id == "Mbpp/0"
        assert list(example.inputs().keys()) == ["input"]

    def test_tests_metric_is_pass_at_1(self):
        metric = build_metric("tests")
        assert metric is tests_metric
        example = CODE_SAMPLES[0].to_example()
        assert (
            metric(example, dspy.Prediction(output="```python\n" + GOOD + "```"))
            is True
        )
        # Two of three asserts passing is still a miss.
        assert metric(example, dspy.Prediction(output=BAD)) is False

    def test_build_metric_respects_the_off_switch(self, monkeypatch):
        monkeypatch.setattr(settings, "code_eval_enabled", False)
        with pytest.raises(EvalError, match="CODE_EVAL_ENABLED"):
            build_metric("tests")

    def test_auto_picks_tests_for_code_and_keeps_the_old_choices(self):
        assert choose_metric("auto", CODE_SAMPLES) == "tests"
        assert choose_metric("auto", LABEL_SAMPLES) == "contains"
        assert choose_metric("auto", PROSE_SAMPLES) == "llm_judge"
        # One sample without asserts and the dataset is not a code dataset;
        # the length rule takes over (these solutions are short).
        mixed = CODE_SAMPLES + [Sample("q", "a")]
        assert choose_metric("auto", mixed) == "contains"

    def test_code_samples_are_not_a_label_dataset(self):
        assert is_label_dataset(CODE_SAMPLES) is False
        assert is_label_dataset(LABEL_SAMPLES) is True


class TestCodeFeedbackMetric:
    def _gold(self):
        return CODE_SAMPLES[0].to_example()

    def test_zero_for_wrong_name(self):
        metric = build_feedback_metric("tests")
        verdict = metric(
            self._gold(), dspy.Prediction(output="def plus(a, b):\n    return a + b")
        )
        assert verdict.score == 0.0
        assert "Define `add` exactly" in verdict.feedback

    def test_fraction_for_partial_pass(self):
        metric = build_feedback_metric("tests")
        verdict = metric(self._gold(), dspy.Prediction(output=BAD))
        assert verdict.score == pytest.approx(1 / 3)
        assert verdict.feedback.startswith(
            "Passed 1/3 asserts. First failure: `assert add(1, 2) == 3`"
        )

    def test_full_pass(self):
        metric = build_feedback_metric("tests")
        verdict = metric(
            self._gold(), dspy.Prediction(output="```python\n" + GOOD + "```")
        )
        assert verdict.score == 1.0
        assert verdict.feedback == "Correct: all 3 asserts passed."


def _load_build_script():
    path = Path(__file__).resolve().parents[3] / "scripts" / "build_mbppplus_dataset.py"
    spec = importlib.util.spec_from_file_location("build_mbppplus_dataset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestSplitScript:
    def test_split_is_stable_and_disjoint(self):
        script = _load_build_script()
        ids = [f"Mbpp/{i}" for i in range(20)]
        first = script.split_ids(ids, seed=1234, sizes=(10, 4, 6))
        second = script.split_ids(list(reversed(ids)), seed=1234, sizes=(10, 4, 6))
        assert first == second
        assert [len(first[k]) for k in ("train", "val", "test")] == [10, 4, 6]
        assert not (set(first["train"]) & set(first["val"]) & set(first["test"]))
        assert set(first["train"]) | set(first["val"]) | set(first["test"]) == set(ids)
        assert len(set(first["train"]) & set(first["test"])) == 0
        assert script.split_ids(ids, seed=1, sizes=(10, 4, 6)) != first

    def test_split_block_and_entry_point(self):
        script = _load_build_script()
        setup, asserts = script.split_test_block(
            "import math\nassert f(1) == 1\nassert math.isclose(f(2),\n    2.0)\n"
        )
        assert setup == ["import math"]
        assert asserts == ["assert f(1) == 1", "assert math.isclose(f(2),\n    2.0)"]
        assert (
            script.entry_point_from_assert(
                "assert set(f(1)) == {1}", "def f(x):\n  return [x]"
            )
            == "f"
        )
