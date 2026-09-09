"""Tests for the batched sandbox, the reported-score audit, the template-aware
rewriter and the benchmark scripts. No network, no model, no dataset download."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from app.services.code_eval_service import (
    CodeEvalResult,
    run_statements,
    run_tests,
    task_timeout,
)
from app.services.eval_service import Sample, report_pass_at_1
from app.services.gepa_service import GepaOptimizer
from app.services.optimization_service import PromptOptimizationService

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ADD = "def add(a, b):\n    return a + b\n"
TESTS = ["assert add(1, 2) == 3", "assert add(2, 2) == 4", "assert add(0, 1) == 1"]


# -- batched sandbox -----------------------------------------------------------


class TestBatchedSandbox:
    def test_per_assert_results_and_timing(self):
        result = run_tests(ADD, TESTS, [], "add", timeout_s=5, memory_mb=256)
        assert result.status == "pass"
        assert [a.index for a in result.asserts] == [0, 1, 2]
        assert all(a.status == "pass" and a.elapsed >= 0 for a in result.asserts)

    def test_base_and_plus_from_one_run(self):
        code = "def add(a, b):\n    return a + b if a < 2 else 0\n"
        result = run_tests(
            code, TESTS, [], "add", timeout_s=5, memory_mb=256, base_count=1
        )
        assert result.status == "assert_failed"
        assert (result.passed, result.total) == (2, 3)
        assert result.base_pass is True  # first assert passes
        assert result.plus_pass is False
        assert "First failure: `assert add(2, 2) == 4`" in result.feedback

    def test_base_fails_when_a_base_assert_fails(self):
        code = "def add(a, b):\n    return a + b if a != 1 else 0\n"
        result = run_tests(
            code, TESTS, [], "add", timeout_s=5, memory_mb=256, base_count=1
        )
        assert result.base_pass is False and result.plus_pass is False

    def test_exception_keeps_running_later_asserts(self):
        code = "def add(a, b):\n    if a == 1:\n        raise ValueError('boom')\n    return a + b\n"
        result = run_tests(code, TESTS, [], "add", timeout_s=5, memory_mb=256)
        assert result.status == "exception"
        assert result.passed == 2
        assert result.feedback.startswith("ValueError: boom while running")
        assert "ValueError" in result.stderr_tail

    def test_timeout_stops_the_run(self):
        code = "def add(a, b):\n    if a == 2:\n        while True: pass\n    return a + b\n"
        result = run_tests(code, TESTS, [], "add", timeout_s=1, memory_mb=256)
        assert result.status == "timeout"
        assert result.passed == 1
        assert "`assert add(2, 2) == 4`" in result.feedback
        assert len(result.asserts) == 2  # the third never ran

    def test_module_level_error_is_an_exception(self):
        result = run_tests(
            "raise RuntimeError('x')\ndef add(a, b): return a + b",
            TESTS,
            [],
            "add",
            timeout_s=5,
            memory_mb=256,
        )
        assert result.status == "exception"
        assert "while loading the code" in result.feedback
        assert result.passed == 0

    def test_user_prints_do_not_break_results(self):
        code = "print('noise')\ndef add(a, b):\n    print(a)\n    return a + b\n"
        assert (
            run_tests(code, TESTS, [], "add", timeout_s=5, memory_mb=256).status
            == "pass"
        )

    def test_run_statements_collects_a_file(self):
        setup = [
            '__f = open("collect.jsonl", "w")',
            "def __cap(i, v):\n    __f.write(repr((i, v)) + chr(10)); __f.flush()",
        ]
        run = run_statements(
            ADD,
            ["__cap(0, add(1, 2))", "__cap(1, add(2, 3))"],
            setup,
            timeout_s=5,
            memory_mb=256,
            collect="collect.jsonl",
        )
        assert run.module_error is None
        assert [r.status for r in run.results] == ["pass", "pass"]
        assert run.collected.splitlines() == ["(0, 3)", "(1, 5)"]

    def test_task_timeout_caps_the_whole_run(self):
        assert task_timeout(10, 3) == 30
        assert task_timeout(10, 1) == 20
        assert task_timeout(10, 500) == 120


# -- reported score audit ------------------------------------------------------


def _fake_result(status: str, passed: int, total: int = 2) -> CodeEvalResult:
    return CodeEvalResult(status, passed, total, "fb", "", base_count=1)


class TestReportedScore:
    def test_report_pass_at_1_counts_only_full_credit(self):
        assert report_pass_at_1([1.0, 0.5, 0.0, 1.0]) == 50.0
        assert report_pass_at_1([]) == 0.0

    def test_gepa_reports_pass_at_1_while_the_loop_metric_sees_fractions(self):
        # 12 dev samples: 5 fully passing, 3 passing half their asserts, 4 failing.
        outcomes = (
            [("pass", 2)] * 5 + [("assert_failed", 1)] * 3 + [("assert_failed", 0)] * 4
        )
        samples = [
            Sample(
                f"task {i}",
                "def f(): pass",
                {
                    "task_id": f"T/{i}",
                    "entry_point": "f",
                    "tests": ["assert 1", "assert 2"],
                    "base_count": 1,
                },
            )
            for i in range(12)
        ]
        by_task = {f"T/{i}": outcomes[i] for i in range(12)}

        def fake_evaluate(response, extra):
            status, passed = by_task[extra["task_id"]]
            return _fake_result(status, passed)

        answers = {
            f"task {i}": {"output": "```python\ndef f(): pass\n```"} for i in range(12)
        }
        with (
            dspy.context(lm=DummyLM(answers)),
            patch("app.services.gepa_service.evaluate_response", fake_evaluate),
        ):
            optimizer = GepaOptimizer(
                samples, metric="tests", budget=10, train=samples[:2], dev=samples
            )
            assert optimizer.report_binary is True
            percent, rows = optimizer._evaluate(optimizer._program("Solve it."))

        assert percent == 41.67  # 5 of 12, exactly
        loop_average = sum(r["score"] for r in rows) / len(rows) * 100
        assert loop_average == pytest.approx(54.17, abs=0.01)  # (5 + 3 * 0.5) / 12
        assert loop_average > percent
        assert sum(r["passed"] for r in rows) == 5

    def test_returned_candidate_is_the_best_on_val_pass_at_1(self):
        from types import SimpleNamespace

        samples = [
            Sample(
                f"task {i}",
                "def f(): pass",
                {
                    "task_id": f"T/{i}",
                    "entry_point": "f",
                    "tests": ["assert 1", "assert 2"],
                    "base_count": 1,
                },
            )
            for i in range(4)
        ]

        class Fake:
            def __init__(self, **kwargs):
                pass

            def compile(self, student, *, trainset, valset):
                program = dspy.Predict(
                    dspy.Signature("input -> output", "aggregate winner")
                )
                # Candidate 1 wins on the fraction metric (GEPA's best_idx),
                # candidate 2 wins on pass@1 with a lower aggregate.
                program.detailed_results = SimpleNamespace(
                    best_idx=1,
                    total_metric_calls=3,
                    candidates=[
                        {"self": "seed"},
                        {"self": "aggregate winner"},
                        {"self": "pass@1 winner"},
                    ],
                    parents=[[None], [0], [0]],
                    val_aggregate_scores=[0.4, 0.8, 0.6],
                    val_subscores=[
                        {0: 0.5, 1: 0.5, 2: 0.3, 3: 0.3},
                        {0: 1.0, 1: 0.9, 2: 0.9, 3: 0.4},
                        {0: 1.0, 1: 1.0, 2: 0.2, 3: 0.2},
                    ],
                )
                return program

        answers = {
            f"task {i}": {"output": "```python\ndef f(): pass\n```"} for i in range(4)
        }
        calls = {"n": 0}

        def improving(response, extra):
            # The baseline evaluation (the first two dev calls) fails, the
            # evolved prompt's evaluation passes, so the run counts as improved.
            calls["n"] += 1
            if calls["n"] > 2:
                return _fake_result("pass", 2)
            return _fake_result("assert_failed", 1)

        with (
            dspy.context(lm=DummyLM(answers)),
            patch("app.services.gepa_service.GEPA", Fake),
            patch("app.services.gepa_service.evaluate_response", improving),
        ):
            outcome = GepaOptimizer(
                samples, metric="tests", budget=10, train=samples[:2], dev=samples[2:]
            ).run("seed")
        assert outcome["gepa"]["best_index"] == 1
        assert outcome["gepa"]["selected_index"] == 2
        assert outcome["gepa"]["selection"] == "val pass@1"
        assert outcome["instructions"] == "pass@1 winner"
        assert [c["score"] for c in outcome["gepa"]["timeline"]] == [0.0, 25.0, 50.0]

    def test_explicit_train_and_dev_bypass_the_caps(self):
        samples = [Sample(f"in {i}", "high") for i in range(6)]
        optimizer = GepaOptimizer(
            samples, metric="contains", train=samples[:4], dev=samples[4:]
        )
        assert len(optimizer.train) == 4 and len(optimizer.dev) == 2


# -- rewriter on template prompts ---------------------------------------------


INPUTS = [
    '"""\nWrite a function to find the shared elements from the given two lists.\nassert similar_elements((3, 4), (4, 5)) == (4,)\n"""',
    '"""\nWrite a python function to identify non-prime numbers.\nassert is_not_prime(2) == False\n"""',
    '"""\nWrite a function to find the n largest integers from a list.\nassert heap_queue_largest([1, 2, 3], 2) == [3, 2]\n"""',
]


class TestTemplateRewriter:
    def test_specific_tokens_flags_words_from_one_input_only(self):
        service = PromptOptimizationService()
        leaked = service.specific_tokens(
            "Write a function similar_elements that returns shared elements.",
            INPUTS,
            "Write a Python function.",
        )
        assert "similar_elements" in leaked
        assert "function" not in leaked  # in the original prompt and every input

    def test_meta_prompt_frames_the_template_and_shows_inputs(self):
        service = PromptOptimizationService()
        text = service._generate_meta_prompt(
            "Write a Python function.", "code", "", INPUTS
        )
        assert "This prompt is a template" in text
        assert text.count("### Example input") == 3

    def test_rewrite_mentioning_one_task_is_regenerated_then_dropped(self):
        service = PromptOptimizationService()
        bad = {
            "optimized_prompt": "Write similar_elements to return the shared elements of two tuples."
        }
        with dspy.context(lm=DummyLM([bad, bad])):
            result = service._optimize_with_meta_prompt(
                "Write a Python function.", "code", None, "", INPUTS
            )
        assert result["optimized_prompt"] == "Write a Python function."
        rejected = result["metadata"]["rewrite_rejected"]
        assert "similar_elements" in rejected["tokens"]
        assert result["metadata"]["regenerated_for"]

    def test_generic_rewrite_is_kept(self):
        service = PromptOptimizationService()
        bad = {"optimized_prompt": "Implement is_not_prime as asked."}
        good = {
            "optimized_prompt": "Write the requested Python function. Respond with one fenced code block."
        }
        with dspy.context(lm=DummyLM([bad, good])):
            result = service._optimize_with_meta_prompt(
                "Write a Python function.", "code", None, "", INPUTS
            )
        assert result["optimized_prompt"].startswith("Write the requested")
        for entry_point in ("similar_elements", "is_not_prime", "heap_queue_largest"):
            assert entry_point not in result["optimized_prompt"]
        assert "rewrite_rejected" not in result["metadata"]


# -- build script helpers ------------------------------------------------------


class TestBuildScript:
    def test_format_assert_and_tolerance(self):
        build = _load("build_mbppplus_dataset")
        assert build.format_assert("f", "1, 'a'", "[1]", 0) == "assert f(1, 'a') == [1]"
        assert build.format_assert("f", "2", "0.5", 1e-6) == (
            "assert __close(f(2), 0.5, 1e-06)"
        )
        # A float expected value gets EvalPlus's default tolerance.
        assert build.format_assert("f", "1", "2.5", 0, is_floats=True) == (
            "assert __close(f(1), 2.5, 1e-06)"
        )
        assert build.format_assert("f", "1", "[1.0]", 0.001) == (
            "assert __close(f(1), [1.0], 0.001)"
        )
        assert build.format_assert("are_equivalent", "'a', 'b'", "True", 0) == (
            "are_equivalent('a', 'b')"
        )
        assert (
            build.format_assert("sum_div", "6", "6", 0) == "assert sum_div(6) in (6, 0)"
        )

    def test_special_oracles_follow_evalplus(self):
        build = _load("build_mbppplus_dataset")
        assert build.format_assert("similar_elements", "(1, 2), (2, 3)", "(2,)", 0) == (
            "assert set(similar_elements((1, 2), (2, 3))) == set((2,))"
        )
        assert build.format_assert("check_str", "'abc'", "False", 0) == (
            "assert __not_none_ok(check_str('abc'), False)"
        )
        task = {
            "task_id": "T/737",
            "code": "import re\ndef check_str(s):\n    return re.match(r'[aeiou]', s)",
            "entry_point": "check_str",
            "setup": [],
            "base_input": [["apple"], ["xyz"]],
            "plus_input": [],
            "atol": 0,
        }
        expected, skipped = build.capture_expected(task)
        assert (
            expected == {0: "True", 1: "False"} and skipped["output_not_literal"] == 0
        )
        tests, _ = build.asserts_for(task, expected)
        setup = build.setup_for(task)
        assert (
            run_tests(
                task["code"], tests, setup, "check_str", timeout_s=5, memory_mb=256
            ).status
            == "pass"
        )
        # A bare boolean answer is accepted too, as in EvalPlus.
        boolean = "def check_str(s):\n    return s[0] in 'aeiou'"
        assert (
            run_tests(
                boolean, tests, setup, "check_str", timeout_s=5, memory_mb=256
            ).status
            == "pass"
        )
        alt = build.format_assert("surface_Area", "3, 4", "45", 0)
        assert (
            alt
            == "assert __alt_ok(surface_Area(3, 4), 45, __alt_surface_Area(3, 4), 0)"
        )
        setup = build.setup_for({"setup": [], "atol": 0, "entry_point": "surface_Area"})
        assert any("def __alt_surface_Area" in line for line in setup)
        assert any("def __alt_ok" in line for line in setup)

    def test_set_order_is_stable_across_sandbox_runs(self):
        code = "def f(xs):\n    return tuple(set(xs))"
        args = "('DRwvS', 'ITntCqEvPi', 'SmJpP', 'tUqF')"
        task = {
            "task_id": "T/2",
            "code": code,
            "entry_point": "f",
            "setup": [],
            "base_input": [[("DRwvS", "ITntCqEvPi", "SmJpP", "tUqF")]],
            "plus_input": [],
            "atol": 0,
        }
        build = _load("build_mbppplus_dataset")
        expected, _ = build.capture_expected(task)
        tests, _ = build.asserts_for(task, expected)
        for _ in range(3):
            assert (
                run_tests(code, tests, [], "f", timeout_s=5, memory_mb=256).status
                == "pass"
            ), args

    def test_args_literal_round_trips_or_none(self):
        build = _load("build_mbppplus_dataset")
        assert (
            build.args_literal([(1, 2), "x", {"k": [1]}]) == "(1, 2), 'x', {'k': [1]}"
        )
        assert build.args_literal([float("nan")]) is None
        assert build.args_literal([float("inf"), (1,)]) == "float('inf'), (1,)"

    def test_capture_writes_inf_and_flags_floats(self):
        build = _load("build_mbppplus_dataset")
        task = {
            "task_id": "T/2",
            "code": "def f(x):\n    return float('inf') if x == 0 else 1.0 / x",
            "entry_point": "f",
            "setup": [],
            "base_input": [[0], [4]],
            "plus_input": [],
            "atol": 0,
        }
        expected, skipped = build.capture_expected(task)
        assert expected == {0: "float('inf')", 1: "0.25"}
        assert skipped["output_not_literal"] == 0
        tests, _ = build.asserts_for(task, expected)
        assert tests == [
            "assert __close(f(0), float('inf'), 1e-06)",
            "assert __close(f(4), 0.25, 1e-06)",
        ]
        setup = build.setup_for(task)
        assert any("def __close" in line for line in setup)
        assert (
            run_tests(
                task["code"], tests, setup, "f", timeout_s=5, memory_mb=256
            ).status
            == "pass"
        )
        # A last-bit float difference passes, as it does in EvalPlus.
        close = "def f(x):\n    return float('inf') if x == 0 else 0.25000000000000006"
        assert (
            run_tests(close, tests, setup, "f", timeout_s=5, memory_mb=256).status
            == "pass"
        )

    def test_missing_import_feedback_names_the_fix(self):
        code = "def area(r):\n    return math.pi * r * r"
        result = run_tests(
            code, ["assert area(0) == 0"], [], "area", timeout_s=5, memory_mb=256
        )
        assert result.status == "exception"
        assert result.feedback.startswith(
            "NameError: the code uses `math` without importing it; add `import math`"
        )
        other = run_tests(
            "def f():\n    return helper()",
            ["assert f()"],
            [],
            "f",
            timeout_s=5,
            memory_mb=256,
        )
        assert "`helper` is used but never defined or imported" in other.feedback

    def test_setup_imports_do_not_leak_into_the_solution(self):
        code = "def area(r):\n    return math.pi * r * r"  # forgot: import math
        tests = ["assert __close(area(1), 3.141592653589793, 1e-06)"]
        setup = ["import math", "def __close(a, b, t):\n    return abs(a - b) <= t"]
        result = run_tests(code, tests, setup, "area", timeout_s=5, memory_mb=256)
        assert result.status == "exception"
        assert result.feedback.startswith("NameError")

    def test_clamp_timeout(self):
        build = _load("build_mbppplus_dataset")
        assert build.clamp_timeout(0.01) == 2.0
        assert build.clamp_timeout(1.0) == 4.0
        assert build.clamp_timeout(30.0) == 20.0

    def test_capture_and_asserts_base_first(self):
        build = _load("build_mbppplus_dataset")
        task = {
            "task_id": "T/1",
            "code": "def add(a, b):\n    return a + b",
            "entry_point": "add",
            "setup": [],
            "base_input": [[1, 2], [2, 2]],
            "plus_input": [[0, 0], [float("nan"), 1]],
            "atol": 0,
        }
        expected, skipped = build.capture_expected(task)
        assert expected == {0: "3", 1: "4", 2: "0"}
        assert skipped == {"args_not_literal": 1, "output_not_literal": 0, "raised": 0}
        tests, base_count = build.asserts_for(task, expected)
        assert tests == [
            "assert add(1, 2) == 3",
            "assert add(2, 2) == 4",
            "assert add(0, 0) == 0",
        ]
        assert base_count == 2

    def test_split_is_stable_and_disjoint(self):
        build = _load("build_mbppplus_dataset")
        ids = [f"Mbpp/{i}" for i in range(20)]
        first = build.split_ids(ids, seed=1234, sizes=(10, 4, 6))
        assert first == build.split_ids(
            list(reversed(ids)), seed=1234, sizes=(10, 4, 6)
        )
        assert not set(first["train"]) & set(first["test"]) and not set(
            first["val"]
        ) & set(first["test"])
        assert set(first["train"]) | set(first["val"]) | set(first["test"]) == set(ids)

    def test_committed_split_matches_the_seed(self):
        split_file = SCRIPTS.parent / "docs" / "benchmarks" / "mbppplus" / "split.json"
        if not split_file.exists():
            pytest.skip("dataset not built")
        build = _load("build_mbppplus_dataset")
        stored = json.loads(split_file.read_text())
        ids = stored["train"] + stored["val"] + stored["test"]
        assert (
            build.split_text(build.split_ids(ids, stored["seed"]), stored["seed"])
            == split_file.read_text()
        )


# -- runner and summarizer -----------------------------------------------------


class TestRunner:
    def test_test_ids_must_be_disjoint(self):
        runner = _load("run_benchmark")
        mk = lambda tid: Sample("q", "a", {"task_id": tid})  # noqa: E731
        runner.assert_disjoint([mk("a"), mk("b")], [mk("c")], [mk("d")])
        with pytest.raises(AssertionError, match="test ids also in train/val"):
            runner.assert_disjoint([mk("a")], [mk("c")], [mk("a")])
        with pytest.raises(AssertionError, match="duplicate"):
            runner.assert_disjoint([mk("a")], [mk("a")], [mk("d")])

    def test_think_block_fails_loudly(self):
        runner = _load("run_benchmark")

        class Program:
            def __call__(self, input):
                return dspy.Prediction(
                    output="<think>hmm</think>\n```python\ndef f(): pass\n```"
                )

        sample = Sample(
            "q",
            "a",
            {"task_id": "T/1", "entry_point": "f", "tests": ["assert f() is None"]},
        )
        with pytest.raises(runner.ThinkBlockError):
            runner.evaluate(Program(), [sample], "x")

    def test_evaluate_scores_base_and_plus(self):
        runner = _load("run_benchmark")

        class Program:
            def __call__(self, input):
                return dspy.Prediction(
                    output="```python\ndef f(x):\n    return x if x < 2 else 0\n```"
                )

        extra = {
            "task_id": "T/1",
            "entry_point": "f",
            "tests": ["assert f(1) == 1", "assert f(3) == 3"],
            "base_count": 1,
        }
        out = runner.evaluate(Program(), [Sample("q", "a", extra)], "x")
        assert (out["base_pass"], out["plus_pass"], out["n"]) == (1, 0, 1)
        assert out["rows"][0]["status"] == "assert_failed"

    def test_result_path_and_prompts(self):
        runner = _load("run_benchmark")
        path = runner.result_path(Path("out"), "qwen3.6:27b", "bare", "gepa", 2)
        assert path == Path("out/qwen3.6-27b/bare/gepa/seed2.json")
        assert runner.PROMPTS["fixed"] == runner.PROMPTS["bare"] + " " + runner.ONE_LINE


class TestSummarizer:
    def test_paired_bootstrap_ci_brackets_the_mean(self):
        summarize = _load("summarize_results")
        diffs = [1.0] * 30 + [0.0] * 160 + [-1.0] * 8
        lo, hi = summarize.paired_bootstrap_ci(diffs, resamples=500)
        mean = sum(diffs) / len(diffs) * 100
        assert lo < mean < hi
        assert lo > 0  # 22 net wins on 198 tasks is clear of zero

    def test_tables_from_two_runs(self, tmp_path):
        summarize = _load("summarize_results")
        rows_a = [
            {"task_id": f"T/{i}", "plus_pass": i < 5, "base_pass": i < 6}
            for i in range(10)
        ]
        rows_b = [
            {"task_id": f"T/{i}", "plus_pass": i < 7, "base_pass": i < 8}
            for i in range(10)
        ]

        def run(method, rows):
            plus = sum(r["plus_pass"] for r in rows)
            base = sum(r["base_pass"] for r in rows)
            return {
                "config": {
                    "model": {"tag": "m:latest", "digest": "abc123456789"},
                    "reflection_model": {"tag": "r"},
                    "git_commit": "deadbeef0",
                    "prompt_name": "bare",
                    "prompt": "P",
                    "method": method,
                    "seed": 1,
                    "budget": 5,
                    "demos": 4,
                },
                "test": {
                    "n": 10,
                    "plus_pass": plus,
                    "base_pass": base,
                    "plus_pass_at_1": plus * 10,
                    "base_pass_at_1": base * 10,
                },
                "val": {"plus_pass_at_1": 50.0},
                "test_rows": rows,
                "wall_clock": {"total_seconds": 60},
            }

        text, facts = summarize.summarize(
            [run("original", rows_a), run("random_demos", rows_b)]
        )
        assert "| original | 1 | 50.0% (5/10) | 60.0% (6/10) | — |" in text
        assert "| random_demos | 1 | 70.0% (7/10) |" in text
        assert facts["m:latest/bare/random_demos"]["delta"] == 20.0
