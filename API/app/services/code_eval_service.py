"""Run model-written Python against a task's asserts in a sandboxed subprocess.

This is the ``tests`` metric behind coding benchmarks such as MBPP+: the model
answers a task with code, and the code is scored by executing the task's
``assert`` statements in a fresh interpreter with a per-statement timer, a
task-level timeout, a memory limit and a guard that disables the destructive
parts of ``os``/``shutil``/``subprocess``. Standard library only, so the API
gains no dependency.

All of a task's asserts run in one interpreter (MBPP+ has about a hundred per
task) through a small harness that records the status and wall-clock of each
statement to a file the parent reads back. It is a subprocess sandbox, not a
container. Treat it as "good enough for a machine you don't mind" and see the
README's warning before turning it on.
"""

import ast
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Literal

from app.core.config import settings

logger = logging.getLogger(__name__)

CodeStatus = Literal[
    "pass",
    "assert_failed",
    "no_code",
    "syntax_error",
    "wrong_name",
    "exception",
    "timeout",
]
AssertStatus = Literal["pass", "assert_failed", "exception", "timeout"]

STDERR_TAIL_LINES = 20
FEEDBACK_ASSERT_CHARS = 120
# A task never runs longer than this in total, however many asserts it has.
MAX_TASK_SECONDS = 120.0

# Everything the executed file gets before the model's code. Modelled on
# EvalPlus's reliability guard: the process is disposable, but the machine it
# runs on is not, so anything that deletes, spawns, signals or talks to the
# network is replaced with None (calling it raises TypeError, which the runner
# reports as ``exception``). Keep the whole list here.
RELIABILITY_GUARD = """\
import builtins as _b, os as _os, shutil as _sh, subprocess as _sp, sys as _sys
_os.environ["OMP_NUM_THREADS"] = "1"
for _name in ("system", "kill", "killpg", "remove", "unlink", "rmdir", "removedirs",
              "fork", "forkpty", "putenv", "rename", "renames", "truncate", "replace",
              "chdir", "chmod", "chown", "chroot", "setuid", "setgid", "fchdir"):
    setattr(_os, _name, None)
for _name in ("rmtree", "move", "chown"):
    setattr(_sh, _name, None)
_sp.Popen = None
_b.exit = None
_b.quit = None
_b.help = None
for _name in ("ipdb", "joblib", "psutil", "tkinter", "resource", "socket",
              "urllib", "http", "requests", "multiprocessing"):
    _sys.modules[_name] = None
del _b, _os, _sh, _sp, _sys, _name
"""

# The harness that runs inside the sandbox. It reads spec.json (setup, code,
# statements, timeout), executes the code once and then every statement in
# turn under a SIGALRM timer, and appends one JSON line per statement to
# results.jsonl. User code's stdout is discarded; its stderr passes through.
HARNESS = (
    """\
import json as _json, os as _os0, signal as _signal, sys as _sys0, time as _time, traceback as _tb
"""
    + RELIABILITY_GUARD
    + """
_spec = _json.load(open("spec.json", encoding="utf-8"))
_results = open("results.jsonl", "w", encoding="utf-8")
_ns = {"__name__": "__main__"}
_sys0.stdout = open(_os0.devnull, "w")


def _emit(**row):
    _results.write(_json.dumps(row) + "\\n")
    _results.flush()


class _Timeout(BaseException):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _tail():
    return "".join(_tb.format_exc().splitlines(True)[-%(tail)d:])


_signal.signal(_signal.SIGALRM, _alarm)
try:
    _signal.setitimer(_signal.ITIMER_REAL, _spec["timeout"])
    exec(compile(_spec["setup"] + "\\n" + _spec["code"], "solution.py", "exec"), _ns)
except _Timeout:
    _emit(index=-1, status="timeout", exc="", msg="", tb="", elapsed=_spec["timeout"])
    _sys0.exit(0)
except BaseException as _e:
    _emit(index=-1, status="exception", exc=type(_e).__name__, msg=str(_e), tb=_tail(), elapsed=0.0)
    _sys0.exit(0)
finally:
    _signal.setitimer(_signal.ITIMER_REAL, 0)

for _i, _stmt in enumerate(_spec["statements"]):
    _t0 = _time.perf_counter()
    _status, _exc, _msg, _trace = "pass", "", "", ""
    _signal.setitimer(_signal.ITIMER_REAL, _spec["timeout"])
    try:
        exec(compile(_stmt, "<statement %%d>" %% _i, "exec"), _ns)
    except _Timeout:
        _emit(index=_i, status="timeout", exc="", msg="", tb="", elapsed=_time.perf_counter() - _t0)
        break
    except AssertionError as _e:
        _status, _exc, _msg, _trace = "assert_failed", "AssertionError", str(_e), _tail()
    except BaseException as _e:
        _status, _exc, _msg, _trace = "exception", type(_e).__name__, str(_e), _tail()
    finally:
        _signal.setitimer(_signal.ITIMER_REAL, 0)
    _emit(index=_i, status=_status, exc=_exc, msg=_msg, tb=_trace, elapsed=_time.perf_counter() - _t0)
_results.close()
"""
) % {"tail": STDERR_TAIL_LINES}


class CodeEvalDisabled(Exception):
    """Raised when the ``tests`` metric is used with CODE_EVAL_ENABLED=false."""


@dataclass
class AssertResult:
    index: int
    status: AssertStatus
    message: str = ""  # exception type and message, empty on pass
    elapsed: float = 0.0
    traceback: str = ""


@dataclass
class CodeEvalResult:
    status: CodeStatus
    passed: int  # asserts passed
    total: int  # asserts run
    feedback: str  # one to three lines a prompt author or reflector can act on
    stderr_tail: str  # last ~20 lines of stderr/traceback, for debugging only
    asserts: list[AssertResult] = field(default_factory=list)
    # How many leading asserts are the benchmark's "base" tests; the rest are
    # the extended ("plus") inputs. None when the dataset does not say.
    base_count: int | None = None

    @property
    def fraction(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def plus_pass(self) -> bool:
        """Every assert passed: pass@1 as MBPP+ defines it."""
        return self.status == "pass"

    @property
    def base_pass(self) -> bool:
        """The base asserts passed (all asserts when base_count is unknown)."""
        if self.base_count is None:
            return self.plus_pass
        if self.status in {"no_code", "syntax_error", "wrong_name"}:
            return False
        statuses = {a.index: a.status for a in self.asserts}
        return all(statuses.get(i) == "pass" for i in range(self.base_count))


# -- extraction ----------------------------------------------------------------

_FENCE = re.compile(r"```[ \t]*(?:python|py|python3)?[ \t]*\r?\n(.*?)```", re.S | re.I)
_ANY_DEF = re.compile(r"^[ \t]*(?:async[ \t]+)?def[ \t]+\w+[ \t]*\(", re.M)


def extract_code(response: str, entry_point: str) -> str | None:
    """Pull the Python out of a model response.

    In order: the first fenced code block; else from ``def <entry_point>`` to
    the end; else from the first ``def`` to the end (so a wrong function name
    can be diagnosed instead of reported as "no code"); else the whole response
    if it parses; else None.
    """
    text = (response or "").replace("\r\n", "\n")
    if not text.strip():
        return None

    fenced = _FENCE.search(text)
    if fenced and fenced.group(1).strip():
        return fenced.group(1).strip("\n")

    named = re.search(
        rf"(?:async[ \t]+)?def[ \t]+{re.escape(entry_point)}[ \t]*\(", text
    )
    if named:
        return text[named.start() :].strip("\n")

    any_def = _ANY_DEF.search(text) or re.search(
        r"(?:async[ \t]+)?def[ \t]+\w+[ \t]*\(", text
    )
    if any_def:
        return text[any_def.start() :].strip("\n")

    try:
        ast.parse(text)
    except SyntaxError:
        return None
    return text.strip("\n")


# -- runner --------------------------------------------------------------------


def _defined_names(tree: ast.Module) -> list[str]:
    """Top-level names the module binds: functions, classes, assignments, imports."""
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            names += [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            names += [a.asname or a.name for a in node.names]
        elif isinstance(node, ast.Import):
            names += [(a.asname or a.name).split(".")[0] for a in node.names]
    return names


def _limit_resources(memory_mb: int) -> None:
    """preexec_fn: cap CPU time and memory; best effort per platform.

    macOS rejects RLIMIT_AS and RLIMIT_DATA outright, so there the memory cap
    does not apply and only the timers protect the machine.
    """
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX
        return
    limits = [
        (getattr(resource, "RLIMIT_CPU", None), int(MAX_TASK_SECONDS) + 10),
        (getattr(resource, "RLIMIT_AS", None), memory_mb * 1024 * 1024),
        (getattr(resource, "RLIMIT_DATA", None), memory_mb * 1024 * 1024),
        (getattr(resource, "RLIMIT_NPROC", None), 64),
    ]
    for name, value in limits:
        if name is None:
            continue
        try:
            resource.setrlimit(name, (value, value))
        except (ValueError, OSError):  # unsupported here; skip that limit
            pass


def _short(text: str, limit: int = FEEDBACK_ASSERT_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _tail(text: str) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-STDERR_TAIL_LINES:])


@dataclass
class StatementRun:
    """Raw outcome of ``run_statements``."""

    results: list[AssertResult]
    module_error: AssertResult | None  # the code itself failed to load
    collected: str  # contents of the ``collect`` file, if one was requested
    stderr_tail: str


def task_timeout(timeout_s: float, statements: int) -> float:
    """Cap on the whole task: every statement may take ``timeout_s`` at most,
    but a task never runs longer than MAX_TASK_SECONDS in total."""
    return min(max(timeout_s * statements, 2 * timeout_s), MAX_TASK_SECONDS)


def run_statements(
    code: str,
    statements: list[str],
    setup: list[str],
    *,
    timeout_s: float,
    memory_mb: int,
    collect: str | None = None,
) -> StatementRun:
    """Execute ``code`` once, then each statement, in one sandboxed interpreter.

    ``timeout_s`` applies to each statement (and to loading the code); the
    whole run is additionally capped by ``task_timeout``. A statement that
    times out ends the run. ``collect`` names a file in the sandbox directory
    whose contents are returned, for callers whose setup code writes there.
    """
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"}
    tmpdir = tempfile.mkdtemp(prefix="promptcraft-code-")
    try:
        with open(os.path.join(tmpdir, "harness.py"), "w", encoding="utf-8") as f:
            f.write(HARNESS)
        with open(os.path.join(tmpdir, "spec.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "setup": "\n".join(setup),
                    "code": code,
                    "statements": statements,
                    "timeout": float(timeout_s),
                },
                f,
            )
        stderr = ""
        killed = False
        try:
            proc = subprocess.run(
                # -P/-s/-S: no script dir, user site or site.py on sys.path.
                # Not -I: it would also ignore PYTHONHASHSEED, and a fixed hash
                # seed is what keeps set iteration order the same between the
                # dataset build and evaluation.
                [sys.executable, "-P", "-s", "-S", "harness.py"],
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=task_timeout(timeout_s, len(statements)),
                preexec_fn=lambda: _limit_resources(memory_mb),
            )
            stderr = proc.stderr or ""
        except subprocess.TimeoutExpired as exc:
            killed = True
            raw = exc.stderr
            stderr = raw.decode(errors="replace") if isinstance(raw, bytes) else (raw or "")

        rows: list[dict[str, Any]] = []
        results_path = os.path.join(tmpdir, "results.jsonl")
        if os.path.exists(results_path):
            with open(results_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            break
        collected = ""
        if collect:
            path = os.path.join(tmpdir, collect)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    collected = f.read()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    def as_result(row: dict[str, Any]) -> AssertResult:
        message = f"{row.get('exc', '')}: {row.get('msg', '')}".strip(": ")
        return AssertResult(
            index=int(row["index"]),
            status=row["status"],
            message=message,
            elapsed=float(row.get("elapsed") or 0.0),
            traceback=str(row.get("tb") or ""),
        )

    module_error = next((as_result(r) for r in rows if r["index"] == -1), None)
    results = [as_result(r) for r in rows if r["index"] >= 0]
    finished = {r.index for r in results}
    stopped = any(r.status == "timeout" for r in results)
    if module_error is None and not stopped and len(results) < len(statements):
        # The interpreter died (task cap, memory, crash) part-way through a
        # statement: that statement is a timeout when we killed it, otherwise
        # an exception carrying whatever stderr says.
        index = next(i for i in range(len(statements)) if i not in finished)
        results.append(
            AssertResult(
                index=index,
                status="timeout" if killed else "exception",
                message="" if killed else f"Interpreter exited: {_short(_tail(stderr), 80)}",
                traceback=_tail(stderr),
            )
        )
    return StatementRun(results, module_error, collected, _tail(stderr))


def run_tests(
    code: str,
    tests: list[str],
    test_imports: list[str],
    entry_point: str,
    *,
    timeout_s: float,
    memory_mb: int,
    base_count: int | None = None,
) -> CodeEvalResult:
    """Execute each assert against ``code`` in one sandbox; count passes.

    Stops early on ``syntax_error``, ``no_code``, ``wrong_name`` and
    ``timeout``. An assert that raises something other than AssertionError
    counts as a failure and is reported as ``exception`` when it is the first.
    """
    total = len(tests)
    if not code or not code.strip():
        return CodeEvalResult(
            "no_code",
            0,
            total,
            "No Python code found in the response. Respond with a single fenced Python code block.",
            "",
            base_count=base_count,
        )

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return CodeEvalResult(
            "syntax_error",
            0,
            total,
            f"SyntaxError at line {exc.lineno or '?'}: {exc.msg}.",
            "",
            base_count=base_count,
        )

    defined = _defined_names(tree)
    if entry_point not in defined:
        names = ", ".join(f"`{n}`" for n in defined[:5]) or "no function"
        return CodeEvalResult(
            "wrong_name",
            0,
            total,
            f"The tests call `{entry_point}` but the code defines {names}. "
            f"Define `{entry_point}` exactly.",
            "",
            base_count=base_count,
        )

    run = run_statements(
        code, tests, test_imports, timeout_s=timeout_s, memory_mb=memory_mb
    )
    if run.module_error is not None:
        err = run.module_error
        if err.status == "timeout":
            return CodeEvalResult(
                "timeout",
                0,
                total,
                f"Timed out after {timeout_s:g}s while loading the code; "
                "likely work at module level.",
                run.stderr_tail,
                base_count=base_count,
            )
        return CodeEvalResult(
            "exception",
            0,
            total,
            f"{_short(err.message, 160)} while loading the code.",
            err.traceback or run.stderr_tail,
            base_count=base_count,
        )

    passed = sum(1 for r in run.results if r.status == "pass")
    first_failure = next((r for r in run.results if r.status != "pass"), None)
    if first_failure is None:
        return CodeEvalResult(
            "pass",
            passed,
            total,
            f"Correct: all {total} asserts passed.",
            "",
            run.results,
            base_count,
        )

    assertion = tests[first_failure.index]
    if first_failure.status == "timeout":
        return CodeEvalResult(
            "timeout",
            passed,
            total,
            f"Timed out after {timeout_s:g}s on `{_short(assertion)}`; "
            "likely an infinite loop or exponential solution.",
            first_failure.traceback or run.stderr_tail,
            run.results,
            base_count,
        )
    exc_type, _, message = first_failure.message.partition(":")
    message = message.strip()
    if first_failure.status == "assert_failed":
        detail = f" ({message})" if message else ""
        return CodeEvalResult(
            "assert_failed",
            passed,
            total,
            f"Passed {passed}/{total} asserts. First failure: `{_short(assertion)}`{detail}",
            first_failure.traceback,
            run.results,
            base_count,
        )
    return CodeEvalResult(
        "exception",
        passed,
        total,
        f"{exc_type}: {_short(message, 80)} while running `{_short(assertion)}`.",
        first_failure.traceback,
        run.results,
        base_count,
    )


# -- entry points used by the metrics ------------------------------------------


def code_fields(extra: dict[str, Any] | None) -> dict[str, Any] | None:
    """The code-eval fields of a sample's extra_data, or None if it has none."""
    if not extra or not isinstance(extra.get("tests"), list) or not extra["tests"]:
        return None
    base_count = extra.get("base_count")
    timeout_s = extra.get("timeout_s")
    return {
        "task_id": extra.get("task_id"),
        "entry_point": str(extra.get("entry_point") or ""),
        "test_imports": [str(line) for line in extra.get("test_imports") or []],
        "tests": [str(t) for t in extra["tests"]],
        "base_count": int(base_count) if isinstance(base_count, int | float) else None,
        "timeout_s": float(timeout_s) if isinstance(timeout_s, int | float) else None,
    }


def evaluate_response(response: str, sample_extra: dict[str, Any]) -> CodeEvalResult:
    """Extract code from ``response`` and run the sample's asserts on it.

    The per-assert timeout is the larger of CODE_EVAL_TIMEOUT_SECONDS and the
    task's own ``timeout_s`` (set by the dataset build from the canonical
    solution's wall-clock), so slow-but-legitimate inputs are not false
    timeouts.
    """
    if not settings.code_eval_enabled:
        raise CodeEvalDisabled(
            "The 'tests' metric executes model-written code and is disabled. "
            "Set CODE_EVAL_ENABLED=true in API/.env to turn it on."
        )
    fields = code_fields(sample_extra)
    if fields is None:
        raise ValueError("The sample has no 'tests' in its extra_data")
    entry_point = fields["entry_point"]
    code = extract_code(response, entry_point) or ""
    timeout_s = max(settings.code_eval_timeout_seconds, fields["timeout_s"] or 0.0)
    return run_tests(
        code,
        fields["tests"],
        fields["test_imports"],
        entry_point,
        timeout_s=timeout_s,
        memory_mb=settings.code_eval_memory_mb,
        base_count=fields["base_count"],
    )


def to_evalplus_sample(task_id: str, code: str) -> dict[str, str]:
    """One line of EvalPlus's samples.jsonl."""
    return {"task_id": task_id, "solution": code}


__all__ = [
    "HARNESS",
    "MAX_TASK_SECONDS",
    "RELIABILITY_GUARD",
    "AssertResult",
    "CodeEvalDisabled",
    "CodeEvalResult",
    "StatementRun",
    "code_fields",
    "evaluate_response",
    "extract_code",
    "run_statements",
    "run_tests",
    "task_timeout",
    "to_evalplus_sample",
]
