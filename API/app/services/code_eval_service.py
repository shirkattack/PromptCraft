"""Run model-written Python against a task's asserts in a sandboxed subprocess.

This is the ``tests`` metric behind coding benchmarks such as MBPP+: the model
answers a task with code, and the code is scored by executing each of the
task's ``assert`` statements in a fresh interpreter with a timeout, a memory
limit and a guard that disables the destructive parts of ``os``/``shutil``/
``subprocess``. Standard library only, so the API gains no dependency.

It is a subprocess sandbox, not a container. Treat it as "good enough for a
machine you don't mind" and see the README's warning before turning it on.
"""

import ast
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
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

STDERR_TAIL_LINES = 20
FEEDBACK_ASSERT_CHARS = 120

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


class CodeEvalDisabled(Exception):
    """Raised when the ``tests`` metric is used with CODE_EVAL_ENABLED=false."""


@dataclass
class CodeEvalResult:
    status: CodeStatus
    passed: int  # asserts passed
    total: int  # asserts run
    feedback: str  # one to three lines a prompt author or reflector can act on
    stderr_tail: str  # last ~20 lines of stderr, for debugging only

    @property
    def fraction(self) -> float:
        return self.passed / self.total if self.total else 0.0


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
    """preexec_fn: cap CPU time and memory; best effort per platform."""
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX
        return
    limits = [
        (getattr(resource, "RLIMIT_CPU", None), 60),
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


def _tail(stderr: str) -> str:
    lines = stderr.strip().splitlines()
    return "\n".join(lines[-STDERR_TAIL_LINES:])


def _exception_summary(stderr: str) -> tuple[str, str]:
    """(exception type, message) from the last non-empty stderr line."""
    for line in reversed(stderr.strip().splitlines()):
        line = line.strip()
        if not line or line.startswith(("File ", "Traceback", "^")):
            continue
        exc_type, _, message = line.partition(":")
        return exc_type.strip(), message.strip()
    return "Error", ""


def run_tests(
    code: str,
    tests: list[str],
    test_imports: list[str],
    entry_point: str,
    *,
    timeout_s: float,
    memory_mb: int,
) -> CodeEvalResult:
    """Execute each assert separately against ``code``; count passes.

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
        )

    header = RELIABILITY_GUARD + "\n".join(test_imports) + "\n" + code + "\n"
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"}
    passed = 0
    first_failure: tuple[str, str, str] | None = None  # (assert, exc type, message)
    stderr_tail = ""

    tmpdir = tempfile.mkdtemp(prefix="promptcraft-code-")
    try:
        for index, assertion in enumerate(tests):
            path = os.path.join(tmpdir, f"case_{index}.py")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(header + "\n" + assertion + "\n")
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", "-S", path],
                    cwd=tmpdir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    preexec_fn=lambda: _limit_resources(memory_mb),
                )
            except subprocess.TimeoutExpired as exc:
                stderr = (
                    exc.stderr.decode()
                    if isinstance(exc.stderr, bytes)
                    else (exc.stderr or "")
                )
                return CodeEvalResult(
                    "timeout",
                    passed,
                    total,
                    f"Timed out after {timeout_s:g}s on `{_short(assertion)}`; "
                    "likely an infinite loop or exponential solution.",
                    _tail(stderr),
                )
            if proc.returncode == 0:
                passed += 1
                continue
            if first_failure is None:
                exc_type, message = _exception_summary(proc.stderr)
                first_failure = (assertion, exc_type, message)
                stderr_tail = _tail(proc.stderr)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if first_failure is None:
        return CodeEvalResult(
            "pass", passed, total, f"Correct: all {total} asserts passed.", ""
        )

    assertion, exc_type, message = first_failure
    if exc_type == "AssertionError":
        detail = f" ({message})" if message else ""
        return CodeEvalResult(
            "assert_failed",
            passed,
            total,
            f"Passed {passed}/{total} asserts. First failure: `{_short(assertion)}`{detail}",
            stderr_tail,
        )
    return CodeEvalResult(
        "exception",
        passed,
        total,
        f"{exc_type}: {_short(message, 80)} while running `{_short(assertion)}`.",
        stderr_tail,
    )


# -- entry points used by the metrics ------------------------------------------


def code_fields(extra: dict[str, Any] | None) -> dict[str, Any] | None:
    """The code-eval fields of a sample's extra_data, or None if it has none."""
    if not extra or not isinstance(extra.get("tests"), list) or not extra["tests"]:
        return None
    return {
        "task_id": extra.get("task_id"),
        "entry_point": str(extra.get("entry_point") or ""),
        "test_imports": [str(line) for line in extra.get("test_imports") or []],
        "tests": [str(t) for t in extra["tests"]],
    }


def evaluate_response(response: str, sample_extra: dict[str, Any]) -> CodeEvalResult:
    """Extract code from ``response`` and run the sample's asserts on it."""
    if not settings.code_eval_enabled:
        raise CodeEvalDisabled(
            "The 'tests' metric executes model-written code and is disabled. "
            "Set CODE_EVAL_ENABLED=true in API/.env to turn it on."
        )
    fields = code_fields(sample_extra)
    if fields is None:
        raise ValueError("The sample has no 'tests' in its extra_data")
    entry_point = fields["entry_point"]
    code = extract_code(response, entry_point)
    if code is None:
        return run_tests(
            "",
            fields["tests"],
            fields["test_imports"],
            entry_point,
            timeout_s=settings.code_eval_timeout_seconds,
            memory_mb=settings.code_eval_memory_mb,
        )
    return run_tests(
        code,
        fields["tests"],
        fields["test_imports"],
        entry_point,
        timeout_s=settings.code_eval_timeout_seconds,
        memory_mb=settings.code_eval_memory_mb,
    )


def to_evalplus_sample(task_id: str, code: str) -> dict[str, str]:
    """One line of EvalPlus's samples.jsonl."""
    return {"task_id": task_id, "solution": code}


__all__ = [
    "RELIABILITY_GUARD",
    "CodeEvalDisabled",
    "CodeEvalResult",
    "code_fields",
    "evaluate_response",
    "extract_code",
    "run_tests",
    "to_evalplus_sample",
]
