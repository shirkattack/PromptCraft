#!/usr/bin/env python3
"""Build the MBPP+ (EvalPlus) train/val/test split PromptCraft optimizes against.

    uv run --project API --with-requirements API/requirements-bench.txt \\
        python scripts/build_mbppplus_dataset.py --out docs/benchmarks/mbppplus --seed 1234

Loads MBPP+ via ``evalplus`` (falling back to the Hugging Face ``datasets``
copy), turns every base and plus input into an assert by running the
canonical solution on it in PromptCraft's sandbox, writes one JSONL row per
task in the importer's ``input``/``output`` schema, splits the task ids
deterministically, and finally runs every canonical solution against its own
asserts, aborting if any fails.

``--with-requirements`` overlays evalplus on the API environment for this run
only; it is not an API dependency.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import datetime as dt
import json
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "API"))

SIZES = (120, 60, 198)  # train / val / test; sums to the 378 MBPP+ tasks
SPLIT_NAMES = ("train", "val", "test")
SOURCE = "evalplus/mbppplus"
ROW_KEYS = (
    "input",
    "output",
    "task_id",
    "entry_point",
    "test_imports",
    "tests",
    "base_count",
    "timeout_s",
    "source",
)
MAX_REPR_CHARS = 20_000

# EvalPlus's special oracles (evalplus/eval/_special_oracle.py), mirrored so a
# solution passes here exactly when it passes there.
SET_EQ_ENTRY_POINTS = {
    "similar_elements",  # Mbpp/2
    "find_char_long",  # Mbpp/7
    "common_in_nested_lists",  # Mbpp/111
    "extract_singly",  # Mbpp/140
    "larg_nnum",  # Mbpp/232
    "intersection_array",  # Mbpp/249
    "find_dissimilar",  # Mbpp/579
    "Diff",  # Mbpp/769
}
NOT_NONE_ENTRY_POINTS = {"check_str", "text_match_three", "text_starta_endb"}
ANY_OUTPUT_ENTRY_POINTS = {"are_equivalent"}  # Mbpp/164: any output accepted
ZERO_OR_EXPECTED_ENTRY_POINTS = {"sum_div"}  # Mbpp/295: 0 is also accepted
FLOAT_ATOL = 1e-6  # EvalPlus's default tolerance for float-typed expected values
ALT_ORACLES = {
    # Mbpp/581: the height may be read as the perpendicular height.
    "surface_Area": """\
import math as __math
def __alt_surface_Area(base_edge, height):
    slant_height = __math.sqrt((base_edge / 2) ** 2 + height**2)
    return round(base_edge**2 + 4 * (base_edge * slant_height) / 2)""",
    # Mbpp/558: the two numbers may be zero-padded to the same length first.
    "digit_distance_nums": """\
def __alt_digit_distance_nums(num1, num2):
    a, b = str(num1), str(num2)
    n = max(len(a), len(b))
    return sum(abs(int(x) - int(y)) for x, y in zip(a.zfill(n), b.zfill(n)))""",
}
ALT_OK_HELPER = """\
def __alt_ok(out, expected, alt, tol):
    if out == expected:
        return True
    if isinstance(out, (int, float)) and isinstance(alt, (int, float)):
        return abs(out - alt) <= tol
    return out == alt"""
BUILD_TIMEOUT_S = 10.0
TIMEOUT_FACTOR = 4.0
TIMEOUT_MIN_S = 2.0
TIMEOUT_MAX_S = 20.0

# Comparison for asserts with a tolerance, mirroring EvalPlus: exact match
# first, otherwise the same type and numpy-style allclose (rtol 1e-7).
CLOSE_HELPER = """\
def __close(out, exp, atol):
    if out == exp:
        return True
    if type(out) is not type(exp):
        return False
    def ok(a, b):
        if isinstance(a, bool) or isinstance(b, bool):
            return a == b
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(a - b) <= atol + 1e-7 * abs(b)
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return len(a) == len(b) and all(ok(x, y) for x, y in zip(a, b))
        return a == b
    return ok(out, exp)"""

# Turns a value into Python source that evaluates back to it. Unlike repr it
# writes float('inf'), and it refuses nan and non-literal objects, so the
# build can skip inputs whose expected value cannot be written into an assert.
LITERAL_SOURCE = """\
def __literal(v):
    if isinstance(v, bool) or v is None:
        return repr(v)
    if isinstance(v, float):
        if v != v:
            raise ValueError("nan")
        if v in (float("inf"), float("-inf")):
            return "float('inf')" if v > 0 else "float('-inf')"
        return repr(v)
    if isinstance(v, (int, str, bytes, complex)):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(__literal(x) for x in v) + "]"
    if isinstance(v, tuple):
        inner = ", ".join(__literal(x) for x in v)
        return "(" + inner + ("," if len(v) == 1 else "") + ")"
    if isinstance(v, frozenset):
        return "frozenset(" + __literal(set(v)) + ")"
    if isinstance(v, set):
        return "{" + ", ".join(__literal(x) for x in v) + "}" if v else "set()"
    if isinstance(v, dict):
        return "{" + ", ".join(__literal(k) + ": " + __literal(x) for k, x in v.items()) + "}"
    raise ValueError(type(v).__name__)


def __is_floats(v):
    if isinstance(v, float):
        return True
    return isinstance(v, (list, tuple)) and bool(v) and all(isinstance(i, float) for i in v)"""

# Setup for the capture pass: record the repr of each canonical output and
# whether it round-trips through ``ast.literal_eval`` (so it can be written
# into an assert).
CAPTURE_SETUP = (
    LITERAL_SOURCE
    + """
import json as __json
__collect = open("collect.jsonl", "w", encoding="utf-8")
def __capture(index, value):
    if isinstance(value, dict) and type(value) is not dict:
        value = dict(value)  # Counter, defaultdict: compare as a plain dict
    try:
        text = __literal(value)
        ok = len(text) <= MAX_REPR_CHARS and eval(text) == value
    except Exception:
        text, ok = repr(value)[:200], False
    __collect.write(__json.dumps({"index": index, "repr": text, "ok": bool(ok), "floats": __is_floats(value)}) + "\\n")
    __collect.flush()""".replace("MAX_REPR_CHARS", str(MAX_REPR_CHARS))
)

# The same literalizer, for the build process itself (call arguments).
_literal_ns: dict[str, Any] = {}
exec(LITERAL_SOURCE, _literal_ns)  # noqa: S102 - our own source


# -- loading -------------------------------------------------------------------


def split_test_block(block: str) -> tuple[list[str], list[str]]:
    """(setup lines, asserts) from a block of test code, one statement each.

    EvalPlus's assert strings can use a stdlib module (``math.isclose``)
    without importing it, because its own harness never executes them as
    written. Such modules get an ``import`` line in the setup list.
    """
    tree = ast.parse(block)
    setup: list[str] = []
    asserts: list[str] = []
    for node in tree.body:
        segment = ast.get_source_segment(block, node) or ""
        if not segment.strip():
            continue
        (asserts if isinstance(node, ast.Assert) else setup).append(segment.strip())
    imported = {
        alias.asname or alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }
    used = {
        node.value.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
    }
    for name in sorted(used - imported):
        if name in sys.stdlib_module_names and not hasattr(builtins, name):
            setup.append(f"import {name}")
    return setup, asserts


def entry_point_from_assert(assertion: str, code: str) -> str:
    """The function an assert calls that the solution defines."""
    defined = set(re.findall(r"^\s*def\s+(\w+)\s*\(", code, re.M))
    tree = ast.parse(assertion)
    called = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    for name in called:
        if name in defined:
            return name
    if called:
        return called[-1]  # innermost call is usually the task function
    raise ValueError(f"No function call in assert: {assertion!r}")


def load_from_evalplus() -> tuple[dict[str, dict[str, Any]], str]:
    """Raw tasks keyed by id: prompt, solution, entry point, inputs, atol."""
    import evalplus  # noqa: PLC0415 - optional, see requirements-bench.txt
    from evalplus.data import get_mbpp_plus

    version = getattr(evalplus, "__version__", "unknown")
    try:
        from evalplus.data import mbpp as mbpp_module

        data_version = getattr(mbpp_module, "MBPP_PLUS_VERSION", "")
    except Exception:  # pragma: no cover - layout differs between releases
        data_version = ""
    source = f"{SOURCE} (evalplus {version}" + (
        f", MBPP+ {data_version})" if data_version else ")"
    )
    tasks: dict[str, dict[str, Any]] = {}
    for task_id, task in get_mbpp_plus().items():
        setup, sample_asserts = split_test_block(task["assertion"])
        code = task["canonical_solution"].strip("\n")
        tasks[task_id] = {
            "task_id": task_id,
            "prompt": task["prompt"],
            "code": code,
            "entry_point": task.get("entry_point")
            or entry_point_from_assert(sample_asserts[0], code),
            "setup": setup,
            "base_input": list(task["base_input"]),
            "plus_input": list(task.get("plus_input") or []),
            "atol": float(task.get("atol") or 0),
        }
    return tasks, source


def load_from_datasets() -> tuple[dict[str, dict[str, Any]], str]:
    """Fallback source. It has no plus inputs, so the build carries only the
    base asserts and says so in the source string."""
    from datasets import load_dataset  # noqa: PLC0415 - optional

    dataset = load_dataset("evalplus/mbppplus", split="test")
    source = f"{SOURCE} (datasets; base inputs only)"
    tasks: dict[str, dict[str, Any]] = {}
    for item in dataset:
        task_id = f"Mbpp/{item['task_id']}"
        code = str(item["code"]).strip("\n")
        asserts = [str(t).strip() for t in item["test_list"] if str(t).strip()]
        entry_point = item.get("entry_point") or entry_point_from_assert(
            asserts[0], code
        )
        prompt = str(item["prompt"]).strip()
        inputs = []
        for assertion in asserts:
            call = next(
                (
                    n
                    for n in ast.walk(ast.parse(assertion))
                    if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id == entry_point
                ),
                None,
            )
            if call is not None:
                inputs.append([ast.literal_eval(a) for a in call.args])
        tasks[task_id] = {
            "task_id": task_id,
            "prompt": f'"""\n{prompt}\n{asserts[0]}\n"""\n',
            "code": code,
            "entry_point": entry_point,
            "setup": [str(t).strip() for t in item.get("test_imports") or []],
            "base_input": inputs,
            "plus_input": [],
            "atol": 0.0,
        }
    return tasks, source


def load_tasks() -> tuple[dict[str, dict[str, Any]], str]:
    try:
        return load_from_evalplus()
    except ImportError as exc:
        print(
            f"evalplus unavailable ({exc}); falling back to datasets", file=sys.stderr
        )
        return load_from_datasets()


# -- asserts from inputs -------------------------------------------------------


def args_literal(args: list[Any]) -> str | None:
    """Source for the call arguments, or None if it would not round-trip."""
    try:
        text = ", ".join(_literal_ns["__literal"](a) for a in args)
        back = eval(f"({text},)") if args else ()  # noqa: S307 - our own literal
    except Exception:
        return None
    return text if list(back) == list(args) else None


def format_assert(
    entry_point: str,
    args_text: str,
    expected: str,
    atol: float,
    is_floats: bool = False,
) -> str:
    """One assert for one input, following EvalPlus's oracle for the task.

    Tolerance applies when the task has one, or when the expected value is a
    float (or list/tuple of floats), where EvalPlus defaults to 1e-6.
    """
    call = f"{entry_point}({args_text})"
    if entry_point in ANY_OUTPUT_ENTRY_POINTS:
        return call  # must run without raising; the value is not checked
    if entry_point in ZERO_OR_EXPECTED_ENTRY_POINTS:
        return f"assert {call} in ({expected}, 0)"
    if entry_point in SET_EQ_ENTRY_POINTS:
        return f"assert set({call}) == set({expected})"
    if entry_point in NOT_NONE_ENTRY_POINTS:
        return f"assert ({call} is not None) == {expected != 'None'}"
    if entry_point in ALT_ORACLES:
        return f"assert __alt_ok({call}, {expected}, __alt_{entry_point}({args_text}), {atol!r})"
    tolerance = atol or (FLOAT_ATOL if is_floats else 0.0)
    if tolerance:
        return f"assert __close({call}, {expected}, {tolerance!r})"
    return f"assert {call} == {expected}"


def setup_for(task: dict[str, Any]) -> list[str]:
    """The task's test_imports: its own setup plus any oracle helpers."""
    setup = list(task["setup"])
    if any("__close(" in t for t in task.get("_tests", [])):
        setup.append(CLOSE_HELPER)
    if task["entry_point"] in ALT_ORACLES:
        setup += [ALT_ORACLES[task["entry_point"]], ALT_OK_HELPER]
    return setup


def capture_expected(
    task: dict[str, Any], timeout_s: float = BUILD_TIMEOUT_S
) -> tuple[dict[int, str], dict[str, int]]:
    """Run the canonical solution on every input; expected reprs by input index.

    Returns the reprs that round-trip, and counts of skipped inputs by reason.
    Also records the argument literals on the task (``_literals``).
    """
    from app.services.code_eval_service import run_statements  # noqa: PLC0415

    inputs = task["base_input"] + task["plus_input"]
    statements: list[str] = []
    skipped = {"args_not_literal": 0, "output_not_literal": 0, "raised": 0}
    literal_for: dict[int, str] = {}
    for index, args in enumerate(inputs):
        text = args_literal(list(args))
        if text is None:
            skipped["args_not_literal"] += 1
            continue
        literal_for[index] = text
        statements.append(f"__capture({index}, {task['entry_point']}({text}))")
    run = run_statements(
        task["code"],
        statements,
        [CAPTURE_SETUP, *task["setup"]],
        timeout_s=timeout_s,
        memory_mb=1024,
        collect="collect.jsonl",
    )
    if run.module_error is not None:
        raise SystemExit(
            f"{task['task_id']}: canonical solution failed to load: "
            f"{run.module_error.message}"
        )
    expected: dict[int, str] = {}
    captured: set[int] = set()
    floats: dict[int, bool] = {}
    for line in run.collected.splitlines():
        row = json.loads(line)
        captured.add(int(row["index"]))
        if row["ok"]:
            expected[int(row["index"])] = row["repr"]
            floats[int(row["index"])] = bool(row.get("floats"))
        else:
            skipped["output_not_literal"] += 1
    skipped["raised"] = len(literal_for) - len(captured)
    task["_literals"] = literal_for
    task["_floats"] = floats
    return expected, skipped


def asserts_for(
    task: dict[str, Any], expected: dict[int, str]
) -> tuple[list[str], int]:
    """The task's asserts, base inputs first; returns them and the base count."""
    base_n = len(task["base_input"])
    literals = task["_literals"]
    floats = task.get("_floats", {})

    def one(i: int) -> str:
        return format_assert(
            task["entry_point"],
            literals[i],
            expected[i],
            task["atol"],
            floats.get(i, False),
        )

    base = [one(i) for i in range(base_n) if i in expected]
    plus = [
        one(i) for i in range(base_n, base_n + len(task["plus_input"])) if i in expected
    ]
    task["_tests"] = base + plus
    return base + plus, len(base)


def clamp_timeout(slowest_assert_s: float) -> float:
    return round(
        min(max(TIMEOUT_FACTOR * slowest_assert_s, TIMEOUT_MIN_S), TIMEOUT_MAX_S), 2
    )


def build_rows(
    tasks: dict[str, dict[str, Any]], source: str
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Capture expected outputs, form asserts, verify the canonical solution."""
    from app.services.code_eval_service import run_tests  # noqa: PLC0415

    rows: dict[str, dict[str, Any]] = {}
    skipped_total = {"args_not_literal": 0, "output_not_literal": 0, "raised": 0}
    failures: list[str] = []
    for index, (task_id, task) in enumerate(tasks.items(), start=1):
        expected, skipped = capture_expected(task)
        for key, value in skipped.items():
            skipped_total[key] += value
        tests, base_count = asserts_for(task, expected)
        setup = setup_for(task)
        if not tests or base_count == 0:
            failures.append(f"{task_id}: no usable base asserts")
            continue
        result = run_tests(
            task["code"],
            tests,
            setup,
            task["entry_point"],
            timeout_s=BUILD_TIMEOUT_S,
            memory_mb=1024,
            base_count=base_count,
        )
        if result.status != "pass":
            failures.append(f"{task_id}: {result.status} - {result.feedback}")
            continue
        slowest = max((a.elapsed for a in result.asserts), default=0.0)
        rows[task_id] = {
            "input": task["prompt"],
            "output": task["code"],
            "task_id": task_id,
            "entry_point": task["entry_point"],
            "test_imports": setup,
            "tests": tests,
            "base_count": base_count,
            "timeout_s": clamp_timeout(slowest),
            "source": source,
        }
        if index % 50 == 0:
            print(f"  built {index}/{len(tasks)}", file=sys.stderr)
    if failures:
        print(
            "Tasks whose canonical solution does not pass its asserts:", file=sys.stderr
        )
        print("\n".join(failures), file=sys.stderr)
        raise SystemExit(f"{len(failures)} tasks failed; not writing the dataset")
    return rows, skipped_total


# -- splitting -----------------------------------------------------------------


def split_ids(
    task_ids: list[str], seed: int, sizes: tuple[int, int, int] = SIZES
) -> dict[str, list[str]]:
    """Deterministic, disjoint train/val/test id lists covering every task."""
    if sum(sizes) != len(task_ids):
        raise SystemExit(
            f"Split sizes {sizes} sum to {sum(sizes)} but there are {len(task_ids)} tasks"
        )
    ids = sorted(task_ids, key=_task_sort_key)
    random.Random(seed).shuffle(ids)
    train, val = sizes[0], sizes[0] + sizes[1]
    return {"train": ids[:train], "val": ids[train:val], "test": ids[val:]}


def _task_sort_key(task_id: str) -> tuple[str, int]:
    prefix, _, number = task_id.rpartition("/")
    return (prefix, int(number) if number.isdigit() else -1)


# -- writing -------------------------------------------------------------------


def summary(rows: dict[str, dict[str, Any]]) -> dict[str, float | int]:
    counts = [len(r["tests"]) for r in rows.values()]
    return {
        "tasks": len(rows),
        "asserts": sum(counts),
        "base_asserts": sum(r["base_count"] for r in rows.values()),
        "mean_per_task": round(statistics.mean(counts), 1) if counts else 0,
        "max_per_task": max(counts, default=0),
        "timeout_max": max((r["timeout_s"] for r in rows.values()), default=0),
    }


def split_text(split: dict[str, list[str]], seed: int) -> str:
    sizes = {name: len(split[name]) for name in SPLIT_NAMES}
    return json.dumps({"seed": seed, "sizes": sizes, **split}, indent=2) + "\n"


def write_dataset(
    rows: dict[str, dict[str, Any]],
    split: dict[str, list[str]],
    out: Path,
    seed: int,
    source: str,
    skipped: dict[str, int],
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    split_file = out / "split.json"
    text = split_text(split, seed)
    if split_file.exists() and split_file.read_text() != text:
        raise SystemExit(
            f"{split_file} would change; the split is fixed. Delete it on purpose "
            "if a new split is really wanted."
        )
    for name in SPLIT_NAMES:
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for task_id in split[name]:
                row = {key: rows[task_id][key] for key in ROW_KEYS}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    split_file.write_text(text)
    sizes = {name: len(split[name]) for name in SPLIT_NAMES}
    stats = summary(rows)
    today = dt.date.today().isoformat()
    (out / "README.md").write_text(
        f"""# MBPP+ for PromptCraft

Built by `scripts/build_mbppplus_dataset.py` on {today}.

- Source: {source}
- Tasks: {stats["tasks"]}
- Asserts: {stats["asserts"]} in total ({stats["base_asserts"]} base, the rest
  EvalPlus's extended inputs); mean {stats["mean_per_task"]} and max
  {stats["max_per_task"]} per task
- Inputs skipped because their arguments or the canonical output do not
  round-trip through `repr`: {sum(skipped.values())}
  (arguments {skipped["args_not_literal"]}, outputs {skipped["output_not_literal"]},
  raised {skipped["raised"]})
- Split: seed {seed}, train {sizes["train"]} / val {sizes["val"]} / test {sizes["test"]}
  (`split.json` lists the task ids of each split; it is fixed and the build
  refuses to change it)
- Every canonical solution passed all of its asserts in PromptCraft's sandbox
  when the files were built.

Each row is one task in the importer's schema: `input` is the task as EvalPlus
presents it (description plus the assert that reveals the function name),
`output` is the canonical solution (used only as a few-shot example, never as a
scoring target), and the remaining keys land in the sample's `extra_data`:

- `tests`: one `assert` per input, the base inputs first. Each expected value
  was produced by running the canonical solution on that input in the sandbox.
  Tasks with a tolerance use `__isclose(...)`, defined in `test_imports`.
- `base_count`: how many leading asserts are the base tests. A task's *base*
  result is those passing; its *plus* result is every assert passing.
- `timeout_s`: per-assert timeout, `clamp(4 x slowest canonical assert, 2, 20)`
  seconds; the runner uses the larger of this and `CODE_EVAL_TIMEOUT_SECONDS`.
- `entry_point`, `test_imports`, `task_id`, `source`.

**`test.jsonl` is never used during optimization.** `train.jsonl` is for
few-shot examples and GEPA's training samples, `val.jsonl` is for GEPA/Pareto
selection and candidate comparison, and `test.jsonl` is evaluated once per
finished candidate by `scripts/run_benchmark.py`.
"""
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "docs" / "benchmarks" / "mbppplus"
    )
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)

    tasks, source = load_tasks()
    print(f"{len(tasks)} tasks from {source}")
    split = split_ids(list(tasks), args.seed)
    print("split:", " / ".join(f"{name} {len(split[name])}" for name in SPLIT_NAMES))

    rows, skipped = build_rows(tasks, source)
    stats = summary(rows)
    print(
        f"tasks {stats['tasks']}, asserts {stats['asserts']} "
        f"({stats['base_asserts']} base), mean {stats['mean_per_task']} and max "
        f"{stats['max_per_task']} per task, skipped inputs {sum(skipped.values())} "
        f"({skipped}), slowest timeout_s {stats['timeout_max']}"
    )
    print(f"canonical solutions: {stats['tasks']}/{len(tasks)} pass")

    write_dataset(rows, split, args.out, args.seed, source, skipped)
    print(f"wrote {args.out}/{{train,val,test}}.jsonl, split.json, README.md")


if __name__ == "__main__":
    main()
