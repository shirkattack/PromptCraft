#!/usr/bin/env python3
"""Build the MBPP+ (EvalPlus) train/val/test split PromptCraft optimizes against.

    uv run --project API --with-requirements API/requirements-bench.txt \
        python scripts/build_mbppplus_dataset.py --out docs/benchmarks/mbppplus --seed 1234

Loads MBPP+ via ``evalplus`` (falling back to the Hugging Face ``datasets``
copy), writes one JSONL row per task in the importer's ``input``/``output``
schema with the test fields in the remaining keys, splits the task ids
deterministically, and finally runs every canonical solution through the same
sandbox the ``tests`` metric uses, aborting if any of them fails its asserts.

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
    "source",
)


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

    rows: dict[str, dict[str, Any]] = {}
    for task_id, task in get_mbpp_plus().items():
        setup, asserts = split_test_block(task["assertion"])
        code = task["canonical_solution"].strip("\n")
        entry_point = task.get("entry_point") or entry_point_from_assert(
            asserts[0], code
        )
        rows[task_id] = {
            "input": task["prompt"],
            "output": code,
            "task_id": task_id,
            "entry_point": entry_point,
            "test_imports": setup,
            "tests": asserts,
            "source": source,
        }
    return rows, source


def load_from_datasets() -> tuple[dict[str, dict[str, Any]], str]:
    from datasets import load_dataset  # noqa: PLC0415 - optional

    dataset = load_dataset("evalplus/mbppplus", split="test")
    source = f"{SOURCE} (datasets)"
    rows: dict[str, dict[str, Any]] = {}
    for item in dataset:
        task_id = f"Mbpp/{item['task_id']}"
        code = str(item["code"]).strip("\n")
        asserts = [str(t).strip() for t in item["test_list"] if str(t).strip()]
        imports = [
            str(t).strip() for t in item.get("test_imports") or [] if str(t).strip()
        ]
        entry_point = item.get("entry_point") or entry_point_from_assert(
            asserts[0], code
        )
        prompt = str(item["prompt"]).strip()
        rows[task_id] = {
            # EvalPlus presents the task as a docstring: description plus the
            # sample assert that reveals the function name.
            "input": f'"""\n{prompt}\n{asserts[0]}\n"""\n',
            "output": code,
            "task_id": task_id,
            "entry_point": entry_point,
            "test_imports": imports,
            "tests": asserts,
            "source": source,
        }
    return rows, source


def load_tasks() -> tuple[dict[str, dict[str, Any]], str]:
    try:
        return load_from_evalplus()
    except ImportError as exc:
        print(
            f"evalplus unavailable ({exc}); falling back to datasets", file=sys.stderr
        )
        return load_from_datasets()


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


# -- sanity check --------------------------------------------------------------


def check_canonical_solutions(
    rows: dict[str, dict[str, Any]], timeout_s: float
) -> None:
    from app.services.code_eval_service import run_tests  # noqa: PLC0415

    failures: list[str] = []
    for index, (task_id, row) in enumerate(rows.items(), start=1):
        result = run_tests(
            row["output"],
            row["tests"],
            row["test_imports"],
            row["entry_point"],
            timeout_s=timeout_s,
            memory_mb=1024,
        )
        if result.status != "pass":
            failures.append(f"{task_id}: {result.status} - {result.feedback}")
        if index % 50 == 0:
            print(f"  checked {index}/{len(rows)}", file=sys.stderr)
    if failures:
        print(
            "Canonical solutions that do not pass their own asserts:", file=sys.stderr
        )
        print("\n".join(failures), file=sys.stderr)
        raise SystemExit(
            f"{len(failures)} canonical solutions failed; not writing the dataset"
        )
    print(f"canonical solutions: {len(rows)}/{len(rows)} pass")


# -- writing -------------------------------------------------------------------


def write_dataset(
    rows: dict[str, dict[str, Any]],
    split: dict[str, list[str]],
    out: Path,
    seed: int,
    source: str,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for task_id in split[name]:
                row = {key: rows[task_id][key] for key in ROW_KEYS}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    sizes = {name: len(split[name]) for name in SPLIT_NAMES}
    (out / "split.json").write_text(
        json.dumps({"seed": seed, "sizes": sizes, **split}, indent=2) + "\n"
    )
    today = dt.date.today().isoformat()
    (out / "README.md").write_text(
        f"""# MBPP+ for PromptCraft

Built by `scripts/build_mbppplus_dataset.py` on {today}.

- Source: {source}
- Tasks: {len(rows)}
- Split: seed {seed}, train {sizes["train"]} / val {sizes["val"]} / test {sizes["test"]}
  (`split.json` lists the task ids of each split, so the split is reproducible
  without the seed)
- Every canonical solution passed its own asserts in PromptCraft's sandbox when
  the files were built.

Each row is one task in the importer's schema: `input` is the task as EvalPlus
presents it (description plus the assert that reveals the function name),
`output` is the canonical solution (used only as a few-shot example, never as a
scoring target), and `task_id`, `entry_point`, `test_imports`, `tests`, `source`
land in the sample's `extra_data`, where the `tests` metric reads them.

**`test.jsonl` is never used during optimization.** `train.jsonl` is for
few-shot examples and GEPA's training samples, `val.jsonl` is for GEPA/Pareto
selection, and `test.jsonl` is only for reporting, through
`scripts/export_evalplus_samples.py`.
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
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="seconds per assert in the sanity check",
    )
    parser.add_argument(
        "--skip-check", action="store_true", help="do not run the canonical solutions"
    )
    args = parser.parse_args(argv)

    rows, source = load_tasks()
    print(f"{len(rows)} tasks from {source}")
    missing = [
        tid for tid, row in rows.items() if not row["entry_point"] or not row["tests"]
    ]
    if missing:
        raise SystemExit(f"Tasks without entry_point or tests: {missing[:10]}")

    split = split_ids(list(rows), args.seed)
    print("split:", " / ".join(f"{name} {len(split[name])}" for name in SPLIT_NAMES))

    if not args.skip_check:
        check_canonical_solutions(rows, args.timeout)

    write_dataset(rows, split, args.out, args.seed, source)
    print(f"wrote {args.out}/{{train,val,test}}.jsonl, split.json, README.md")


if __name__ == "__main__":
    main()
