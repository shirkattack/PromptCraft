#!/usr/bin/env python3
"""Score one PromptCraft split from an EvalPlus results file.

    uv run --project API python scripts/evalplus_split_score.py \\
        --results samples_eval_results.json --split test

``evalplus.evaluate`` needs every MBPP+ task in its samples file and reports
pass@1 over all of them, so tasks outside the split (exported with an empty
solution) drag its headline number down. This reads the per-task statuses it
writes next to the samples file and reports base and base+extra pass@1 over
the split's task ids only, which is the number to compare with PromptCraft's.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")


def split_task_ids(benchmark: Path, split: str) -> list[str]:
    split_file = benchmark / "split.json"
    if not split_file.exists():
        raise SystemExit(
            f"{split_file} not found; run scripts/build_mbppplus_dataset.py first"
        )
    return list(json.loads(split_file.read_text())[split])


def score(results: dict, task_ids: list[str]) -> dict[str, float | int]:
    evaluated = results.get("eval", results)
    base = plus = missing = 0
    for task_id in task_ids:
        entries = evaluated.get(task_id) or []
        if not entries:
            missing += 1
            continue
        first = entries[0]  # one solution per task in PromptCraft's export
        base_ok = first.get("base_status") == "pass"
        base += base_ok
        # EvalPlus's headline mbpp+ number needs base and plus to pass.
        plus += base_ok and first.get("plus_status") == "pass"
    n = len(task_ids)
    return {
        "tasks": n,
        "missing": missing,
        "base_pass": base,
        "plus_pass": plus,
        "base_pass_at_1": round(base / n * 100, 1) if n else 0.0,
        "plus_pass_at_1": round(plus / n * 100, 1) if n else 0.0,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="*_eval_results.json from evalplus.evaluate",
    )
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument(
        "--benchmark", type=Path, default=REPO_ROOT / "docs" / "benchmarks" / "mbppplus"
    )
    args = parser.parse_args(argv)

    results = json.loads(args.results.read_text())
    summary = score(results, split_task_ids(args.benchmark, args.split))
    if summary["missing"]:
        print(
            f"warning: {summary['missing']} split tasks are missing from {args.results}"
        )
    print(f"EvalPlus on the {args.split} split ({summary['tasks']} tasks)")
    print(
        f"  mbpp  (base tests):        {summary['base_pass']}/{summary['tasks']} = {summary['base_pass_at_1']}%"
    )
    print(
        f"  mbpp+ (base + extra tests): {summary['plus_pass']}/{summary['tasks']} = {summary['plus_pass_at_1']}%"
    )


if __name__ == "__main__":
    main()
