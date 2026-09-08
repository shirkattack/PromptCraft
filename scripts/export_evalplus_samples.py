#!/usr/bin/env python3
"""Export a session's completions on a benchmark split in EvalPlus's samples.jsonl format.

    python scripts/export_evalplus_samples.py --session <id> --split test --out samples.jsonl

Runs the session's optimized prompt (its instructions plus the few-shot
examples that were measured) on every task of the split, extracts the code
from each answer and writes one ``{"task_id", "solution"}`` line per task.
Completions are stored on the session (``completions_json``) so a second
export does not re-run the model; pass ``--regenerate`` to force it.

Cross-check with the official harness:

    evalplus.evaluate --dataset mbpp --samples samples.jsonl

The script also prints PromptCraft's own pass@1 on the split (base asserts
only) so the two numbers can be compared. Needs Ollama and the session's model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "API"))

SPLITS = ("train", "val", "test")


def load_split(benchmark: Path, split: str) -> list[dict[str, Any]]:
    path = benchmark / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(
            f"{path} not found; run scripts/build_mbppplus_dataset.py first"
        )
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def session_program(session: Any) -> tuple[str, list[dict[str, str]]]:
    """Instructions and demos of the measured program, from the stored result."""
    instructions = session.optimized_prompt or session.original_prompt
    demos: list[dict[str, str]] = []
    if session.result_json:
        try:
            evaluation = (
                json.loads(session.result_json).get("metadata", {}).get("eval") or {}
            )
        except json.JSONDecodeError:
            evaluation = {}
        instructions = evaluation.get("instructions") or instructions
        demos = [
            {"input": str(d["input"]), "output": str(d["output"])}
            for d in evaluation.get("demos") or []
            if isinstance(d, dict) and "input" in d and "output" in d
        ]
    return instructions, demos


def complete_tasks(
    session: Any,
    tasks: list[dict[str, Any]],
    cached: dict[str, str],
    temperature: float,
    max_tokens: int,
) -> dict[str, str]:
    import dspy  # noqa: PLC0415

    from app.services.lm_manager import LMManager  # noqa: PLC0415

    todo = [t for t in tasks if t["task_id"] not in cached]
    if not todo:
        return cached
    instructions, demos = session_program(session)
    lm = LMManager.get_lm(
        provider=session.provider,
        model_name=session.model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    program = dspy.Predict(dspy.Signature("input -> output", instructions.strip()))
    program.demos = [dspy.Example(**d).with_inputs("input") for d in demos]
    print(
        f"completing {len(todo)} tasks with {session.model} ({len(demos)} demos)",
        file=sys.stderr,
    )
    started = time.time()
    with dspy.context(lm=lm):
        for index, task in enumerate(todo, start=1):
            try:
                cached[task["task_id"]] = str(program(input=task["input"]).output or "")
            except Exception as exc:  # one failed call is an empty completion
                print(f"  {task['task_id']}: {exc}", file=sys.stderr)
                cached[task["task_id"]] = ""
            if index % 10 == 0 or index == len(todo):
                print(
                    f"  {index}/{len(todo)} ({time.time() - started:.0f}s)",
                    file=sys.stderr,
                )
    return cached


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--session", required=True, help="optimization session id")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--out", type=Path, default=Path("samples.jsonl"))
    parser.add_argument(
        "--benchmark", type=Path, default=REPO_ROOT / "docs" / "benchmarks" / "mbppplus"
    )
    parser.add_argument(
        "--regenerate", action="store_true", help="ignore stored completions"
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args(argv)

    from app.core.database import SessionLocal  # noqa: PLC0415
    from app.models.optimization import OptimizationSession  # noqa: PLC0415
    from app.services.code_eval_service import (  # noqa: PLC0415
        extract_code,
        run_tests,
        to_evalplus_sample,
    )

    tasks = load_split(args.benchmark, args.split)
    db = SessionLocal()
    try:
        session = db.get(OptimizationSession, args.session)
        if session is None:
            raise SystemExit(f"Session {args.session} not found")
        if not session.optimized_prompt:
            raise SystemExit(f"Session {args.session} has no optimized prompt yet")

        stored: dict[str, dict[str, str]] = {}
        if session.completions_json and not args.regenerate:
            try:
                stored = json.loads(session.completions_json)
            except json.JSONDecodeError:
                stored = {}
        completions = complete_tasks(
            session,
            tasks,
            dict(stored.get(args.split) or {}),
            args.temperature,
            args.max_tokens,
        )
        stored[args.split] = completions
        session.completions_json = json.dumps(stored)
        db.commit()
    finally:
        db.close()

    passed = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for task in tasks:
            completion = completions.get(task["task_id"], "")
            code = extract_code(completion, task["entry_point"]) or completion
            handle.write(json.dumps(to_evalplus_sample(task["task_id"], code)) + "\n")
            result = run_tests(
                code,
                task["tests"],
                task.get("test_imports") or [],
                task["entry_point"],
                timeout_s=10.0,
                memory_mb=1024,
            )
            passed += result.status == "pass"
    print(f"wrote {len(tasks)} samples to {args.out}")
    print(
        f"PromptCraft pass@1 on {args.split} (base asserts): {passed}/{len(tasks)} = {passed / len(tasks) * 100:.1f}%"
    )
    print(f"cross-check: evalplus.evaluate --dataset mbpp --samples {args.out}")


if __name__ == "__main__":
    main()
