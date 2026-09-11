#!/usr/bin/env python3
"""Re-score stored benchmark runs after a change to how answers are scored.

    uv run --project API python scripts/rescore_results.py --results docs/results \\
        --reason "extraction keeps imports above the function" [--dry-run]

Every run was made at temperature 0 through DSPy, whose disk cache keeps each
raw model answer. For each ``seed*.json`` this rebuilds the run's program
(instructions and demos) and replays the test split from that cache; the model
is never called, and an answer missing from the cache keeps its old verdict and
is reported. Each answer is extracted again with the current ``extract_code``;
where the code differs from the stored completion the sandbox runs it again.
The sandbox gives the same verdict for the same code (the replayed seeds of the
deterministic methods agree task for task), so unchanged code keeps its
verdict. Also fills in which answers hit max_tokens.

Updates ``test_rows``, the test counts and the completions file, and appends
what changed to ``rescored``. Val scores are not recomputed, and a GEPA run's
search is not redone: both used the scoring of the day.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "API"))
# An address nothing listens on: a cache miss fails at once instead of
# quietly asking the model for a fresh answer.
NO_SERVER = "http://127.0.0.1:9"


def _runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "run_benchmark", REPO_ROOT / "scripts" / "run_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def replay_settings(config: dict[str, Any]) -> list[dict[str, Any]]:
    """LM keyword sets that may match how the run called the model.

    DSPy's cache key includes every request parameter. Runs record their call
    timeout; runs made before it was recorded are tried with the values the
    runner and the app used, with and without keep_alive.
    """
    runner = _runner()
    think = {"think": False} if config.get("thinking") == "off" else {}
    recorded = (config.get("timeouts_s") or {}).get("task")
    timeouts: list[int | None] = [recorded] if recorded else [300, 120, None]
    settings = []
    for timeout in timeouts:
        for keep_alive in (runner.KEEP_ALIVE, None):
            extra = dict(think)
            if keep_alive is not None:
                extra["keep_alive"] = keep_alive
            if timeout is not None:
                extra["timeout"] = timeout
            settings.append(extra)
    return settings


def replay_lm(config: dict[str, Any], extra: dict[str, Any]) -> Any:
    """The task LM as the run configured it, pointed at no server."""
    import dspy  # noqa: PLC0415

    return dspy.LM(
        f"ollama_chat/{config['model']['tag']}",
        api_base=NO_SERVER,
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        num_retries=0,
        **extra,
    )


def find_replay_lm(config: dict[str, Any], program: Any, probe: Any) -> Any | None:
    """The first LM setting whose answer to ``probe`` is in the cache."""
    import dspy  # noqa: PLC0415

    for extra in replay_settings(config):
        lm = replay_lm(config, extra)
        try:
            with dspy.context(lm=lm):
                program(input=probe.input_text)
        except Exception:
            continue
        return lm
    return None


def rescore_run(
    path: Path, dataset: Path, reason: str, dry_run: bool
) -> dict[str, Any]:
    import dspy  # noqa: PLC0415

    from app.services.code_eval_service import (  # noqa: PLC0415
        evaluate_response,
        extract_code,
    )

    runner = _runner()
    data = json.loads(path.read_text())
    config = data["config"]
    train = {runner.task_id(s): s for s in runner.load_split(dataset, "train")}
    test = {runner.task_id(s): s for s in runner.load_split(dataset, "test")}
    program = runner.build_program(
        data["final_prompt"], [train[d] for d in data.get("demos") or []]
    )
    samples_path = path.with_suffix(".samples.jsonl")
    completions = {
        row["task_id"]: row
        for row in map(json.loads, samples_path.read_text().splitlines())
    }

    lm = find_replay_lm(config, program, test[data["test_rows"][0]["task_id"]])
    if lm is None:
        return {
            "run": str(path),
            "skipped": "no LM setting found whose answers are in the DSPy cache",
        }
    counter = runner.TruncationCounter()
    lm_logger = logging.getLogger("dspy.clients.lm")
    lm_logger.addHandler(counter)
    changes: list[dict[str, Any]] = []
    misses: list[str] = []
    try:
        with dspy.context(lm=lm):
            for row in data["test_rows"]:
                tid = row["task_id"]
                sample = test[tid]
                before = counter.count
                try:
                    response = str(program(input=sample.input_text).output or "")
                except Exception:
                    misses.append(tid)
                    continue
                row["truncated"] = counter.count > before
                extra = sample.extra_data or {}
                code = extract_code(response, extra.get("entry_point", "")) or ""
                if code == (completions[tid].get("solution") or ""):
                    continue
                result = evaluate_response(response, extra)
                changes.append(
                    {
                        "task_id": tid,
                        "status": [row["status"], result.status],
                        "plus_pass": [bool(row["plus_pass"]), result.plus_pass],
                        "base_pass": [bool(row["base_pass"]), result.base_pass],
                    }
                )
                row.update(
                    status=result.status,
                    passed=result.passed,
                    total=result.total,
                    base_pass=result.base_pass,
                    plus_pass=result.plus_pass,
                )
                completions[tid]["solution"] = code
    finally:
        lm_logger.removeHandler(counter)

    rows = data["test_rows"]
    n = len(rows)
    old = (data["test"]["base_pass"], data["test"]["plus_pass"])
    base = sum(bool(r["base_pass"]) for r in rows)
    plus = sum(bool(r["plus_pass"]) for r in rows)
    summary = {
        "run": str(
            path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
        ),
        "changed": len(changes),
        "cache_misses": len(misses),
        "base": [old[0], base],
        "plus": [old[1], plus],
        "truncated": sum(1 for r in rows if r.get("truncated")),
    }
    if dry_run:
        return summary
    data["test"].update(
        base_pass=base,
        plus_pass=plus,
        base_pass_at_1=round(base / n * 100, 2) if n else 0.0,
        plus_pass_at_1=round(plus / n * 100, 2) if n else 0.0,
        truncated=summary["truncated"],
    )
    data.setdefault("rescored", []).append(
        {
            "at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "commit": runner.git_commit(),
            "reason": reason,
            "base": summary["base"],
            "plus": summary["plus"],
            "cache_misses": misses,
            "changed": changes,
        }
    )
    path.write_text(json.dumps(data, indent=2) + "\n")
    with samples_path.open("w", encoding="utf-8") as handle:
        for tid in (r["task_id"] for r in rows):
            handle.write(json.dumps(completions[tid]) + "\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "docs" / "results")
    parser.add_argument(
        "--dataset", type=Path, default=REPO_ROOT / "docs" / "benchmarks" / "mbppplus"
    )
    parser.add_argument("--reason", required=True)
    parser.add_argument("--only", default="*/*/*", help="glob under --results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("dspy.clients.lm").setLevel(logging.WARNING)
    for noisy in ("LiteLLM", "httpx", "dspy.utils.parallelizer", "dspy.adapters"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    paths = sorted(args.results.glob(f"{args.only}/seed*.json"))
    if not paths:
        raise SystemExit(f"no runs under {args.results}/{args.only}")
    for path in paths:
        summary = rescore_run(path, args.dataset, args.reason, args.dry_run)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
