#!/usr/bin/env python3
"""Run prompt-optimization methods on a coding benchmark and score them on its test split.

    uv run --project API python scripts/run_benchmark.py \\
        --dataset docs/benchmarks/mbppplus \\
        --model llama3.2 --reflection-model qwen3.6:27b \\
        --prompt bare --methods original one_line random_demos coverage_demos gepa gepa_demos \\
        --seeds 1 2 3 --budget 500 --demos 4 --out docs/results

Calls the services directly: no HTTP, none of the app's EVAL_MAX_* caps.
``train.jsonl`` is the demo pool and GEPA's training set, ``val.jsonl`` is
GEPA's selection set and where every candidate's val score comes from, and
``test.jsonl`` is read exactly once per finished candidate, at the end. The
runner asserts the test ids are disjoint from everything the optimizer sees.

Every (model, prompt, method, seed) writes
``<out>/<model>/<prompt>/<method>/seed<N>.json`` plus the completions as
``seed<N>.samples.jsonl`` the moment it finishes, and is skipped on a rerun if
that file exists. Temperature 0, 512 tokens, thinking off for the task model.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "API"))

PROMPTS = {
    "bare": "Write a Python function for the task below.",
    "fixed": (
        "Write a Python function for the task below. "
        "Respond with a single fenced Python code block and nothing else."
    ),
}
ONE_LINE = "Respond with a single fenced Python code block and nothing else."
METHODS = (
    "original",
    "one_line",
    "random_demos",
    "coverage_demos",
    "gepa",
    "gepa_demos",
)
MAX_TOKENS = 512
REFLECTION_MAX_TOKENS = 2048
KEEP_ALIVE = "2h"

log = logging.getLogger("benchmark")


# -- data ----------------------------------------------------------------------


def load_split(dataset: Path, name: str) -> list[Any]:
    from app.services.eval_service import Sample  # noqa: PLC0415

    path = dataset / f"{name}.jsonl"
    samples = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            extra = {k: v for k, v in row.items() if k not in ("input", "output")}
            samples.append(Sample(row["input"], row["output"], extra))
    if not samples:
        raise SystemExit(f"{path} is empty")
    return samples


def task_id(sample: Any) -> str:
    return str((sample.extra_data or {}).get("task_id"))


def assert_disjoint(train: list[Any], val: list[Any], test: list[Any]) -> None:
    """The optimizer must never see a test task."""
    seen = {task_id(s) for s in train} | {task_id(s) for s in val}
    leaked = sorted(seen & {task_id(s) for s in test})
    if leaked:
        raise AssertionError(f"test ids also in train/val: {leaked[:5]}")
    if len(seen) != len(train) + len(val):
        raise AssertionError("duplicate task ids across train and val")


# -- models --------------------------------------------------------------------


def ollama_get(base_url: str, path: str, body: dict[str, Any] | None = None) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        method="POST" if body is not None else "GET",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def model_info(base_url: str, model: str) -> dict[str, Any]:
    """Digest and capabilities of an Ollama model tag."""
    tag = model if ":" in model else f"{model}:latest"
    tags = ollama_get(base_url, "/api/tags")
    digest = next(
        (m["digest"] for m in tags.get("models", []) if m["name"] == tag), None
    )
    if digest is None:
        raise SystemExit(f"Model {tag} is not pulled; run: ollama pull {tag}")
    show = ollama_get(base_url, "/api/show", {"model": tag})
    return {
        "tag": tag,
        "digest": digest,
        "capabilities": list(show.get("capabilities") or []),
    }


def preload(base_url: str, tag: str) -> None:
    """Load the model and ask Ollama to keep it resident between calls."""
    ollama_get(base_url, "/api/generate", {"model": tag, "keep_alive": KEEP_ALIVE})


def make_lm(info: dict[str, Any], *, max_tokens: int) -> Any:
    from app.services.lm_manager import LMManager  # noqa: PLC0415

    extra: dict[str, Any] = {}
    if "thinking" in info["capabilities"]:
        extra["think"] = False  # Ollama rejects the field on models that cannot think
    return LMManager.get_lm(
        provider="ollama",
        model_name=info["tag"],
        temperature=0.0,
        max_tokens=max_tokens,
        **extra,
    )


# -- programs and evaluation ---------------------------------------------------


def fenced(code: str) -> str:
    return f"```python\n{code.strip()}\n```"


def build_program(instructions: str, demos: list[Any]) -> Any:
    import dspy  # noqa: PLC0415

    program = dspy.Predict(dspy.Signature("input -> output", instructions.strip()))
    # Demos go before the task input, so Ollama's prefix cache is reused.
    program.demos = [
        dspy.Example(input=d.input_text, output=fenced(d.expected_output)).with_inputs(
            "input"
        )
        for d in demos
    ]
    return program


class ThinkBlockError(RuntimeError):
    pass


def evaluate(program: Any, samples: list[Any], label: str) -> dict[str, Any]:
    """Run the program on every sample and score it; per-task rows and counts."""
    from app.services.code_eval_service import (  # noqa: PLC0415
        evaluate_response,
        extract_code,
    )

    started = time.time()
    rows: list[dict[str, Any]] = []
    completions: dict[str, str] = {}
    for index, sample in enumerate(samples, start=1):
        tid = task_id(sample)
        try:
            response = str(program(input=sample.input_text).output or "")
        except Exception as exc:  # one failed call is an empty answer
            log.warning("%s %s: model call failed: %s", label, tid, exc)
            response = ""
        if "<think>" in response:
            raise ThinkBlockError(f"{label} {tid}: response contains a <think> block")
        completions[tid] = response
        result = evaluate_response(response, sample.extra_data or {})
        rows.append(
            {
                "task_id": tid,
                "status": result.status,
                "passed": result.passed,
                "total": result.total,
                "base_pass": result.base_pass,
                "plus_pass": result.plus_pass,
                "code": extract_code(response, sample.extra_data.get("entry_point", ""))
                or "",
            }
        )
        if index % 25 == 0 or index == len(samples):
            base = sum(r["base_pass"] for r in rows)
            plus = sum(r["plus_pass"] for r in rows)
            log.info(
                "%s %d/%d base %d plus %d (%.0fs)",
                label,
                index,
                len(samples),
                base,
                plus,
                time.time() - started,
            )
    n = len(rows)
    base = sum(r["base_pass"] for r in rows)
    plus = sum(r["plus_pass"] for r in rows)
    return {
        "n": n,
        "base_pass": base,
        "plus_pass": plus,
        "base_pass_at_1": round(base / n * 100, 2) if n else 0.0,
        "plus_pass_at_1": round(plus / n * 100, 2) if n else 0.0,
        "rows": rows,
        "completions": completions,
        "seconds": round(time.time() - started, 1),
    }


# -- methods -------------------------------------------------------------------


def pick_demos(method: str, train: list[Any], k: int, seed: int) -> list[Any]:
    if method in ("random_demos",):
        return random.Random(seed).sample(train, k)
    if method in ("coverage_demos", "gepa_demos"):
        from app.services.embedding_service import coverage_selection  # noqa: PLC0415

        inputs = [s.input_text for s in train]
        chosen, _ = coverage_selection(inputs, inputs, k)
        return [train[i] for i in chosen]
    return []


def run_gepa(
    prompt: str,
    train: list[Any],
    val: list[Any],
    budget: int,
    seed: int,
    reflection_lm: Any,
) -> dict[str, Any]:
    from app.services.gepa_service import GepaOptimizer  # noqa: PLC0415

    optimizer = GepaOptimizer(
        train + val,
        metric="tests",
        budget=budget,
        reflection_lm=reflection_lm,
        seed=seed,
        train=train,
        dev=val,
    )
    outcome = optimizer.run(prompt)
    return {
        "instructions": outcome["instructions"],
        "improved": outcome["gepa"]["improved"],
        "baseline_val_pass_at_1": outcome["gepa"]["baseline_score"],
        "final_val_pass_at_1": outcome["gepa"]["final_score"],
        "metric_calls": outcome["gepa"]["metric_calls"],
        "iterations": outcome["gepa"]["iterations"],
        "timeline": outcome["gepa"]["timeline"],
        "seconds": outcome["gepa"]["elapsed_seconds"],
    }


def result_path(
    out: Path, model: str, prompt_name: str, method: str, seed: int
) -> Path:
    safe_model = model.replace(":", "-").replace("/", "-")
    return out / safe_model / prompt_name / method / f"seed{seed}.json"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def run_one(
    *,
    method: str,
    seed: int,
    prompt_name: str,
    prompt: str,
    train: list[Any],
    val: list[Any],
    test: list[Any],
    lm: Any,
    reflection_lm: Any,
    args: argparse.Namespace,
    config: dict[str, Any],
    gepa_cache: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    import dspy  # noqa: PLC0415

    started = time.time()
    instructions = prompt
    if method == "one_line":
        instructions = f"{prompt} {ONE_LINE}"
    gepa: dict[str, Any] | None = None
    with dspy.context(lm=lm):
        if method in ("gepa", "gepa_demos"):
            if seed not in gepa_cache:
                gepa_cache[seed] = run_gepa(
                    prompt, train, val, args.budget, seed, reflection_lm
                )
            gepa = gepa_cache[seed]
            instructions = gepa["instructions"]
        demos = pick_demos(method, train, args.demos, seed)
        program = build_program(instructions, demos)
        optimization_seconds = round(time.time() - started, 1)

        val_result = evaluate(program, val, f"{method} seed{seed} val")
        test_result = evaluate(program, test, f"{method} seed{seed} test")

    return {
        "config": {
            **config,
            "method": method,
            "seed": seed,
            "prompt_name": prompt_name,
            "prompt": prompt,
        },
        "final_prompt": instructions,
        "demos": [task_id(d) for d in demos],
        "gepa": gepa,
        "val": {
            k: v for k, v in val_result.items() if k not in ("rows", "completions")
        },
        "test": {
            k: v for k, v in test_result.items() if k not in ("rows", "completions")
        },
        "test_rows": [
            {k: v for k, v in r.items() if k != "code"} for r in test_result["rows"]
        ],
        "wall_clock": {
            "optimization_seconds": optimization_seconds,
            "val_eval_seconds": val_result["seconds"],
            "test_eval_seconds": test_result["seconds"],
            "total_seconds": round(time.time() - started, 1),
        },
        "finished_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "_samples": [
            {"task_id": r["task_id"], "solution": r["code"]}
            for r in test_result["rows"]
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset", type=Path, default=REPO_ROOT / "docs" / "benchmarks" / "mbppplus"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--reflection-model", default=None, help="defaults to --model")
    parser.add_argument(
        "--prompt", default="bare", help="bare, fixed, or a path to a text file"
    )
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--budget", type=int, default=500, help="GEPA scored calls")
    parser.add_argument("--demos", type=int, default=4)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "results")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    from app.core.config import settings  # noqa: PLC0415

    if args.prompt in PROMPTS:
        prompt_name, prompt = args.prompt, PROMPTS[args.prompt]
    else:
        path = Path(args.prompt)
        prompt_name, prompt = path.stem, path.read_text().strip()

    train = load_split(args.dataset, "train")
    val = load_split(args.dataset, "val")
    test = load_split(args.dataset, "test")
    assert_disjoint(train, val, test)
    log.info(
        "train %d val %d test %d; test ids disjoint", len(train), len(val), len(test)
    )

    base_url = settings.ollama_base_url
    task_info = model_info(base_url, args.model)
    reflection_info = model_info(base_url, args.reflection_model or args.model)
    preload(base_url, task_info["tag"])
    lm = make_lm(task_info, max_tokens=MAX_TOKENS)
    reflection_lm = make_lm(reflection_info, max_tokens=REFLECTION_MAX_TOKENS)
    log.info(
        "task model %s (%s); reflection %s (%s)",
        task_info["tag"],
        task_info["digest"][:12],
        reflection_info["tag"],
        reflection_info["digest"][:12],
    )

    config = {
        "dataset": str(args.dataset),
        "model": task_info,
        "reflection_model": reflection_info,
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "reflection_max_tokens": REFLECTION_MAX_TOKENS,
        "thinking": "off"
        if "thinking" in task_info["capabilities"]
        else "not supported by model",
        "budget": args.budget,
        "demos": args.demos,
        "code_eval_timeout_seconds": settings.code_eval_timeout_seconds,
        "code_eval_memory_mb": settings.code_eval_memory_mb,
        "git_commit": git_commit(),
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
    }

    methods = list(args.methods)
    if prompt_name == "fixed" and "one_line" in methods:
        log.info("skipping one_line for the fixed prompt: it already carries the line")
        methods.remove("one_line")

    gepa_cache: dict[int, dict[str, Any]] = {}
    batch_started = time.time()
    for method in methods:
        for seed in args.seeds:
            path = result_path(args.out, args.model, prompt_name, method, seed)
            if path.exists():
                log.info(
                    "skip %s (exists)",
                    path.relative_to(REPO_ROOT)
                    if path.is_relative_to(REPO_ROOT)
                    else path,
                )
                if method == "gepa" and seed not in gepa_cache:
                    stored = json.loads(path.read_text())
                    if stored.get("gepa"):
                        gepa_cache[seed] = stored["gepa"]
                continue
            log.info(
                "=== %s / %s / %s / seed %d", args.model, prompt_name, method, seed
            )
            result = run_one(
                method=method,
                seed=seed,
                prompt_name=prompt_name,
                prompt=prompt,
                train=train,
                val=val,
                test=test,
                lm=lm,
                reflection_lm=reflection_lm,
                args=args,
                config=config,
                gepa_cache=gepa_cache,
            )
            samples = result.pop("_samples")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.with_suffix(".samples.jsonl").open(
                "w", encoding="utf-8"
            ) as handle:
                for row in samples:
                    handle.write(json.dumps(row) + "\n")
            path.write_text(json.dumps(result, indent=2) + "\n")
            log.info(
                "done %s seed %d: val plus %s%% test base %s%% plus %s%% in %ss",
                method,
                seed,
                result["val"]["plus_pass_at_1"],
                result["test"]["base_pass_at_1"],
                result["test"]["plus_pass_at_1"],
                result["wall_clock"]["total_seconds"],
            )
    log.info("batch finished in %.0fs", time.time() - batch_started)


if __name__ == "__main__":
    main()
