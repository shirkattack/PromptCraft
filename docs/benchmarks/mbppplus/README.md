# MBPP+ for PromptCraft

Built by `scripts/build_mbppplus_dataset.py` on 2026-09-08.

- Source: evalplus/mbppplus (evalplus 0.3.1, MBPP+ v0.2.0)
- Tasks: 378
- Asserts: 40831 in total (1169 base, the rest
  EvalPlus's extended inputs); mean 108.0 and max
  150 per task
- Inputs skipped because their arguments or the canonical output do not
  round-trip through `repr`: 184
  (arguments 12, outputs 172,
  raised 0)
- Split: seed 1234, train 120 / val 60 / test 198
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
