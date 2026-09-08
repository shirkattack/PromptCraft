# MBPP+ for PromptCraft

Built by `scripts/build_mbppplus_dataset.py` on 2026-09-08.

- Source: evalplus/mbppplus (evalplus 0.3.1, MBPP+ v0.2.0)
- Tasks: 378
- Split: seed 1234, train 120 / val 60 / test 198
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
