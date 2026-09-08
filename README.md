# PromptCraft

Prompt optimization that measures instead of guessing. Runs entirely on your machine through Ollama.

Paste a rough prompt and get a rewrite, a score and a diff. Give it 10–20 input/output examples and it scores every candidate on held-out data, picks few-shot examples for coverage, and runs GEPA to evolve the instructions from its own misses. No API keys, nothing leaves the box.

![PromptCraft demo](docs/demo.gif)

## Why

Most prompt-optimization tools either send your prompts to a cloud API or hand back a rewrite with no evidence it is better than what you started with. PromptCraft keeps everything local and treats "is it better?" as a measurement question: a candidate only wins if it scores higher on samples it was not tuned on. If the rewrite scores below your original, it says so.

## Quick start

Requires [Ollama](https://ollama.ai), Node 18+ and Python 3.11+.

```bash
git clone https://github.com/shirkattack/PromptCraft
cd PromptCraft

ollama pull llama3.2           # default model
ollama pull nomic-embed-text   # optional: few-shot selection and dedup fall back to simpler methods without it

npm run install:web
npm run install:api
cp API/.env.example API/.env

npm run dev
```

Frontend at http://localhost:3000, API at http://127.0.0.1:8000 (interactive API docs at `/docs`). The schema is managed with Alembic and migrations run on API startup.

To verify the install: `./test_setup.sh`, then `python test_optimization.py` with the API running. [TESTING_GUIDE.md](TESTING_GUIDE.md) has the manual walkthrough.

## What it does

- **Rewrite and score.** Original vs. optimized side by side, a 0–100 score, and a word-level diff.
- **Held-out evaluation.** With a dataset, every candidate (original, rewrite, and few-shot variants of each) is scored on samples it was not tuned on. Fixed split or k-fold, class-balanced for label data.
- **Coverage-based few-shot selection.** Examples are chosen to cover the space of training inputs using nomic-embed-text embeddings, not "the first ones that passed".
- **GEPA.** Reflective prompt evolution: run the prompt, hand feedback on every miss to a reflection model, let it rewrite the instructions, keep a Pareto front. Full lineage with a diff per generation.
- **Feedback as constraints.** Thumbs-down a result with a note ("too long", "changed the meaning") and the next run treats it as a constraint.
- **Try it.** Run original vs. optimized on any input you type.
- **Dataset generation.** Generate more samples locally, with embedding-based near-duplicate rejection.
- **Separate reflection model.** Let a bigger model write instructions while a small one handles the scoring calls.
- **Session history** in SQLite, with export and import.

## How scoring works

**No dataset.** The score is a rubric, and the UI shows the itemised breakdown: 50 base points, plus points for a rewrite that is 1.2–3× the length of the original, uses structure, and mentions examples or an output format; a rewrite past 3× loses points for verbosity. It catches vague prompts and nothing more. Do not read it as accuracy.

**With a dataset** of input/output pairs:

1. Candidates are built: the original prompt, the rewrite, and each of those with selected few-shot examples.
2. Each candidate is scored on held-out samples with the chosen metric: `exact`, `contains`, `llm_judge` (a local model judges free-text answers), or `tests` (the answer's code is run against the sample's asserts, see [Coding benchmarks](#coding-benchmarks-mbpp)). `auto` picks `tests` when every sample carries asserts, `contains` for short answers and the judge for longer ones.
3. Evaluation is a fixed 80/20 split by default, or k-fold, where every sample is held out once. Splits and folds are stratified so every class appears in every fold.
4. Few-shot examples are selected for coverage of the training inputs, so they are representative rather than lucky.
5. The winner is whichever candidate scores highest held-out.

**GEPA** ([Agrawal et al., 2025](https://arxiv.org/abs/2507.19457)) adds a loop on top, built on `dspy.teleprompt.GEPA`:

1. Run the current prompt on training samples.
2. The metric scores each answer and writes feedback for every miss. It gives full credit for an exact label, half credit when the right label is present but buried in a longer answer, and says which case it was.
3. The reflection model reads the prompt and the feedback and proposes rewritten instructions.
4. The proposal is scored held-out and added to a Pareto front of candidates.
5. Repeat until the budget of scored calls runs out. The UI shows the lineage with a word diff at each step.

PromptCraft supplies the feedback metric, the split and budget controls, the separate reflection model, and the lineage view; the search itself is DSPy's.

**Budgets.** A run uses at most 40 train and 20 held-out samples, 5 folds and 8 few-shot examples by default; GEPA gets 60 scored calls. That keeps a run to minutes on a laptop. Raise the `EVAL_MAX_*` values in `API/.env` and the GEPA budget in the run settings for stricter numbers.

## Coding benchmarks (MBPP+)

PromptCraft can optimize a prompt for Python code generation and score it by running tests. The dataset is MBPP+ from [EvalPlus](https://github.com/evalplus/evalplus): 378 short programming tasks, each with a function name and three asserts.

**Warning.** The `tests` metric executes code written by the model on your machine, inside a subprocess with a timeout, a memory cap and a guard that disables `os.system`, file deletion, process spawning and sockets. It is not a container. Run it on a machine you don't mind, and set `CODE_EVAL_ENABLED=false` in `API/.env` to switch it off.

**Build and import the dataset.**

```bash
uv run --project API --with-requirements API/requirements-bench.txt \
    python scripts/build_mbppplus_dataset.py --out docs/benchmarks/mbppplus --seed 1234
```

`--with-requirements` overlays EvalPlus on the API environment for that one command. It is not an API dependency and `uv sync` never installs it.

This writes `train.jsonl` (120 tasks), `val.jsonl` (60) and `test.jsonl` (198) plus `split.json` with the task ids, and checks that every canonical solution passes its own asserts in the sandbox. Import `train.jsonl` through the Import dialog (JSON Lines): `input` and `output` map as usual and `task_id`, `entry_point`, `test_imports` and `tests` land in each sample's extra data, where the metric reads them. The app holds part of the imported dataset out as its own dev split. `test.jsonl` is never used during optimization; it is only for the cross-check below.

**How it scores.** The metric extracts the first fenced code block (or the `def`) from the answer, checks that it defines the task's function, and runs each assert in its own interpreter. A sample passes only when every assert passes, which is pass@1 as MBPP+ defines it. GEPA's feedback metric additionally scores the fraction of asserts passed and says why the rest failed (no code, syntax error, wrong function name, the failing assert, an exception, or a timeout), so the reflection model has something to act on. The reported score of a run is always the binary pass@1, never the fraction.

**Recommended run settings for code.** Temperature 0, thinking off, a `max_tokens` of about 512, and few-shot examples placed before the task input (the default rendering). Raise `EVAL_MAX_TRAIN_SAMPLES` to use all 120 training tasks; the default caps keep a run to 60 samples.

**Cross-check with EvalPlus.** Export a session's completions on the fixed test split, once for the prompt as written and once for the optimized prompt, and score them with the official harness:

```bash
uv run --project API python scripts/export_evalplus_samples.py --session <session id> --variant original  --split test --out samples-original.jsonl
uv run --project API python scripts/export_evalplus_samples.py --session <session id> --variant optimized --split test --out samples-optimized.jsonl
uv run --project API --with-requirements API/requirements-bench.txt evalplus.evaluate --dataset mbpp --samples samples-optimized.jsonl
uv run --project API python scripts/evalplus_split_score.py --split test --results samples-optimized_eval_results.json
```

The export prints PromptCraft's own pass@1 on the base asserts. EvalPlus needs every MBPP+ task in the file, so tasks outside the split are written with empty solutions and its headline pass@1 covers all 378; the last command reads its per-task results for the split alone, with and without the extended tests. Report PromptCraft's number, EvalPlus's base number (they should agree to within a task or two) and EvalPlus's extended number.

On macOS, prefix `evalplus.evaluate` with `EVALPLUS_MAX_MEMORY_BYTES=-1`. EvalPlus sets a memory cap that macOS rejects, and without this every task is reported as a timeout.

**Contamination.** llama3.2 and most models have seen MBPP during training. Absolute scores say little; compare within one model, before and after optimization, on the fixed test split.

## Results

All runs: llama3.2 3B via Ollama on a MacBook Pro (Apple M4 Pro, 48 GB), on the 18-sample support-ticket priority dataset in [`docs/examples/support-tickets.csv`](docs/examples/support-tickets.csv) (12 hand-written tickets plus 6 generated in the app). Prompt: *Classify the priority of this support ticket as high, medium or low.* Metric: `contains`, with half credit when the right label is present but buried in a longer answer.

| Method | Original | Best candidate | Protocol | Wall-clock |
|---|---|---|---|---|
| Meta-prompt rewrite | 44% | 50% | 5-fold, every sample held out once. Best candidate was the rewrite + 4 examples; the rewrite alone scored 39%, below the original. | 4m 47s |
| GEPA, seed 1 | 50% | 67% | 50/50 split, 9 held out (3 per class), 60 scored calls. | 34s |
| GEPA, seed 2 | 28% | 44% | same | 50s |
| GEPA, seed 3 | 39% | 89% | same | 41s |

The "original" numbers differ from run to run because each split holds out different samples, and with 9 of them one sample is 11 points.

The baseline misses are the same in every run: the right label buried in a sentence or two of explanation (the feedback reads, roughly, *the right answer is in the response but buried in 38 words; respond with the label alone, no explanation*), or the model asking a clarifying question instead of classifying. The seed-3 winner defines each priority level and ends with an instruction to answer with a single word.

At the default 80/20 split (4 held out), three seeds produced no improvement at all: 25%, 50% and 25% before and after. With 4 held-out samples one sample is 25 points, and no proposal beat the original, so the run returned the original prompt, as it should. The 50/50 runs above set `train_ratio` on the GEPA service directly; the UI always uses the 80/20 default, so a bigger dataset is the way to get a bigger held-out set from the app.

GEPA picks candidates on the same held-out samples it reports, so its scores are optimistic. Nine held-out samples is a demonstration, not a benchmark. 7–8B results are next.

## Configuration

`API/.env`:

```
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL_NAME=llama3.2:latest
EMBEDDING_MODEL=nomic-embed-text:latest

EVAL_MAX_TRAIN_SAMPLES=40
EVAL_MAX_DEV_SAMPLES=20
EVAL_MAX_FOLDS=5
EVAL_MAX_DEMOS=8

CODE_EVAL_ENABLED=true
CODE_EVAL_TIMEOUT_SECONDS=10
CODE_EVAL_MEMORY_MB=1024

DATABASE_URL=sqlite:///./app.db
LOG_LEVEL=INFO
```

No provider API keys exist in this codebase. Any model in `ollama list` can be selected in the UI; the reflection model is chosen separately. `REQUIRE_API_KEY` / `API_KEY` protect the API itself and are off by default. See `API/.env.example` for everything else.

## Architecture

```
API/   FastAPI + DSPy + SQLAlchemy/SQLite
       app/services/optimization_service.py   rewrite, heuristic score, candidate assembly
       app/services/eval_service.py           splits, folds, metrics, held-out scoring
       app/services/gepa_service.py           feedback metric, dspy.GEPA run, lineage tracking
       app/services/code_eval_service.py      sandboxed test runner behind the "tests" metric
       app/services/embedding_service.py      coverage-based example selection, dedup
       app/services/training_service.py       datasets, import/export, synthetic samples
       app/services/try_service.py            side-by-side "Try it"
Web/   Next.js 15, TypeScript, Tailwind, shadcn/ui, Recharts
```

DSPy sits between the optimizer and the model: signatures define the input/output contract, `Predict` and `ChainOfThought` do the work, and the same code runs against any Ollama model.

## Limitations

- Small models are weak reflectors. Put a larger model on that step with the reflection model picker.
- The no-dataset score is a rubric about the prompt's shape. It says nothing about whether the prompt works.
- `llm_judge` is a small local model judging free text. Expect noise; prefer `exact` or `contains` whenever answers are short.
- Budgets are capped by default. Raise them for real runs and expect longer wall-clock time.
- The `tests` metric runs model-written code in a subprocess, not a container. Its guard blocks the obvious damage, not a determined adversary.

## Contributing

Issues and PRs welcome. If you add a metric or an optimizer, include a run in the Results table.

## Credits

- [DSPy](https://github.com/stanfordnlp/dspy)
- Agrawal et al., *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning*, [arXiv:2507.19457](https://arxiv.org/abs/2507.19457)

## License

MIT. See [LICENSE](LICENSE).
