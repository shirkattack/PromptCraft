# PromptCraft

Prompt optimization that measures instead of guessing. Runs entirely on your machine through Ollama.

Paste a rough prompt and get a rewrite, a score and a diff. Give it 10–20 input/output examples and it scores every candidate on held-out data, picks few-shot examples for coverage, and runs GEPA to evolve the instructions from its own misses. No API keys, nothing leaves the box.

![PromptCraft](public/promptcraft.png)
<!-- TODO: swap for docs/demo.gif once it's committed -->

## Why

Most prompt-optimization tools either send your prompts to a cloud API or hand back a rewrite with no evidence it is better than what you started with. PromptCraft keeps everything local and treats "is it better?" as a measurement question: a candidate only wins if it scores higher on samples it was not tuned on. If the winner turns out to be your original prompt plus three examples, it says so.

## Quick start

Requires [Ollama](https://ollama.ai), Node 18+ and Python 3.11+.

```bash
git clone https://github.com/shirkattack/PromptCraft
cd PromptCraft

ollama pull llama3.2           # default model
ollama pull nomic-embed-text   # embeddings for few-shot selection and dedup
# TODO: confirm tag, and whether the app pulls this itself

npm run install:web
npm run install:api
cp API/.env.example API/.env

npm run dev
```

Frontend at http://localhost:3000, API at http://127.0.0.1:8000 (interactive API docs at `/docs`).

To verify the install: `./test_setup.sh`, then `python test_optimization.py` with the API running. [TESTING_GUIDE.md](TESTING_GUIDE.md) has the manual walkthrough.

## What it does

- **Rewrite and score.** Original vs. optimized side by side, a 0–100 score, and a word-level diff.
- **Held-out evaluation.** With a dataset, every candidate (original, rewrite, and few-shot variants of each) is scored on samples it was not tuned on. K-fold, class-balanced for label data.
- **Coverage-based few-shot selection.** Examples are chosen to cover the space of training inputs using nomic-embed-text embeddings, not "the first ones that passed".
- **GEPA.** Reflective prompt evolution: run the prompt, write feedback for every miss, have a reflection model rewrite the instructions, keep a Pareto front. Full lineage with a diff per generation.
- **Feedback as constraints.** Thumbs-down a result with a note ("too long", "changed the meaning") and the next run treats it as a constraint.
- **Try it.** Run original vs. optimized on any input you type.
- **Dataset generation.** Generate more samples locally, with embedding-based near-duplicate rejection.
- **Separate reflection model.** Let a bigger model write instructions while a small one handles the scoring calls.
- **Session history** in SQLite, with export and import.

## How scoring works

**No dataset.** The score is a structural heuristic and the UI labels it as one. It catches vague prompts and nothing more.
<!-- TODO: one line on what the heuristic actually checks -->

**With a dataset** of input/output pairs:

1. Candidates are built: the original prompt, the rewrite, and each of those with selected few-shot examples.
2. Each candidate is evaluated on held-out folds. For classification data the folds are stratified so every class appears in every fold.
3. Few-shot examples are selected for coverage of the training inputs, so they are representative rather than lucky.
4. The winner is whichever candidate scores highest held-out.

**GEPA** ([Agrawal et al., 2025](https://arxiv.org/abs/2507.19457)) adds a loop on top:

1. Run the current prompt on training samples.
2. For every miss, write natural-language feedback on what went wrong.
3. Hand the prompt and the feedback to the reflection model, which proposes rewritten instructions.
4. Score the proposal held-out and add it to a Pareto front of candidates.
5. Repeat for N generations. The UI shows the lineage with a word diff at each step.

Evaluation runs are capped so a run finishes in minutes on a laptop rather than an hour.
<!-- TODO: state the default cap and how to raise it -->

## Results

All runs: llama3.2 3B via Ollama, an 18-sample support-ticket priority dataset.
<!-- TODO: path to the dataset in the repo, hardware, wall-clock time per run -->

| Method | Original | Best candidate | Notes |
|---|---|---|---|
| Meta-prompt rewrite | 50% | 56% | Best candidate was the original prompt + 3 examples. The rewrite lost. |
| GEPA | 25% | 87.5% | Held-out split, 2 generations. |

The two "original" numbers differ because the runs use different protocols: k-fold across all 18 samples for the meta-prompt run, a fixed held-out split for GEPA.
<!-- TODO: confirm this explanation and state the split size (87.5% reads as 7/8) -->

The feedback GEPA wrote before the winning rewrite, roughly: *the correct label is in the answer but buried in 19 words; answer with one word.* The winning rewrite came out of that feedback.

These are small-n demonstrations, not benchmarks. Variance across seeds and results for 7–8B models are next.
<!-- TODO: replace this line with the numbers once you have them -->

## Configuration

`API/.env`:

```
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL_NAME=llama3.2:latest
DATABASE_URL=sqlite:///./app.db
LOG_LEVEL=INFO
API_HOST=127.0.0.1
API_PORT=8000
```

No API keys are needed and none are read. Any model in `ollama list` can be selected in the UI; the reflection model is chosen separately.
<!-- TODO: if OpenAI/Anthropic provider code paths still exist, either remove them or document them as optional and off by default -->

## Architecture

```
API/   FastAPI + DSPy + SQLAlchemy/SQLite
       app/services/optimization_service.py   candidate generation, scoring, GEPA loop
       app/services/ollama_service.py         Ollama client
       app/services/lm_manager.py             DSPy LM setup
Web/   Next.js 15, TypeScript, Tailwind, shadcn/ui, Recharts
```

DSPy sits between the optimizer and the model: signatures define the input/output contract, `Predict` and `ChainOfThought` do the work, and the same code runs against any Ollama model.

## Limitations

- Small models are weak reflectors. Put a larger model on that step with the reflection model picker.
- The no-dataset score is a heuristic. Do not read it as accuracy.
- Eval budgets are capped by default. Raise them for real runs and expect longer wall-clock time.
<!-- TODO: what metric is used for non-label data, if any -->

## Contributing

Issues and PRs welcome. If you add a metric or an optimizer, include a run in the Results table.

## Credits

- [DSPy](https://github.com/stanfordnlp/dspy)
- Agrawal et al., *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning*, [arXiv:2507.19457](https://arxiv.org/abs/2507.19457)

## License

MIT. See [LICENSE](LICENSE).
