# Testing guide

Two layers: automated checks that need no model, and a manual walkthrough of the app against a live Ollama.

## Automated

```sh
./test_setup.sh                 # environment: Node, uv, Ollama, models, API import, health endpoints
npm run test:api                # backend suite (pytest); hermetic, no Ollama needed
npm run lint:api                # ruff + strict mypy
cd Web && npx tsc --noEmit      # frontend types; the production build fails on errors
cd Web && npm run build
uv run --project API python test_optimization.py   # end-to-end against a running API (npm run dev:api first)
npm run e2e:install && npm run e2e                  # browser smoke test against a running `npm run dev` (needs Ollama)
```

The backend suite replaces model calls with DSPy's `DummyLM` and embeddings with a deterministic stand-in, so it runs in a few seconds anywhere.

## Manual walkthrough

Start everything with `npm run dev` and open <http://localhost:3000>. Expected times are for `llama3.2` (3B) on a laptop.

### 1. Providers and models

- The Model dropdown lists your Ollama models with parameter size, quantization, context window and capabilities. The configured default is first, then smallest to largest; embedding-only models are hidden.
- Stop Ollama and reload: the provider shows as unavailable with a reason, and Start Optimization is disabled. Start it again and reload.

### 2. Plain rewrite

- Paste `Help me write a python script that makes exact change.`, method Meta-Prompt, Start.
- Progress shows "Rewriting the prompt"; a result appears in a few seconds with a heuristic score badged as such (hover it for the rubric).
- Try DSPy Chain-of-Thought: Optimization Insights shows the model's reasoning. Try Simple as a baseline.
- Rendered / Raw / Compare views work; Copy and Export work; the draft survives a reload (local storage).

### 3. Dataset

- Training Data tab → Import Dataset → `docs/examples/support-tickets.csv` (file or paste). Toast says 12 samples; the dataset appears with a class-balanced quality badge.
- Preview samples; Export JSON and CSV download files; Generate more samples (6, any local model): the toast reports how many were added and how many near-duplicates were rejected.
- Delete a dataset with confirmation; bulk select and delete.

### 4. Measured run

- Paste `Classify the priority of this support ticket as high, medium or low.` or pick the dataset first and click **Insert a starter prompt**.
- Pick the dataset under Measure against a dataset. Note the held-out hint; with few samples it suggests k-fold.
- Meta-Prompt, Start. Progress reports each candidate being compiled and scored. About 20s for a hold-out run, a few minutes for k-fold.
- Eval Results: original vs selected score, the candidate scoreboard, the examples kept with "covers inputs like", and the per-sample table switchable between the selected prompt and the original. The score badge reads "measured".
- Switch the strategy to K-fold and run again: every sample is scored and the card says "cross-validated".

### 5. GEPA

- Method GEPA without a dataset: Start is disabled with a warning. Pick the dataset: budget slider and reflection-model picker appear.
- Start. Progress shows "Generation N: …" with the best score so far; about a minute at budget 60.
- Prompt Evolution: original vs evolved score, one row per candidate with generation, parent, the feedback the reflection read, and a word diff against the parent. If nothing beat the original, the original is kept and the card says so.

### 6. Feedback and analytics

- Thumbs up on the result: toast, Session Stats feedback count updates. Thumbs down opens a note box; save a note.
- Analytics tab: success rate, average improvement, success by day, model performance, task types and recent activity all reflect the sessions you just ran. View Detailed Analytics opens the charts.
- Sidebar Recent Sessions shows each run with its score.

### 7. Robustness

- Start a run and restart the API (`npm run dev:api`): on restart the interrupted session is marked failed instead of staying "running".
- Remove the embedding model (`ollama rm nomic-embed-text`) and run a measured run: examples fall back to the first validated ones and the card tooltip says why. Pull it again afterwards.

## Reporting a problem

Include the method, whether a dataset was used and its size, the model, the API log line (`API/logs/promptcraft.log`) and the session id from the sidebar.
