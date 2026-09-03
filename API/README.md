# PromptCraft API

The FastAPI backend for [PromptCraft](../README.md): prompt optimization with DSPy against local Ollama models. This file covers backend development; the product, examples and installation are in the root README.

## Setup

The root `npm install` sets this package up with [uv](https://docs.astral.sh/uv/) and creates `.env`. To work on the API alone:

```sh
cd API
uv sync --all-extras          # creates .venv from pyproject.toml / uv.lock
cp .env.example .env          # if it does not exist
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Migrations run on startup. Docs are at <http://127.0.0.1:8000/docs> when `DEBUG=true`.

## Everyday commands

```sh
uv run pytest tests -q        # test suite; hermetic, no Ollama needed
uv run make lint              # ruff + strict mypy
uv run make format            # black, isort, ruff --fix
uv run alembic upgrade head   # apply migrations by hand
uv run alembic revision --autogenerate -m "describe change"
```

Or from the repository root: `npm run test:api`, `npm run lint:api`, `npm run migrate:api`.

## Layout

```
app/
├── api/v1/endpoints/   sessions (optimize, jobs, feedback, analytics), providers, training
├── core/               config (pydantic-settings), database, migrations, auth, logging
├── models/             SQLAlchemy 2.0 models
├── schemas/            Pydantic request/response models
└── services/
    ├── optimization_service.py   method dispatch, heuristic score, dataset hand-off
    ├── eval_service.py           splits (stratified, k-fold), metrics, BootstrapFewShot candidates
    ├── gepa_service.py           GEPA wrapper: feedback metrics, iteration tracking, lineage
    ├── embedding_service.py      Ollama embeddings, coverage selection, de-duplication
    ├── job_manager.py            in-process background jobs with progress
    ├── ollama_service.py         model listing with real metadata
    ├── training_service.py       synthetic data generation, import parsing
    └── lm_manager.py             dspy.LM construction
alembic/                migrations (0001 baseline, 0002 eval columns, 0003 feedback)
tests/                  pytest; conftest replaces embeddings with a deterministic stand-in
```

## Conventions

- Model calls run in a worker thread inside `dspy.context(lm=...)`; never on the event loop.
- Long work reports progress through the `ProgressCallback` protocol so background jobs can expose it.
- Anything that needs the embedding model catches `EmbeddingUnavailable` and falls back, stating why in the response.
- Schema changes go through an Alembic migration; `create_all` is only a safety net.
- `make lint` and the test suite are expected to be clean on every change.

## Configuration

See `.env.example`. Notable settings: `DEFAULT_MODEL_NAME`, `EMBEDDING_MODEL`, the `EVAL_*` caps that bound model calls per run, `SYNTHETIC_DUPLICATE_THRESHOLD`, optional `API_KEY` / `REQUIRE_API_KEY`, `ALLOWED_HOSTS`, `MAX_PAGE_SIZE`.

The API runs as a single worker process; background jobs live in memory and interrupted sessions are marked failed on the next start.
