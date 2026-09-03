# Changelog

All notable changes to the PromptCraft API will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Ollama models are listed with the configured default first, then smallest to
  largest. Clients take the first entry as the default; Ollama's own order (by
  download time) put a 35B model ahead of llama3.2 and made the default run
  ten times slower.
- Sessions left in `running` by an interrupted process are marked `failed` at
  startup instead of staying "running" forever.
- **Dataset sample counts were inflated**: bulk sample creation and synthetic
  generation added `len(new_samples)` on top of a `COUNT(*)` that already
  included the flushed rows, so `sample_count` and `size` roughly doubled.
  Counts are now read back from the database after each mutation.
- **Blocking model calls on the event loop**: prompt optimization and synthetic
  data generation invoked the synchronous DSPy/Ollama client directly inside
  `async def` handlers. With `OLLAMA_TIMEOUT=120` a single request could stall
  every other request on the worker. Both now run in a threadpool.
- **Failed optimizations were reported as successes**: when every strategy fell
  back, the API returned HTTP 200 with the original prompt and a 50.0
  "improvement" score. A run that produces nothing now returns 502 and marks the
  session failed; an unchanged prompt scores 0.
- **`GET /{anything}` returned 405 instead of 404**: the catch-all
  `@app.options("/{full_path:path}")` route matched every path, shadowing 404s.
  Removed -- `CORSMiddleware` already answers preflight.
- **CSV import and export corrupted data**: hand-rolled `split(",")` parsing and
  manual quote escaping broke on any field containing a comma, quote or newline.
  Both now use the `csv` module.
- **`TrustedHostMiddleware` was hardcoded to localhost**, rejecting every request
  in any other deployment (and in the test suite). Now driven by `ALLOWED_HOSTS`.
- **Ollama client used a hardcoded 30s timeout**, ignoring `OLLAMA_TIMEOUT`, and
  its connection pool was never closed. It now honours the setting, sends
  `keep_alive`, and closes on shutdown.
- **`LMManager.get_available_models()` never awaited** the coroutine it called,
  so it always fell through to the default model. Now `async`.
- **`GET /providers` 502'd entirely when Ollama was down.** The catalogue is
  still returned, with Ollama marked unavailable.
- **`GET /training/{id}` could orphan samples**: assigning to `dataset.samples`
  on a `delete-orphan` relationship marked existing rows for deletion. Sample
  loading is now decided in the query.
- **Model heuristics had unreachable branches**: `codellama` matched the generic
  `llama` case before its own, reporting 8192 instead of a 16384 context window.
- Synthetic data and dataset import failures no longer return silently empty
  results -- they raise with an error code explaining what went wrong.
- Exceptions raised while handling another exception now chain with `from`.

### Added
- `POST /sessions/{id}/feedback`: thumbs up/down with an optional note on the
  optimized prompt (migration `0003` adds `feedback_rating`, `feedback_comment`,
  `feedback_at`); a null rating clears it, and 409 if the session has no
  optimized prompt yet. `thumbs_up` / `thumbs_down` counts on the analytics
  endpoint.
- Context length is read from `/api/show` for models that omit it from
  `/api/tags` (gemma3n reports 32K there, not the 4K fallback). Cached per
  model for the process lifetime.
- **Coverage-based example selection.** BootstrapFewShot now validates a pool
  of up to `EVAL_DEMO_POOL` (12) examples; the `max_demos` that best span the
  training inputs are kept, chosen by farthest-point sampling over
  `EMBEDDING_MODEL` (nomic-embed-text) vectors, one per class first for label
  datasets. Each kept example carries `covers`, the training inputs it stands
  closest to, and the report carries `demo_selection`. Falls back to the first
  validated examples, with the reason, when the embedding model is missing.
- **Near-duplicate rejection in synthetic generation.** Generated samples whose
  embedding similarity to an existing sample (or an earlier new one) reaches
  `SYNTHETIC_DUPLICATE_THRESHOLD` (0.92) are dropped; the response lists them
  under `duplicates` with what they matched, or `dedup_skipped_reason` when
  embeddings were unavailable.
- **Stratified splits and k-fold evaluation.** Label datasets (a small set of
  short expected outputs) are split so the held-out samples cover the classes
  instead of, say, two samples with the same label. `eval_strategy: "kfold"`
  holds every sample out once across up to `EVAL_MAX_FOLDS` (5) folds, scores
  each candidate type on all of them, and refits the winner on the whole
  dataset for the returned prompt; scores then move in steps of 1/N rather
  than 1/dev_size. The eval report carries a `split` block (strategy, folds,
  stratified, class counts). GEPA uses the stratified hold-out split.
- **GEPA optimization method** (`optimization_method: "gepa"`, needs
  `dataset_id`). Wraps `dspy.teleprompt.GEPA`: the prompt runs on training
  samples, the metric writes feedback for each miss (for example "the label is
  buried in a 40-word answer"), a reflection model rewrites the instructions to
  address it, and candidates that win on different samples are kept on a Pareto
  front. `gepa_budget` (10-500 scored calls, default 60) and `reflection_model`
  (defaults to the task model) are new request fields. The response's
  `metadata.gepa` carries the lineage: every candidate's instructions, score,
  parent, generation and the feedback that preceded it, plus `metadata.eval`
  in the same shape as measured runs. Each GEPA iteration is reported as
  progress on the background job. `GET /optimization-methods` marks it with
  `requires_dataset`.
- **Background optimization**: `POST /sessions/{id}/optimize/start` returns 202
  and a job snapshot; `GET /sessions/{id}/optimize/status` reports the stage,
  a step counter, the best score so far, the full step history and, when done,
  the same payload the synchronous route returns. One job per session at a
  time (409 otherwise). The synchronous `POST /optimize` is unchanged; both
  share one code path. Jobs are in-process, so the API must run as a single
  worker, which is how `make dev` and the Makefile targets start it.
- Per-exception HTTP status codes (503 for a down Ollama, 404, 422, 502, ...)
  instead of flattening every `PromptCraftException` to 400.
- `available` / `unavailable_reason` on provider responses: OpenAI and Anthropic
  are listed for the catalogue but cannot be driven by this build, and were
  previously indistinguishable from working providers.
- Pagination guards (`MAX_PAGE_SIZE`, `ge`/`le` bounds) on every list endpoint.
- Request validation on the write schemas: non-empty text, quality scores in
  `[0, 1]`, bulk size limits, and `Literal` import/export formats.
- Database connectivity check in `/health` (it previously only claimed to).
- SQLite `PRAGMA foreign_keys=ON`, which SQLite otherwise ignores.
- 54 tests covering the fixes above (24 -> 78 total): dataset counting, CSV
  round-trips, session optimize flows, provider degradation, and API key auth.
- `GET /training/stats`: dataset and sample totals, a per-task-type breakdown
  and the most recently modified datasets, counted from the sample rows.
- `avg_quality_score` on dataset summaries (`GET /training/`), computed in one
  aggregate query; datasets are now listed newest first.
- **Measured optimization**: `POST /sessions/{id}/optimize` accepts
  `dataset_id`, `eval_metric` (`auto` / `exact` / `contains` / `llm_judge`) and
  `max_demos`. The dataset is split into train and held-out samples; the
  original prompt, the rewrite and few-shot versions of each (examples chosen by
  DSPy `BootstrapFewShot` on the train split) are scored on the held-out
  samples, and the best candidate is returned. `performance_score` is then the
  measured pass rate, `score_type` says which kind of score it is, and
  `metadata.eval` carries the scoreboard, demos and per-sample results.
- Sessions now store `optimization_method`, `processing_time` and, for measured
  runs, `dataset_id`, `baseline_score`, `eval_score`, `eval_metric` and
  `eval_sample_count`. `total_processing_time` in the analytics endpoint is
  summed from the database instead of an in-process history.
- **Alembic migrations**. `alembic upgrade head` (or just starting the API)
  brings any database current: `0001` records the pre-migration schema and
  only creates tables that are missing, `0002` adds the session columns above.
  Databases created by `create_all` before this release migrate in place.
- `EVAL_MAX_TRAIN_SAMPLES`, `EVAL_MAX_DEV_SAMPLES` and `EVAL_MAX_DEMOS` settings.

### Changed
- The API is managed with `uv` (`API/uv.lock` committed). A root `npm install`
  now runs `uv sync` (or a venv + pip fallback), installs the web app and
  creates `API/.env`; `npm run dev:api`, `test:api`, `lint:api` and
  `migrate:api` run inside that environment via `scripts/run-api.mjs`.
- Models use SQLAlchemy 2.0 `Mapped[...]` columns and the whole `app` package
  passes strict `mypy`; `make lint` is green.
- OpenAI and Anthropic are no longer listed by `GET /providers/`. They were
  placeholder catalogue entries with `available=False`; nothing could drive them.
- `DSPy` optimization now actually calls DSPy (a `ChainOfThought` over a
  `PromptRewrite` signature) instead of returning a fixed template string; the
  template remains as a fallback and says so in the response metadata.
- Synthetic data generation defaults to the configured local provider. It
  defaulted to `openai` / `gpt-3.5-turbo`, which `LMManager` always rejected.
- Anthropic entries in the provider catalogue refreshed to current models.
- Optimization history is bounded (`OPTIMIZATION_HISTORY_SIZE`); it was an
  unbounded list on a process-lifetime singleton.
- `PerformanceMetrics.total_processing_time` is derived from real runs and
  `cost_savings` is null rather than the previous hardcoded `120.0` / `50.0`.
- `requirements.txt` realigned with `pyproject.toml`. The old `fastapi>=0.104.0`
  floor resolved to a Starlette incompatible with `httpx>=0.28`, which broke
  every `TestClient` test at collection.
- Ruff config moved under `[tool.ruff.lint]`; `ruff check` is clean (was 449
  errors). `mypy` still reports 75 pre-existing strict-mode errors (down from
  86); annotating the SQLAlchemy models with `Mapped[...]` is the remaining
  work to make `make lint` green.
- `black` and `isort` applied across `app/` and `tests/`, so `make format` and
  the pre-commit hooks are now a no-op instead of a 24-file diff.

## [1.0.0] - 2024-10-24

### Added
- **Professional Codebase Refactoring**: Complete transformation to production-ready standards
- **Custom Exception System**: Comprehensive error handling with detailed error codes and context
- **Structured Logging**: Professional logging configuration with rotation, levels, and JSON formatting
- **Comprehensive Test Suite**: Full pytest test coverage with fixtures, mocks, and async testing
- **Modern Python Features**: Updated to Python 3.11+ with type hints throughout
- **Development Tooling**: Pre-commit hooks, linting (Ruff), formatting (Black), and type checking (MyPy)
- **Professional Documentation**: Complete API documentation with examples and best practices
- **Security Enhancements**: Trusted host middleware, input validation, and secure error handling
- **Performance Monitoring**: Health checks, performance logging, and async architecture
- **Dependency Management**: Modern pyproject.toml with optional dependencies and proper versioning

### Changed
- **FastAPI Application**: Modernized with lifespan events, proper middleware, and exception handlers
- **Service Architecture**: Improved error handling and logging throughout all services
- **Database Models**: Enhanced with proper type hints and documentation
- **Configuration Management**: Centralized settings with environment variable validation
- **Project Structure**: Organized following Python packaging best practices

### Improved
- **Code Quality**: 100% type hint coverage and comprehensive docstrings
- **Error Handling**: Custom exceptions with detailed context and proper HTTP status codes
- **Logging**: Replaced all print statements with structured logging
- **Testing**: Added unit tests, integration tests, and service-specific test suites
- **Documentation**: Professional README, API documentation, and inline code documentation
- **Developer Experience**: Makefile for common tasks, pre-commit hooks, and automated formatting

### Technical Debt Resolved
- ✅ Replaced print statements with proper logging
- ✅ Added comprehensive type hints throughout codebase
- ✅ Implemented custom exception classes
- ✅ Created professional test suite with high coverage
- ✅ Modernized dependency management
- ✅ Enhanced security practices
- ✅ Improved code organization and structure
- ✅ Added comprehensive documentation

### Development Tools Added
- **Pre-commit Configuration**: Automated code quality checks
- **Makefile**: Common development tasks automation
- **Ruff**: Fast Python linting and code analysis
- **Black**: Consistent code formatting
- **MyPy**: Static type checking
- **pytest**: Comprehensive testing framework
- **Coverage**: Test coverage reporting

### Security Enhancements
- **Input Validation**: Pydantic schemas with comprehensive validation
- **CORS Configuration**: Secure cross-origin resource sharing
- **Trusted Hosts**: Host validation middleware
- **Error Sanitization**: Secure error response handling
- **Environment Variables**: Secure configuration management

## [0.1.0] - 2024-10-01

### Added
- Initial FastAPI application setup
- Basic prompt optimization functionality
- Ollama integration for local AI models
- SQLite database with SQLAlchemy
- Basic API endpoints for sessions and providers
- DSPy integration for prompt optimization
- Multi-provider AI support (OpenAI, Anthropic, Ollama)

### Features
- Prompt optimization using meta-prompting and DSPy
- Session management and persistence
- Training data generation and management
- Real-time analytics and performance tracking
- Multi-provider AI model support

---

## Version History Summary

- **v1.0.0**: Professional production-ready refactoring with comprehensive improvements
- **v0.1.0**: Initial development version with basic functionality

## Upcoming Features

### Planned for v1.1.0
- [ ] Advanced caching system for improved performance
- [ ] WebSocket support for real-time optimization updates
- [ ] Enhanced analytics with detailed metrics
- [ ] Database migration system with Alembic
- [ ] Docker containerization
- [ ] CI/CD pipeline configuration

### Planned for v1.2.0
- [ ] Advanced prompt optimization algorithms
- [ ] Multi-language support
- [ ] Enhanced security features
- [ ] Performance optimizations
- [ ] Extended AI provider support
