"""The startup migration must work on old, fresh and already-current databases."""

import sqlite3

import sqlalchemy as sa

from app.core.migrations import run_migrations

NEW_COLUMNS = {
    "optimization_method",
    "processing_time",
    "dataset_id",
    "baseline_score",
    "eval_score",
    "eval_metric",
    "eval_sample_count",
}

PRE_ALEMBIC_SCHEMA = """
CREATE TABLE optimization_sessions (
    id VARCHAR NOT NULL, name VARCHAR NOT NULL, original_prompt TEXT NOT NULL,
    optimized_prompt TEXT, provider VARCHAR NOT NULL, model VARCHAR NOT NULL,
    task_type VARCHAR NOT NULL, performance_score FLOAT, created_at DATETIME,
    status VARCHAR(9), PRIMARY KEY (id)
);
CREATE TABLE training_datasets (
    id VARCHAR NOT NULL, name VARCHAR NOT NULL, description TEXT, sample_count INTEGER,
    task_type VARCHAR NOT NULL, created_at DATETIME, last_modified DATETIME, size VARCHAR,
    PRIMARY KEY (id)
);
CREATE TABLE training_samples (
    id VARCHAR NOT NULL, dataset_id VARCHAR NOT NULL, input_text TEXT NOT NULL,
    expected_output TEXT NOT NULL, extra_data TEXT, quality_score FLOAT, created_at DATETIME,
    PRIMARY KEY (id), FOREIGN KEY(dataset_id) REFERENCES training_datasets (id)
);
INSERT INTO optimization_sessions (id, name, original_prompt, provider, model, task_type, performance_score, status)
VALUES ('s1', 'old', 'p', 'ollama', 'llama3.2:latest', 'general', 72.0, 'COMPLETED');
"""


def _columns(url: str, table: str) -> set[str]:
    engine = sa.create_engine(url)
    try:
        return {col["name"] for col in sa.inspect(engine).get_columns(table)}
    finally:
        engine.dispose()


def _version(url: str) -> str:
    engine = sa.create_engine(url)
    try:
        with engine.connect() as conn:
            return conn.execute(
                sa.text("SELECT version_num FROM alembic_version")
            ).scalar()
    finally:
        engine.dispose()


def test_upgrades_a_database_created_before_alembic(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(PRE_ALEMBIC_SCHEMA)
    url = f"sqlite:///{path}"

    run_migrations(url)

    assert NEW_COLUMNS <= _columns(url, "optimization_sessions")
    assert _version(url) == "0002"
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT performance_score, eval_score FROM optimization_sessions WHERE id='s1'"
        ).fetchone()
    assert row == (72.0, None)  # existing rows survive the table rebuild


def test_builds_a_fresh_database(tmp_path):
    url = f"sqlite:///{tmp_path / 'fresh.db'}"

    run_migrations(url)

    engine = sa.create_engine(url)
    tables = set(sa.inspect(engine).get_table_names())
    engine.dispose()
    assert {"optimization_sessions", "training_datasets", "training_samples"} <= tables
    assert NEW_COLUMNS <= _columns(url, "optimization_sessions")


def test_is_idempotent(tmp_path):
    url = f"sqlite:///{tmp_path / 'twice.db'}"
    run_migrations(url)
    run_migrations(url)
    assert _version(url) == "0002"
