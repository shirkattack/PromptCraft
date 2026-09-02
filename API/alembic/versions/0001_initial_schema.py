"""Initial schema: the tables as they were before migrations existed.

Databases created before this revision were built by ``Base.metadata.create_all``
and have no ``alembic_version`` table, so each table is only created when it is
missing. That lets this revision run against both a fresh database and one that
predates Alembic.

Revision ID: 0001
Revises:
Create Date: 2026-09-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _existing_tables() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def upgrade() -> None:
    existing = _existing_tables()

    if "optimization_sessions" not in existing:
        op.create_table(
            "optimization_sessions",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("original_prompt", sa.Text(), nullable=False),
            sa.Column("optimized_prompt", sa.Text(), nullable=True),
            sa.Column("provider", sa.String(), nullable=False),
            sa.Column("model", sa.String(), nullable=False),
            sa.Column("task_type", sa.String(), nullable=False),
            sa.Column("performance_score", sa.Float(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("(CURRENT_TIMESTAMP)"),
                nullable=True,
            ),
            sa.Column(
                "status",
                sa.Enum("COMPLETED", "RUNNING", "FAILED", name="sessionstatus"),
                nullable=True,
            ),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "ix_optimization_sessions_id", "optimization_sessions", ["id"]
        )

    if "training_datasets" not in existing:
        op.create_table(
            "training_datasets",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("sample_count", sa.Integer(), nullable=True),
            sa.Column("task_type", sa.String(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("(CURRENT_TIMESTAMP)"),
                nullable=True,
            ),
            sa.Column(
                "last_modified",
                sa.DateTime(timezone=True),
                server_default=sa.text("(CURRENT_TIMESTAMP)"),
                nullable=True,
            ),
            sa.Column("size", sa.String(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_training_datasets_id", "training_datasets", ["id"])

    if "training_samples" not in existing:
        op.create_table(
            "training_samples",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("dataset_id", sa.String(), nullable=False),
            sa.Column("input_text", sa.Text(), nullable=False),
            sa.Column("expected_output", sa.Text(), nullable=False),
            sa.Column("extra_data", sa.Text(), nullable=True),
            sa.Column("quality_score", sa.Float(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("(CURRENT_TIMESTAMP)"),
                nullable=True,
            ),
            sa.ForeignKeyConstraint(["dataset_id"], ["training_datasets.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_training_samples_id", "training_samples", ["id"])


def downgrade() -> None:
    # Dropping the tables would destroy user data; the base revision is not
    # reversible on purpose.
    raise RuntimeError("Downgrading below the initial schema is not supported")
