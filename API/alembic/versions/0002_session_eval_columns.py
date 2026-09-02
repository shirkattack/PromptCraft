"""Store the method, duration and dataset-measured scores on sessions.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE = "optimization_sessions"

NEW_COLUMNS = [
    sa.Column("optimization_method", sa.String(), nullable=True),
    sa.Column("processing_time", sa.Float(), nullable=True),
    sa.Column("dataset_id", sa.String(), nullable=True),
    sa.Column("baseline_score", sa.Float(), nullable=True),
    sa.Column("eval_score", sa.Float(), nullable=True),
    sa.Column("eval_metric", sa.String(), nullable=True),
    sa.Column("eval_sample_count", sa.Integer(), nullable=True),
]


def upgrade() -> None:
    present = {col["name"] for col in sa.inspect(op.get_bind()).get_columns(TABLE)}
    missing = [column for column in NEW_COLUMNS if column.name not in present]
    if not missing:
        return

    with op.batch_alter_table(TABLE) as batch:
        for column in missing:
            batch.add_column(column)
        if "dataset_id" not in present:
            batch.create_foreign_key(
                "fk_optimization_sessions_dataset_id",
                "training_datasets",
                ["dataset_id"],
                ["id"],
                ondelete="SET NULL",
            )


def downgrade() -> None:
    with op.batch_alter_table(TABLE) as batch:
        batch.drop_constraint("fk_optimization_sessions_dataset_id", type_="foreignkey")
        for column in reversed(NEW_COLUMNS):
            batch.drop_column(column.name)
