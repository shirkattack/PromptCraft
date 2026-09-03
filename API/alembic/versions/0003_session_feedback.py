"""Store user feedback (thumbs up/down, comment) on sessions.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE = "optimization_sessions"

NEW_COLUMNS = [
    sa.Column("feedback_rating", sa.String(), nullable=True),
    sa.Column("feedback_comment", sa.Text(), nullable=True),
    sa.Column("feedback_at", sa.DateTime(timezone=True), nullable=True),
]


def upgrade() -> None:
    present = {col["name"] for col in sa.inspect(op.get_bind()).get_columns(TABLE)}
    missing = [column for column in NEW_COLUMNS if column.name not in present]
    if not missing:
        return
    with op.batch_alter_table(TABLE) as batch:
        for column in missing:
            batch.add_column(column)


def downgrade() -> None:
    with op.batch_alter_table(TABLE) as batch:
        for column in reversed(NEW_COLUMNS):
            batch.drop_column(column.name)
