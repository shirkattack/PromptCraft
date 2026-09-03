"""Store the full optimization result on the session.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE = "optimization_sessions"


def upgrade() -> None:
    present = {col["name"] for col in sa.inspect(op.get_bind()).get_columns(TABLE)}
    if "result_json" in present:
        return
    with op.batch_alter_table(TABLE) as batch:
        batch.add_column(sa.Column("result_json", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table(TABLE) as batch:
        batch.drop_column("result_json")
