"""Apply Alembic migrations programmatically (used at API startup)."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.util.exc import CommandError

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("migrations")

API_ROOT = Path(__file__).resolve().parents[2]


def alembic_config(database_url: str | None = None) -> Config:
    config = Config(str(API_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(API_ROOT / "alembic"))
    url = database_url or settings.database_url
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return config


def run_migrations(database_url: str | None = None) -> None:
    """Upgrade the database to the latest revision.

    Safe on a fresh database, on one created by ``create_all`` before Alembic
    was introduced, and on one that is already current. A database stamped
    with a revision this code does not know (it was last run from a newer
    branch) is left alone with a warning instead of failing startup; every
    migration guards its own changes, so re-running later is harmless.
    """
    try:
        command.upgrade(alembic_config(database_url), "head")
    except CommandError as exc:
        if "Can't locate revision" not in str(exc):
            raise
        logger.warning(
            "Database is at a migration revision unknown to this code (%s). "
            "It was probably last run from a newer branch. Continuing without "
            "migrating; switch back to that branch, or reset the marker with: "
            "sqlite3 API/app.db \"update alembic_version set version_num='<known>'\"",
            exc,
        )
        return
    logger.info("Database migrations applied")
