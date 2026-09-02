"""Apply Alembic migrations programmatically (used at API startup)."""

from pathlib import Path

from alembic import command
from alembic.config import Config

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
    was introduced, and on one that is already current.
    """
    command.upgrade(alembic_config(database_url), "head")
    logger.info("Database migrations applied")
