import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.core.database import Base


class SessionStatus(enum.StrEnum):
    COMPLETED = "completed"
    RUNNING = "running"
    FAILED = "failed"


class OptimizationSession(Base):
    __tablename__ = "optimization_sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    original_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    optimized_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    provider: Mapped[str] = mapped_column(String, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)
    task_type: Mapped[str] = mapped_column(String, nullable=False)
    performance_score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    status: Mapped[SessionStatus] = mapped_column(
        Enum(SessionStatus), default=SessionStatus.RUNNING
    )

    # Added by migration 0002. All nullable: sessions created before it, and
    # runs without a dataset, simply leave them empty.
    optimization_method: Mapped[str | None] = mapped_column(String, nullable=True)
    processing_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    dataset_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("training_datasets.id", ondelete="SET NULL"),
        nullable=True,
    )
    # Measured on held-out samples of `dataset_id`: percentage of samples the
    # metric accepted for the original prompt (baseline) and the optimized one.
    baseline_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    eval_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    eval_metric: Mapped[str | None] = mapped_column(String, nullable=True)
    eval_sample_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Added by migration 0003: a thumbs up/down and note from the user on the
    # optimized prompt, a second evaluation signal alongside the measured score.
    feedback_rating: Mapped[str | None] = mapped_column(String, nullable=True)
    feedback_comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    feedback_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Added by migration 0004: the optimization_details payload (method,
    # scores, eval scoreboard, GEPA timeline) as JSON, so a past session can
    # be reopened with everything the run produced.
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
