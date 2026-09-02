import enum

from sqlalchemy import Column, DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.sql import func

from app.core.database import Base


class SessionStatus(enum.StrEnum):
    COMPLETED = "completed"
    RUNNING = "running"
    FAILED = "failed"


class OptimizationSession(Base):
    __tablename__ = "optimization_sessions"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    original_prompt = Column(Text, nullable=False)
    optimized_prompt = Column(Text, nullable=True)
    provider = Column(String, nullable=False)
    model = Column(String, nullable=False)
    task_type = Column(String, nullable=False)
    performance_score = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(Enum(SessionStatus), default=SessionStatus.RUNNING)

    # Added by migration 0002. All nullable: sessions created before it, and
    # runs without a dataset, simply leave them empty.
    optimization_method = Column(String, nullable=True)
    processing_time = Column(Float, nullable=True)
    dataset_id = Column(
        String,
        ForeignKey("training_datasets.id", ondelete="SET NULL"),
        nullable=True,
    )
    # Measured on held-out samples of `dataset_id`: percentage of samples the
    # metric accepted for the original prompt (baseline) and the optimized one.
    baseline_score = Column(Float, nullable=True)
    eval_score = Column(Float, nullable=True)
    eval_metric = Column(String, nullable=True)
    eval_sample_count = Column(Integer, nullable=True)
