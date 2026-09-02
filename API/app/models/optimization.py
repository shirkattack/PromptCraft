import enum

from sqlalchemy import Column, DateTime, Enum, Float, String, Text
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
