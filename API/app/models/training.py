from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base


class TrainingDataset(Base):
    __tablename__ = "training_datasets"

    id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    task_type: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_modified: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    size: Mapped[str | None] = mapped_column(String, nullable=True)

    samples: Mapped[list["TrainingSample"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )


class TrainingSample(Base):
    __tablename__ = "training_samples"

    id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    dataset_id: Mapped[str] = mapped_column(
        String, ForeignKey("training_datasets.id"), nullable=False
    )
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    expected_output: Mapped[str] = mapped_column(Text, nullable=False)
    extra_data: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    quality_score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    dataset: Mapped["TrainingDataset"] = relationship(back_populates="samples")
