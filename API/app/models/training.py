from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base

class TrainingDataset(Base):
    __tablename__ = "training_datasets"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    sample_count = Column(Integer, default=0)
    task_type = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_modified = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    size = Column(String, nullable=True)
    
    # Relationship to training samples
    samples = relationship("TrainingSample", back_populates="dataset", cascade="all, delete-orphan")

class TrainingSample(Base):
    __tablename__ = "training_samples"
    
    id = Column(String, primary_key=True, index=True)
    dataset_id = Column(String, ForeignKey("training_datasets.id"), nullable=False)
    input_text = Column(Text, nullable=False)
    expected_output = Column(Text, nullable=False)
    extra_data = Column(Text, nullable=True)  # JSON string for additional data
    quality_score = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship back to dataset
    dataset = relationship("TrainingDataset", back_populates="samples")