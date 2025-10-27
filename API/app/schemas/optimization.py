from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
from enum import Enum

class SessionStatus(str, Enum):
    COMPLETED = "completed"
    RUNNING = "running"
    FAILED = "failed"

class AIModelResponse(BaseModel):
    id: str
    name: str
    context_window: int
    cost_per_1k_tokens: float
    speed_rating: int
    best_use_case: str
    is_free: Optional[bool] = False

class AIProviderResponse(BaseModel):
    id: str
    name: str
    logo: str
    models: List[AIModelResponse]

class OptimizationSessionBase(BaseModel):
    name: str
    original_prompt: str
    provider: str
    model: str
    task_type: str

class OptimizationSessionCreate(OptimizationSessionBase):
    pass

class OptimizationSessionUpdate(BaseModel):
    optimized_prompt: Optional[str] = None
    performance_score: Optional[float] = None
    status: Optional[SessionStatus] = None

class OptimizationSessionResponse(OptimizationSessionBase):
    id: str
    optimized_prompt: Optional[str] = None
    performance_score: float
    created_at: datetime
    status: SessionStatus

    class Config:
        from_attributes = True

class TrainingDatasetBase(BaseModel):
    name: str
    description: Optional[str] = None
    task_type: str

class TrainingDatasetCreate(TrainingDatasetBase):
    sample_count: int = 0
    size: Optional[str] = None

class TrainingDatasetResponse(TrainingDatasetBase):
    id: str
    sample_count: int
    created_at: datetime
    last_modified: datetime
    size: Optional[str] = None

    class Config:
        from_attributes = True

class PerformanceMetrics(BaseModel):
    total_optimizations: int
    average_improvement: float
    success_rate: float
    total_processing_time: float
    cost_savings: float

class ProviderPerformance(BaseModel):
    provider: str
    speed: float
    quality: float
    cost: float
    reliability: float