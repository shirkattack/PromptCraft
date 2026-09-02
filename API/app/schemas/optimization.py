from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class SessionStatus(StrEnum):
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
    is_free: bool | None = False
    # Reported by the runtime (Ollama /api/tags) rather than guessed from the name.
    parameter_size: str | None = None  # e.g. "3.2B"
    quantization: str | None = None  # e.g. "Q4_K_M"
    family: str | None = None  # e.g. "llama"
    size_bytes: int | None = None  # on-disk size
    capabilities: list[str] = []  # e.g. ["completion", "tools", "vision"]


class AIProviderResponse(BaseModel):
    id: str
    name: str
    logo: str
    models: list[AIModelResponse]
    # The backend can only drive providers it has an LM adapter for. Listing a
    # provider without this flag let clients pick a model that was guaranteed
    # to fail at optimization time.
    available: bool = True
    unavailable_reason: str | None = None


class OptimizationSessionBase(BaseModel):
    name: str
    original_prompt: str
    provider: str
    model: str
    task_type: str


class OptimizationSessionCreate(OptimizationSessionBase):
    pass


class OptimizationSessionUpdate(BaseModel):
    optimized_prompt: str | None = None
    performance_score: float | None = None
    status: SessionStatus | None = None


OutputFormat = Literal["auto", "markdown", "plain", "json"]
TargetLength = Literal["auto", "concise", "balanced", "detailed"]
EvalMetric = Literal["auto", "exact", "contains", "llm_judge"]


class OptimizeRequest(BaseModel):
    """Options for POST /sessions/{id}/optimize.

    temperature / max_tokens go straight to the model. The rest become explicit
    constraints in the rewrite instructions.
    """

    optimization_method: Literal["meta_prompt", "dspy", "simple"] = "meta_prompt"
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=64, le=8192)
    output_format: OutputFormat = "auto"
    target_length: TargetLength = "auto"
    preserve_wording: bool = False
    # Measure against a training dataset. When set, few-shot candidates are
    # compiled with DSPy and the returned prompt is the one that scored best on
    # held-out samples; performance_score becomes that measured score.
    dataset_id: str | None = None
    eval_metric: EvalMetric = "auto"
    max_demos: int = Field(default=4, ge=1, le=8)


class OptimizationSessionResponse(OptimizationSessionBase):
    id: str
    optimized_prompt: str | None = None
    performance_score: float
    created_at: datetime
    status: SessionStatus
    optimization_method: str | None = None
    processing_time: float | None = None
    # Present only for runs measured against a dataset.
    dataset_id: str | None = None
    baseline_score: float | None = None
    eval_score: float | None = None
    eval_metric: str | None = None
    eval_sample_count: int | None = None

    class Config:
        from_attributes = True


class TrainingDatasetBase(BaseModel):
    name: str
    description: str | None = None
    task_type: str


class TrainingDatasetCreate(TrainingDatasetBase):
    sample_count: int = 0
    size: str | None = None


class TrainingDatasetResponse(TrainingDatasetBase):
    id: str
    sample_count: int
    created_at: datetime
    last_modified: datetime
    size: str | None = None

    class Config:
        from_attributes = True


class PerformanceMetrics(BaseModel):
    total_optimizations: int
    average_improvement: float
    success_rate: float
    # Wall-clock time is only tracked for optimizations run by the current
    # process; it is not persisted per session yet. `cost_savings` has no
    # source of truth at all, so it stays null rather than being invented.
    total_processing_time: float | None = None
    cost_savings: float | None = None


class ProviderPerformance(BaseModel):
    provider: str
    speed: float
    quality: float
    cost: float
    reliability: float
