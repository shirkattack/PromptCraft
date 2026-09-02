import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.core.config import settings


# Training Sample Schemas
class TrainingSampleBase(BaseModel):
    input_text: str
    expected_output: str
    extra_data: dict[str, Any] | None = None
    quality_score: float | None = 0.0


class TrainingSampleCreate(TrainingSampleBase):
    # Validation lives on the write models only; TrainingSampleResponse also
    # inherits from Base and has to serialise whatever is already stored.
    input_text: str = Field(min_length=1)
    expected_output: str = Field(min_length=1)
    quality_score: float | None = Field(default=0.0, ge=0.0, le=1.0)


class TrainingSampleUpdate(BaseModel):
    input_text: str | None = Field(default=None, min_length=1)
    expected_output: str | None = Field(default=None, min_length=1)
    extra_data: dict[str, Any] | None = None
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)


class TrainingSampleResponse(TrainingSampleBase):
    id: str
    dataset_id: str
    created_at: datetime

    @field_validator("extra_data", mode="before")
    @classmethod
    def parse_extra_data(cls, v):
        if isinstance(v, str) and v:
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return v or {}

    class Config:
        from_attributes = True


# Bulk Operations
class TrainingSampleBulkCreate(BaseModel):
    samples: list[TrainingSampleCreate] = Field(min_length=1, max_length=1000)


class TrainingSampleBulkResponse(BaseModel):
    created_count: int
    failed_count: int
    created_samples: list[TrainingSampleResponse]
    errors: list[str]


# Training Dataset Schemas (Enhanced)
class TrainingDatasetBase(BaseModel):
    name: str
    description: str | None = None
    task_type: str


class TrainingDatasetCreate(TrainingDatasetBase):
    samples: list[TrainingSampleCreate] | None = []


class TrainingDatasetUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    task_type: str | None = None


class TrainingDatasetResponse(TrainingDatasetBase):
    id: str
    sample_count: int
    created_at: datetime
    last_modified: datetime
    size: str | None = None
    samples: list[TrainingSampleResponse] | None = None

    class Config:
        from_attributes = True


class TrainingDatasetSummary(BaseModel):
    id: str
    name: str
    description: str | None
    task_type: str
    sample_count: int
    created_at: datetime
    last_modified: datetime
    size: str | None

    class Config:
        from_attributes = True


# Synthetic Data Generation
class SyntheticDataRequest(BaseModel):
    dataset_id: str
    sample_count: int = Field(
        default=settings.default_synthetic_data_size, ge=1, le=200
    )
    base_prompt: str = Field(min_length=1)
    task_type: str
    # Defaulted to the configured local provider: the previous openai /
    # gpt-3.5-turbo defaults were rejected by LMManager on every request.
    provider: str = settings.default_model_provider
    model: str = settings.default_model_name
    creativity_level: float = Field(
        default=settings.default_temperature, ge=0.0, le=2.0
    )


class SyntheticDataResponse(BaseModel):
    dataset_id: str
    generated_count: int
    failed_count: int
    samples: list[TrainingSampleResponse]
    processing_time: float


# Import/Export
class DatasetImportRequest(BaseModel):
    name: str = Field(min_length=1)
    description: str | None = None
    task_type: str
    file_format: Literal["json", "csv"] = "json"
    data: str = Field(min_length=1)  # Raw JSON or CSV text


class DatasetExportRequest(BaseModel):
    dataset_id: str
    format: Literal["json", "csv"] = "json"
    include_metadata: bool = True


class DatasetExportResponse(BaseModel):
    dataset_name: str
    format: str
    data: str  # Exported data
    sample_count: int
    export_timestamp: datetime


# Search and Filtering
class DatasetSearchRequest(BaseModel):
    query: str | None = None
    task_types: list[str] | None = None
    min_samples: int | None = None
    max_samples: int | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None


class SampleSearchRequest(BaseModel):
    dataset_id: str
    query: str | None = None
    min_quality_score: float | None = None
    max_quality_score: float | None = None
