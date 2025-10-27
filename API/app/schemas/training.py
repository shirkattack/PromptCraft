from pydantic import BaseModel, field_validator
from datetime import datetime
from typing import Optional, List, Dict, Any
import json

# Training Sample Schemas
class TrainingSampleBase(BaseModel):
    input_text: str
    expected_output: str
    extra_data: Optional[Dict[str, Any]] = None
    quality_score: Optional[float] = 0.0

class TrainingSampleCreate(TrainingSampleBase):
    pass

class TrainingSampleUpdate(BaseModel):
    input_text: Optional[str] = None
    expected_output: Optional[str] = None
    extra_data: Optional[Dict[str, Any]] = None
    quality_score: Optional[float] = None

class TrainingSampleResponse(TrainingSampleBase):
    id: str
    dataset_id: str
    created_at: datetime

    @field_validator('extra_data', mode='before')
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
    samples: List[TrainingSampleCreate]

class TrainingSampleBulkResponse(BaseModel):
    created_count: int
    failed_count: int
    created_samples: List[TrainingSampleResponse]
    errors: List[str]

# Training Dataset Schemas (Enhanced)
class TrainingDatasetBase(BaseModel):
    name: str
    description: Optional[str] = None
    task_type: str

class TrainingDatasetCreate(TrainingDatasetBase):
    samples: Optional[List[TrainingSampleCreate]] = []

class TrainingDatasetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    task_type: Optional[str] = None

class TrainingDatasetResponse(TrainingDatasetBase):
    id: str
    sample_count: int
    created_at: datetime
    last_modified: datetime
    size: Optional[str] = None
    samples: Optional[List[TrainingSampleResponse]] = None

    class Config:
        from_attributes = True

class TrainingDatasetSummary(BaseModel):
    id: str
    name: str
    description: Optional[str]
    task_type: str
    sample_count: int
    created_at: datetime
    last_modified: datetime
    size: Optional[str]

    class Config:
        from_attributes = True

# Synthetic Data Generation
class SyntheticDataRequest(BaseModel):
    dataset_id: str
    sample_count: int = 10
    base_prompt: str
    task_type: str
    provider: str = "openai"
    model: str = "gpt-3.5-turbo"
    creativity_level: float = 0.7

class SyntheticDataResponse(BaseModel):
    dataset_id: str
    generated_count: int
    failed_count: int
    samples: List[TrainingSampleResponse]
    processing_time: float

# Import/Export
class DatasetImportRequest(BaseModel):
    name: str
    description: Optional[str] = None
    task_type: str
    file_format: str = "json"  # json, csv, jsonl
    data: str  # Base64 encoded file content or raw JSON

class DatasetExportRequest(BaseModel):
    dataset_id: str
    format: str = "json"  # json, csv, jsonl
    include_metadata: bool = True

class DatasetExportResponse(BaseModel):
    dataset_name: str
    format: str
    data: str  # Exported data
    sample_count: int
    export_timestamp: datetime

# Search and Filtering
class DatasetSearchRequest(BaseModel):
    query: Optional[str] = None
    task_types: Optional[List[str]] = None
    min_samples: Optional[int] = None
    max_samples: Optional[int] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None

class SampleSearchRequest(BaseModel):
    dataset_id: str
    query: Optional[str] = None
    min_quality_score: Optional[float] = None
    max_quality_score: Optional[float] = None