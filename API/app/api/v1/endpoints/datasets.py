"""Legacy dataset endpoints.

Superseded by `app.api.v1.endpoints.training`, which owns sample management,
import/export and synthetic generation. Kept for compatibility with existing
clients; new work belongs in the training router.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.auth import verify_api_key
from app.core.config import settings
from app.core.database import get_db
from app.models.training import TrainingDataset
from app.schemas.training import TrainingDatasetCreate, TrainingDatasetResponse

router = APIRouter(dependencies=[Depends(verify_api_key)], deprecated=True)


@router.post("/", response_model=TrainingDatasetResponse)
def create_dataset(
    dataset_data: TrainingDatasetCreate, db: Session = Depends(get_db)
) -> TrainingDataset:
    """Create a new training dataset"""
    dataset_id = str(uuid.uuid4())

    db_dataset = TrainingDataset(
        id=dataset_id,
        name=dataset_data.name,
        description=dataset_data.description,
        task_type=dataset_data.task_type,
    )

    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)

    return db_dataset


@router.get("/", response_model=list[TrainingDatasetResponse])
def get_datasets(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=settings.max_page_size),
    db: Session = Depends(get_db),
) -> list[TrainingDataset]:
    """Get all training datasets"""
    datasets = db.query(TrainingDataset).offset(skip).limit(limit).all()
    return datasets


@router.get("/{dataset_id}", response_model=TrainingDatasetResponse)
def get_dataset(dataset_id: str, db: Session = Depends(get_db)) -> TrainingDataset:
    """Get a specific training dataset"""
    dataset = db.query(TrainingDataset).filter(TrainingDataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    """Delete a training dataset"""
    dataset = db.query(TrainingDataset).filter(TrainingDataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    db.delete(dataset)
    db.commit()
    return {"message": "Dataset deleted successfully"}


@router.get("/task-types/")
def get_task_types() -> dict[str, list[str]]:
    """Get available task types"""
    return {
        "task_types": [
            "Text Generation",
            "Question Answering",
            "Summarization",
            "Translation",
            "Code Generation",
            "Creative Writing",
            "Analysis",
            "Extraction",
        ]
    }
