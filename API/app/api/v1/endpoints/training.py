from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
import json
from datetime import datetime

from app.core.database import get_db
from app.models.training import TrainingDataset, TrainingSample
from app.schemas.training import (
    TrainingDatasetCreate,
    TrainingDatasetResponse,
    TrainingDatasetUpdate,
    TrainingDatasetSummary,
    TrainingSampleCreate,
    TrainingSampleResponse,
    TrainingSampleUpdate,
    TrainingSampleBulkCreate,
    TrainingSampleBulkResponse,
    SyntheticDataRequest,
    SyntheticDataResponse,
    DatasetImportRequest,
    DatasetExportRequest,
    DatasetExportResponse,
    DatasetSearchRequest,
    SampleSearchRequest
)
from app.services.training_service import training_service

router = APIRouter()

# Dataset CRUD Operations
@router.post("/", response_model=TrainingDatasetResponse)
async def create_dataset(
    dataset_data: TrainingDatasetCreate,
    db: Session = Depends(get_db)
):
    """Create a new training dataset with optional initial samples"""
    dataset_id = str(uuid.uuid4())
    
    # Create dataset
    db_dataset = TrainingDataset(
        id=dataset_id,
        name=dataset_data.name,
        description=dataset_data.description,
        task_type=dataset_data.task_type,
        sample_count=len(dataset_data.samples) if dataset_data.samples else 0,
        size=f"{len(dataset_data.samples)} samples" if dataset_data.samples else "0 samples"
    )
    
    db.add(db_dataset)
    db.flush()  # Get the ID
    
    # Add initial samples if provided
    if dataset_data.samples:
        for sample_data in dataset_data.samples:
            sample_id = str(uuid.uuid4())
            db_sample = TrainingSample(
                id=sample_id,
                dataset_id=dataset_id,
                input_text=sample_data.input_text,
                expected_output=sample_data.expected_output,
                extra_data=json.dumps(sample_data.extra_data) if sample_data.extra_data else None,
                quality_score=sample_data.quality_score or 0.0
            )
            db.add(db_sample)
    
    db.commit()
    db.refresh(db_dataset)
    
    return db_dataset

@router.get("/", response_model=List[TrainingDatasetSummary])
def get_datasets(
    skip: int = 0, 
    limit: int = 100,
    task_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all training datasets with optional filtering"""
    query = db.query(TrainingDataset)
    
    if task_type:
        query = query.filter(TrainingDataset.task_type == task_type)
    
    datasets = query.offset(skip).limit(limit).all()
    return datasets

@router.get("/{dataset_id}", response_model=TrainingDatasetResponse)
def get_dataset(dataset_id: str, include_samples: bool = False, db: Session = Depends(get_db)):
    """Get a specific training dataset"""
    dataset = db.query(TrainingDataset).filter(TrainingDataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if include_samples:
        # Load samples
        dataset.samples = db.query(TrainingSample).filter(
            TrainingSample.dataset_id == dataset_id
        ).all()
    
    return dataset

@router.put("/{dataset_id}", response_model=TrainingDatasetResponse)
def update_dataset(
    dataset_id: str,
    dataset_update: TrainingDatasetUpdate,
    db: Session = Depends(get_db)
):
    """Update a training dataset"""
    dataset = db.query(TrainingDataset).filter(TrainingDataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    update_data = dataset_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(dataset, field, value)
    
    dataset.last_modified = datetime.utcnow()
    db.commit()
    db.refresh(dataset)
    
    return dataset

@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Delete a training dataset and all its samples"""
    dataset = db.query(TrainingDataset).filter(TrainingDataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Delete all samples first (cascade should handle this, but being explicit)
    db.query(TrainingSample).filter(TrainingSample.dataset_id == dataset_id).delete()
    
    db.delete(dataset)
    db.commit()
    
    return {"message": "Dataset deleted successfully"}

# Sample CRUD Operations
@router.post("/{dataset_id}/samples", response_model=TrainingSampleResponse)
def create_sample(
    dataset_id: str,
    sample_data: TrainingSampleCreate,
    db: Session = Depends(get_db)
):
    """Add a single sample to a dataset"""
    dataset = db.query(TrainingDataset).filter(TrainingDataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    sample_id = str(uuid.uuid4())
    
    # Validate quality if not provided
    if sample_data.quality_score is None:
        sample_data.quality_score = training_service.validate_sample_quality(sample_data)
    
    db_sample = TrainingSample(
        id=sample_id,
        dataset_id=dataset_id,
        input_text=sample_data.input_text,
        expected_output=sample_data.expected_output,
        extra_data=json.dumps(sample_data.extra_data) if sample_data.extra_data else None,
        quality_score=sample_data.quality_score
    )
    
    db.add(db_sample)
    
    # Update dataset sample count
    dataset.sample_count = db.query(TrainingSample).filter(
        TrainingSample.dataset_id == dataset_id
    ).count() + 1
    dataset.last_modified = datetime.utcnow()
    dataset.size = f"{dataset.sample_count} samples"
    
    db.commit()
    db.refresh(db_sample)
    
    return db_sample

@router.post("/{dataset_id}/samples/bulk", response_model=TrainingSampleBulkResponse)
def create_samples_bulk(
    dataset_id: str,
    bulk_data: TrainingSampleBulkCreate,
    db: Session = Depends(get_db)
):
    """Add multiple samples to a dataset"""
    dataset = db.query(TrainingDataset).filter(TrainingDataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    created_samples = []
    errors = []
    
    for i, sample_data in enumerate(bulk_data.samples):
        try:
            sample_id = str(uuid.uuid4())
            
            # Validate quality if not provided
            if sample_data.quality_score is None:
                sample_data.quality_score = training_service.validate_sample_quality(sample_data)
            
            db_sample = TrainingSample(
                id=sample_id,
                dataset_id=dataset_id,
                input_text=sample_data.input_text,
                expected_output=sample_data.expected_output,
                extra_data=json.dumps(sample_data.extra_data) if sample_data.extra_data else None,
                quality_score=sample_data.quality_score
            )
            
            db.add(db_sample)
            db.flush()  # To get the created sample
            created_samples.append(db_sample)
            
        except Exception as e:
            errors.append(f"Sample {i}: {str(e)}")
    
    # Update dataset stats
    dataset.sample_count = db.query(TrainingSample).filter(
        TrainingSample.dataset_id == dataset_id
    ).count() + len(created_samples)
    dataset.last_modified = datetime.utcnow()
    dataset.size = f"{dataset.sample_count} samples"
    
    db.commit()
    
    return TrainingSampleBulkResponse(
        created_count=len(created_samples),
        failed_count=len(errors),
        created_samples=created_samples,
        errors=errors
    )

@router.get("/{dataset_id}/samples", response_model=List[TrainingSampleResponse])
def get_samples(
    dataset_id: str,
    skip: int = 0,
    limit: int = 100,
    min_quality: Optional[float] = None,
    db: Session = Depends(get_db)
):
    """Get samples from a dataset"""
    query = db.query(TrainingSample).filter(TrainingSample.dataset_id == dataset_id)
    
    if min_quality is not None:
        query = query.filter(TrainingSample.quality_score >= min_quality)
    
    samples = query.offset(skip).limit(limit).all()
    return samples

@router.put("/{dataset_id}/samples/{sample_id}", response_model=TrainingSampleResponse)
def update_sample(
    dataset_id: str,
    sample_id: str,
    sample_update: TrainingSampleUpdate,
    db: Session = Depends(get_db)
):
    """Update a training sample"""
    sample = db.query(TrainingSample).filter(
        TrainingSample.id == sample_id,
        TrainingSample.dataset_id == dataset_id
    ).first()
    
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    
    update_data = sample_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        if field == "extra_data" and value is not None:
            setattr(sample, field, json.dumps(value))
        else:
            setattr(sample, field, value)
    
    # Update dataset modified time
    dataset = db.query(TrainingDataset).filter(TrainingDataset.id == dataset_id).first()
    if dataset:
        dataset.last_modified = datetime.utcnow()
    
    db.commit()
    db.refresh(sample)
    
    return sample

@router.delete("/{dataset_id}/samples/{sample_id}")
def delete_sample(dataset_id: str, sample_id: str, db: Session = Depends(get_db)):
    """Delete a training sample"""
    sample = db.query(TrainingSample).filter(
        TrainingSample.id == sample_id,
        TrainingSample.dataset_id == dataset_id
    ).first()
    
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    
    db.delete(sample)
    
    # Update dataset sample count
    dataset = db.query(TrainingDataset).filter(TrainingDataset.id == dataset_id).first()
    if dataset:
        dataset.sample_count = db.query(TrainingSample).filter(
            TrainingSample.dataset_id == dataset_id
        ).count() - 1
        dataset.last_modified = datetime.utcnow()
        dataset.size = f"{dataset.sample_count} samples"
    
    db.commit()
    
    return {"message": "Sample deleted successfully"}

# Synthetic Data Generation
@router.post("/{dataset_id}/generate", response_model=SyntheticDataResponse)
async def generate_synthetic_data(
    dataset_id: str,
    request: SyntheticDataRequest,
    db: Session = Depends(get_db)
):
    """Generate synthetic training data for a dataset"""
    dataset = db.query(TrainingDataset).filter(TrainingDataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    start_time = datetime.now()
    
    # Override dataset_id in request
    request.dataset_id = dataset_id
    
    # Generate synthetic samples
    synthetic_samples = await training_service.generate_synthetic_data(request)
    
    created_samples = []
    failed_count = 0
    
    # Add generated samples to database
    for sample_data in synthetic_samples:
        try:
            sample_id = str(uuid.uuid4())
            db_sample = TrainingSample(
                id=sample_id,
                dataset_id=dataset_id,
                input_text=sample_data.input_text,
                expected_output=sample_data.expected_output,
                extra_data=json.dumps(sample_data.extra_data) if sample_data.extra_data else json.dumps({"synthetic": True}),
                quality_score=sample_data.quality_score
            )
            
            db.add(db_sample)
            db.flush()
            created_samples.append(db_sample)
            
        except Exception as e:
            failed_count += 1
    
    # Update dataset stats
    dataset.sample_count = db.query(TrainingSample).filter(
        TrainingSample.dataset_id == dataset_id
    ).count() + len(created_samples)
    dataset.last_modified = datetime.utcnow()
    dataset.size = f"{dataset.sample_count} samples"
    
    db.commit()
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    return SyntheticDataResponse(
        dataset_id=dataset_id,
        generated_count=len(created_samples),
        failed_count=failed_count,
        samples=created_samples,
        processing_time=processing_time
    )

# Import/Export Operations
@router.post("/import", response_model=TrainingDatasetResponse)
async def import_dataset(
    request: DatasetImportRequest,
    db: Session = Depends(get_db)
):
    """Import a dataset from external data"""
    # Parse the import data
    samples = training_service.parse_import_data(request)
    
    # Create dataset
    dataset_id = str(uuid.uuid4())
    db_dataset = TrainingDataset(
        id=dataset_id,
        name=request.name,
        description=request.description,
        task_type=request.task_type,
        sample_count=len(samples),
        size=f"{len(samples)} samples"
    )
    
    db.add(db_dataset)
    db.flush()
    
    # Add samples
    for sample_data in samples:
        sample_id = str(uuid.uuid4())
        db_sample = TrainingSample(
            id=sample_id,
            dataset_id=dataset_id,
            input_text=sample_data.input_text,
            expected_output=sample_data.expected_output,
            extra_data=json.dumps(sample_data.extra_data) if sample_data.extra_data else None,
            quality_score=sample_data.quality_score
        )
        db.add(db_sample)
    
    db.commit()
    db.refresh(db_dataset)
    
    return db_dataset

@router.post("/{dataset_id}/export", response_model=DatasetExportResponse)
def export_dataset(
    dataset_id: str,
    request: DatasetExportRequest,
    db: Session = Depends(get_db)
):
    """Export a dataset in various formats"""
    dataset = db.query(TrainingDataset).filter(TrainingDataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    samples = db.query(TrainingSample).filter(TrainingSample.dataset_id == dataset_id).all()
    
    # Export data based on format
    if request.format == "json":
        export_data = []
        for sample in samples:
            item = {
                "input": sample.input_text,
                "output": sample.expected_output
            }
            if request.include_metadata and sample.extra_data:
                try:
                    item["extra_data"] = json.loads(sample.extra_data)
                except:
                    item["extra_data"] = {"raw": sample.extra_data}
            export_data.append(item)
        
        data_str = json.dumps(export_data, indent=2)
    
    elif request.format == "csv":
        lines = ["input,output"]
        for sample in samples:
            # Basic CSV escaping
            input_escaped = f'"{sample.input_text.replace(chr(34), chr(34) + chr(34))}"'
            output_escaped = f'"{sample.expected_output.replace(chr(34), chr(34) + chr(34))}"'
            lines.append(f"{input_escaped},{output_escaped}")
        
        data_str = "\n".join(lines)
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported export format")
    
    return DatasetExportResponse(
        dataset_name=dataset.name,
        format=request.format,
        data=data_str,
        sample_count=len(samples),
        export_timestamp=datetime.now()
    )