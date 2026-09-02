import csv
import io
import json
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, noload, selectinload

from app.core.auth import verify_api_key
from app.core.config import settings
from app.core.database import get_db
from app.core.logging import get_logger
from app.models.training import TrainingDataset, TrainingSample
from app.schemas.training import (
    DatasetExportRequest,
    DatasetExportResponse,
    DatasetImportRequest,
    SyntheticDataRequest,
    SyntheticDataResponse,
    TrainingDatasetCreate,
    TrainingDatasetResponse,
    TrainingDatasetSummary,
    TrainingDatasetUpdate,
    TrainingSampleBulkCreate,
    TrainingSampleBulkResponse,
    TrainingSampleCreate,
    TrainingSampleResponse,
    TrainingSampleUpdate,
)
from app.services.training_service import training_service

router = APIRouter(dependencies=[Depends(verify_api_key)])
logger = get_logger("training_endpoints")


def _get_dataset_or_404(db: Session, dataset_id: str) -> TrainingDataset:
    dataset = db.query(TrainingDataset).filter(TrainingDataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


def _refresh_dataset_stats(db: Session, dataset: TrainingDataset) -> None:
    """Recount samples from the database and stamp the dataset as modified.

    Deriving the count arithmetically (``count() + len(new)``) double-counted
    rows that had already been flushed into the transaction, so the count is
    always read back instead.
    """
    db.flush()
    dataset.sample_count = (
        db.query(TrainingSample).filter(TrainingSample.dataset_id == dataset.id).count()
    )
    dataset.size = f"{dataset.sample_count} samples"
    dataset.last_modified = datetime.now(UTC)


def _new_sample(dataset_id: str, sample_data: TrainingSampleCreate) -> TrainingSample:
    return TrainingSample(
        id=str(uuid.uuid4()),
        dataset_id=dataset_id,
        input_text=sample_data.input_text,
        expected_output=sample_data.expected_output,
        extra_data=(
            json.dumps(sample_data.extra_data) if sample_data.extra_data else None
        ),
        quality_score=sample_data.quality_score or 0.0,
    )


# Dataset CRUD Operations
@router.post("/", response_model=TrainingDatasetResponse)
def create_dataset(
    dataset_data: TrainingDatasetCreate,
    db: Session = Depends(get_db),
):
    """Create a new training dataset with optional initial samples"""
    dataset_id = str(uuid.uuid4())

    db_dataset = TrainingDataset(
        id=dataset_id,
        name=dataset_data.name,
        description=dataset_data.description,
        task_type=dataset_data.task_type,
    )

    db.add(db_dataset)
    db.flush()  # Get the ID

    for sample_data in dataset_data.samples or []:
        db.add(_new_sample(dataset_id, sample_data))

    _refresh_dataset_stats(db, db_dataset)
    db.commit()
    db.refresh(db_dataset)

    return db_dataset


@router.get("/", response_model=list[TrainingDatasetSummary])
def get_datasets(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=settings.max_page_size),
    task_type: str | None = None,
    db: Session = Depends(get_db),
):
    """Get all training datasets with optional filtering"""
    query = db.query(TrainingDataset)

    if task_type:
        query = query.filter(TrainingDataset.task_type == task_type)

    return query.offset(skip).limit(limit).all()


@router.get("/{dataset_id}", response_model=TrainingDatasetResponse)
def get_dataset(
    dataset_id: str, include_samples: bool = False, db: Session = Depends(get_db)
):
    """Get a specific training dataset"""
    # Load (or explicitly skip) samples in the query. Assigning to
    # `dataset.samples` afterwards would mark the existing rows as orphans on a
    # relationship configured with delete-orphan.
    loader = (
        selectinload(TrainingDataset.samples)
        if include_samples
        else noload(TrainingDataset.samples)
    )
    dataset = (
        db.query(TrainingDataset)
        .options(loader)
        .filter(TrainingDataset.id == dataset_id)
        .first()
    )
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    response = TrainingDatasetResponse.model_validate(dataset)
    if not include_samples:
        # null rather than [] so clients can tell "not loaded" from "none".
        response.samples = None
    return response


@router.put("/{dataset_id}", response_model=TrainingDatasetResponse)
def update_dataset(
    dataset_id: str,
    dataset_update: TrainingDatasetUpdate,
    db: Session = Depends(get_db),
):
    """Update a training dataset"""
    dataset = _get_dataset_or_404(db, dataset_id)

    for field, value in dataset_update.model_dump(exclude_unset=True).items():
        setattr(dataset, field, value)

    dataset.last_modified = datetime.now(UTC)
    db.commit()
    db.refresh(dataset)

    return dataset


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Delete a training dataset and all its samples"""
    dataset = _get_dataset_or_404(db, dataset_id)

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
    db: Session = Depends(get_db),
):
    """Add a single sample to a dataset"""
    dataset = _get_dataset_or_404(db, dataset_id)

    # Validate quality if not provided
    if sample_data.quality_score is None:
        sample_data.quality_score = training_service.validate_sample_quality(
            sample_data
        )

    db_sample = _new_sample(dataset_id, sample_data)
    db.add(db_sample)

    _refresh_dataset_stats(db, dataset)
    db.commit()
    db.refresh(db_sample)

    return db_sample


@router.post("/{dataset_id}/samples/bulk", response_model=TrainingSampleBulkResponse)
def create_samples_bulk(
    dataset_id: str,
    bulk_data: TrainingSampleBulkCreate,
    db: Session = Depends(get_db),
):
    """Add multiple samples to a dataset"""
    dataset = _get_dataset_or_404(db, dataset_id)

    created_samples = []
    errors = []

    for i, sample_data in enumerate(bulk_data.samples):
        try:
            if sample_data.quality_score is None:
                sample_data.quality_score = training_service.validate_sample_quality(
                    sample_data
                )

            db_sample = _new_sample(dataset_id, sample_data)
            db.add(db_sample)
            db.flush()  # To get the created sample
            created_samples.append(db_sample)

        except Exception as e:
            errors.append(f"Sample {i}: {str(e)}")

    _refresh_dataset_stats(db, dataset)
    db.commit()

    return TrainingSampleBulkResponse(
        created_count=len(created_samples),
        failed_count=len(errors),
        created_samples=created_samples,
        errors=errors,
    )


@router.get("/{dataset_id}/samples", response_model=list[TrainingSampleResponse])
def get_samples(
    dataset_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=settings.max_page_size),
    min_quality: float | None = Query(None, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
):
    """Get samples from a dataset"""
    _get_dataset_or_404(db, dataset_id)

    query = db.query(TrainingSample).filter(TrainingSample.dataset_id == dataset_id)

    if min_quality is not None:
        query = query.filter(TrainingSample.quality_score >= min_quality)

    return query.offset(skip).limit(limit).all()


@router.put("/{dataset_id}/samples/{sample_id}", response_model=TrainingSampleResponse)
def update_sample(
    dataset_id: str,
    sample_id: str,
    sample_update: TrainingSampleUpdate,
    db: Session = Depends(get_db),
):
    """Update a training sample"""
    sample = (
        db.query(TrainingSample)
        .filter(TrainingSample.id == sample_id, TrainingSample.dataset_id == dataset_id)
        .first()
    )

    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    for field, value in sample_update.model_dump(exclude_unset=True).items():
        if field == "extra_data" and value is not None:
            setattr(sample, field, json.dumps(value))
        else:
            setattr(sample, field, value)

    # Update dataset modified time
    dataset = db.query(TrainingDataset).filter(TrainingDataset.id == dataset_id).first()
    if dataset:
        dataset.last_modified = datetime.now(UTC)

    db.commit()
    db.refresh(sample)

    return sample


@router.delete("/{dataset_id}/samples/{sample_id}")
def delete_sample(dataset_id: str, sample_id: str, db: Session = Depends(get_db)):
    """Delete a training sample"""
    sample = (
        db.query(TrainingSample)
        .filter(TrainingSample.id == sample_id, TrainingSample.dataset_id == dataset_id)
        .first()
    )

    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    db.delete(sample)

    dataset = db.query(TrainingDataset).filter(TrainingDataset.id == dataset_id).first()
    if dataset:
        _refresh_dataset_stats(db, dataset)

    db.commit()

    return {"message": "Sample deleted successfully"}


# Synthetic Data Generation
@router.post("/{dataset_id}/generate", response_model=SyntheticDataResponse)
async def generate_synthetic_data(
    dataset_id: str,
    request: SyntheticDataRequest,
    db: Session = Depends(get_db),
):
    """Generate synthetic training data for a dataset"""
    dataset = _get_dataset_or_404(db, dataset_id)

    start_time = datetime.now(UTC)

    # Override dataset_id in request
    request.dataset_id = dataset_id

    # Generate synthetic samples (raises SyntheticDataGenerationError on failure)
    synthetic_samples = await training_service.generate_synthetic_data(request)

    created_samples = []
    failed_count = 0

    for sample_data in synthetic_samples:
        # A savepoint per sample so one bad row does not poison the whole
        # transaction (a plain rollback would discard the good samples too).
        try:
            with db.begin_nested():
                db_sample = _new_sample(dataset_id, sample_data)
                db_sample.extra_data = json.dumps(
                    sample_data.extra_data or {"synthetic": True}
                )
                db.add(db_sample)
            created_samples.append(db_sample)

        except Exception as e:
            logger.warning(f"Skipped a generated sample for dataset {dataset_id}: {e}")
            failed_count += 1

    _refresh_dataset_stats(db, dataset)
    db.commit()

    return SyntheticDataResponse(
        dataset_id=dataset_id,
        generated_count=len(created_samples),
        failed_count=failed_count,
        samples=created_samples,
        processing_time=(datetime.now(UTC) - start_time).total_seconds(),
    )


# Import/Export Operations
@router.post("/import", response_model=TrainingDatasetResponse)
def import_dataset(
    request: DatasetImportRequest,
    db: Session = Depends(get_db),
):
    """Import a dataset from external data"""
    # Parse the import data (raises TrainingDataError on malformed input)
    samples = training_service.parse_import_data(request)

    dataset_id = str(uuid.uuid4())
    db_dataset = TrainingDataset(
        id=dataset_id,
        name=request.name,
        description=request.description,
        task_type=request.task_type,
    )

    db.add(db_dataset)
    db.flush()

    for sample_data in samples:
        db.add(_new_sample(dataset_id, sample_data))

    _refresh_dataset_stats(db, db_dataset)
    db.commit()
    db.refresh(db_dataset)

    return db_dataset


@router.post("/{dataset_id}/export", response_model=DatasetExportResponse)
def export_dataset(
    dataset_id: str,
    request: DatasetExportRequest,
    db: Session = Depends(get_db),
):
    """Export a dataset in various formats"""
    dataset = _get_dataset_or_404(db, dataset_id)

    samples = (
        db.query(TrainingSample).filter(TrainingSample.dataset_id == dataset_id).all()
    )

    if request.format == "json":
        export_data = []
        for sample in samples:
            item = {"input": sample.input_text, "output": sample.expected_output}
            if request.include_metadata and sample.extra_data:
                try:
                    item["extra_data"] = json.loads(sample.extra_data)
                except json.JSONDecodeError:
                    item["extra_data"] = {"raw": sample.extra_data}
            export_data.append(item)

        data_str = json.dumps(export_data, indent=2)

    elif request.format == "csv":
        # csv.writer handles quoting and embedded newlines, which hand-rolled
        # escaping got wrong for any prompt containing a comma or a quote.
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["input", "output"])
        for sample in samples:
            writer.writerow([sample.input_text, sample.expected_output])

        data_str = buffer.getvalue()

    else:
        raise HTTPException(status_code=400, detail="Unsupported export format")

    return DatasetExportResponse(
        dataset_name=dataset.name,
        format=request.format,
        data=data_str,
        sample_count=len(samples),
        export_timestamp=datetime.now(UTC),
    )
