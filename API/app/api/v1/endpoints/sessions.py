import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.auth import verify_api_key
from app.core.config import settings
from app.core.database import get_db
from app.models.optimization import OptimizationSession, SessionStatus
from app.models.training import TrainingDataset, TrainingSample
from app.schemas.optimization import (
    OptimizationSessionCreate,
    OptimizationSessionResponse,
    OptimizationSessionUpdate,
    OptimizeRequest,
    PerformanceMetrics,
)
from app.services.eval_service import EvalError, Sample
from app.services.optimization_service import optimization_service

router = APIRouter(dependencies=[Depends(verify_api_key)])

OPTIMIZATION_METHODS = {"meta_prompt", "dspy", "simple"}


def _load_dataset_samples(db: Session, dataset_id: str) -> list[Sample]:
    """Samples of a dataset for evaluation; 404 if missing, 422 if too small."""
    if (
        not db.query(TrainingDataset.id)
        .filter(TrainingDataset.id == dataset_id)
        .first()
    ):
        raise HTTPException(status_code=404, detail="Dataset not found")

    rows = (
        db.query(TrainingSample.input_text, TrainingSample.expected_output)
        .filter(TrainingSample.dataset_id == dataset_id)
        .order_by(TrainingSample.created_at, TrainingSample.id)
        .limit(settings.eval_max_train_samples + settings.eval_max_dev_samples)
        .all()
    )
    samples = [Sample(input_text, expected) for input_text, expected in rows]
    if len(samples) < 2:
        raise HTTPException(
            status_code=422,
            detail="The dataset needs at least 2 samples to evaluate a prompt against",
        )
    return samples


def _get_session_or_404(db: Session, session_id: str) -> OptimizationSession:
    session = (
        db.query(OptimizationSession)
        .filter(OptimizationSession.id == session_id)
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.post("/", response_model=OptimizationSessionResponse)
def create_session(
    session_data: OptimizationSessionCreate,
    db: Session = Depends(get_db),
) -> OptimizationSession:
    """Create a new optimization session"""
    db_session = OptimizationSession(
        id=str(uuid.uuid4()),
        name=session_data.name,
        original_prompt=session_data.original_prompt,
        provider=session_data.provider,
        model=session_data.model,
        task_type=session_data.task_type,
    )

    db.add(db_session)
    db.commit()
    db.refresh(db_session)

    return db_session


@router.get("/optimization-methods")
async def get_optimization_methods() -> dict[str, list[dict[str, Any]]]:
    """Describe the available optimization methods.

    The text is written to match what the code does, so the UI can explain the
    choice honestly rather than with marketing copy.
    """
    return {
        "methods": [
            {
                "id": "meta_prompt",
                "name": "Meta-Prompt",
                "description": "One structured rewrite guided by a prompt-engineering rubric.",
                "how_it_works": (
                    "A single dspy.Predict call. The model is handed your prompt inside a "
                    "meta-prompt that asks for clarity, context, structure, examples and "
                    "constraints, and returns the rewritten prompt."
                ),
                "best_for": "Most prompts. The fastest structured option and a good default.",
                "returns_reasoning": False,
                "relative_speed": "fast",
                "recommended_for": ["general", "creative", "analysis"],
            },
            {
                "id": "dspy",
                "name": "DSPy Chain-of-Thought",
                "description": "The model reasons about the prompt first, then rewrites it.",
                "how_it_works": (
                    "dspy.ChainOfThought over a PromptRewrite signature: the model writes "
                    "out its reasoning about what the prompt needs, then produces the "
                    "rewrite. That reasoning is shown in Optimization Insights. If the "
                    "model cannot follow the structured output format, a template "
                    "rewrite is used and flagged as a fallback."
                ),
                "best_for": (
                    "Prompts that need judgement -- ambiguous asks, multi-step tasks, "
                    "or when you want to see why changes were made."
                ),
                "returns_reasoning": True,
                "relative_speed": "slower",
                "recommended_for": ["structured", "reasoning", "code"],
            },
            {
                "id": "simple",
                "name": "Simple",
                "description": "A plain completion asked to improve the prompt.",
                "how_it_works": (
                    "The model receives a short instruction ('improve this prompt') and "
                    "your text, with no DSPy structure or rubric. Whatever it returns is "
                    "the result."
                ),
                "best_for": "A quick baseline, or comparing against the structured methods.",
                "returns_reasoning": False,
                "relative_speed": "fastest",
                "recommended_for": ["quick", "basic", "testing"],
            },
        ]
    }


@router.get("/", response_model=list[OptimizationSessionResponse])
def get_sessions(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=settings.max_page_size),
    db: Session = Depends(get_db),
) -> list[OptimizationSession]:
    """Get all optimization sessions"""
    return (
        db.query(OptimizationSession)
        .order_by(OptimizationSession.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


@router.get("/analytics/performance", response_model=PerformanceMetrics)
def get_performance_metrics(db: Session = Depends(get_db)) -> PerformanceMetrics:
    """Get overall performance metrics"""
    total_optimizations = db.query(func.count(OptimizationSession.id)).scalar() or 0

    completed = (
        db.query(
            func.count(OptimizationSession.id),
            func.avg(OptimizationSession.performance_score),
        )
        .filter(OptimizationSession.status == SessionStatus.COMPLETED)
        .one()
    )
    completed_count, average_improvement = completed[0] or 0, completed[1] or 0.0

    # Sessions from before processing_time was stored contribute nothing.
    timed_count, total_processing_time = (
        db.query(
            func.count(OptimizationSession.processing_time),
            func.sum(OptimizationSession.processing_time),
        )
        .filter(OptimizationSession.processing_time.isnot(None))
        .one()
    )

    return PerformanceMetrics(
        total_optimizations=total_optimizations,
        average_improvement=float(average_improvement),
        success_rate=(
            (completed_count / total_optimizations * 100)
            if total_optimizations
            else 0.0
        ),
        total_processing_time=(float(total_processing_time) if timed_count else None),
        cost_savings=None,  # Not tracked: local Ollama runs have no billed cost.
    )


@router.get("/{session_id}", response_model=OptimizationSessionResponse)
def get_session(session_id: str, db: Session = Depends(get_db)) -> OptimizationSession:
    """Get a specific optimization session"""
    return _get_session_or_404(db, session_id)


@router.put("/{session_id}", response_model=OptimizationSessionResponse)
def update_session(
    session_id: str,
    session_update: OptimizationSessionUpdate,
    db: Session = Depends(get_db),
) -> OptimizationSession:
    """Update an optimization session"""
    session = _get_session_or_404(db, session_id)

    for field, value in session_update.model_dump(exclude_unset=True).items():
        setattr(session, field, value)

    db.commit()
    db.refresh(session)
    return session


@router.delete("/{session_id}")
def delete_session(session_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    """Delete an optimization session"""
    session = _get_session_or_404(db, session_id)

    db.delete(session)
    db.commit()
    return {"message": "Session deleted successfully"}


@router.post("/{session_id}/optimize")
async def optimize_prompt(
    session_id: str,
    options: OptimizeRequest | None = None,
    optimization_method: str | None = Query(
        None,
        description="Overrides options.optimization_method; kept for older clients.",
    ),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Optimize a prompt using Promptomatix algorithms"""
    options = options or OptimizeRequest()
    method = optimization_method or options.optimization_method
    if method not in OPTIMIZATION_METHODS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown optimization method. Expected one of: {sorted(OPTIMIZATION_METHODS)}",
        )

    session = _get_session_or_404(db, session_id)
    dataset_samples = (
        _load_dataset_samples(db, options.dataset_id) if options.dataset_id else None
    )

    # Update session status to running
    session.status = SessionStatus.RUNNING
    session.optimization_method = method
    session.dataset_id = options.dataset_id
    db.commit()

    try:
        optimization_result = await optimization_service.optimize_prompt(
            original_prompt=session.original_prompt,
            provider=session.provider,
            model=session.model,
            task_type=session.task_type,
            optimization_method=method,
            temperature=options.temperature,
            max_tokens=options.max_tokens,
            output_format=options.output_format,
            target_length=options.target_length,
            preserve_wording=options.preserve_wording,
            dataset_samples=dataset_samples,
            eval_metric=options.eval_metric,
            max_demos=options.max_demos,
        )
    except EvalError as e:
        session.status = SessionStatus.FAILED
        db.commit()
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        session.status = SessionStatus.FAILED
        db.commit()
        raise HTTPException(
            status_code=500, detail=f"Optimization error: {str(e)}"
        ) from e

    if not optimization_result["success"]:
        session.status = SessionStatus.FAILED
        db.commit()
        raise HTTPException(
            status_code=502,
            detail=f"Optimization failed: {optimization_result.get('error', 'Unknown error')}",
        )

    session.optimized_prompt = optimization_result["optimized_prompt"]
    session.performance_score = optimization_result["improvement_score"]
    session.processing_time = optimization_result["processing_time"]
    session.baseline_score = optimization_result.get("baseline_score")
    session.eval_score = optimization_result.get("eval_score")
    session.eval_metric = optimization_result.get("eval_metric")
    session.eval_sample_count = optimization_result.get("eval_sample_count")
    session.status = SessionStatus.COMPLETED

    db.commit()
    db.refresh(session)

    # No response_model on this route, so the return annotation is the
    # response schema; the ORM row must be converted explicitly.
    return {
        "message": "Prompt optimized successfully",
        "session": OptimizationSessionResponse.model_validate(session),
        "optimization_details": {
            "method": optimization_result["method"],
            "improvement_score": optimization_result["improvement_score"],
            "score_type": optimization_result.get("score_type", "heuristic"),
            "processing_time": optimization_result["processing_time"],
            "metadata": optimization_result.get("metadata", {}),
        },
    }
