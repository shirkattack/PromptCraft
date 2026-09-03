import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import func
from sqlalchemy.orm import Session, sessionmaker

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
    SessionFeedback,
    TryRequest,
    TryResponse,
    TryResult,
)
from app.services.eval_service import EvalError, Sample
from app.services.job_manager import JobAlreadyRunning, OptimizationJob, job_manager
from app.services.optimization_service import optimization_service
from app.services.progress import ProgressCallback, no_progress
from app.services.try_service import try_prompts

router = APIRouter(dependencies=[Depends(verify_api_key)])

OPTIMIZATION_METHODS = {"meta_prompt", "dspy", "simple", "gepa"}


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


def fail_orphaned_sessions(db: Session) -> int:
    """Mark sessions still RUNNING as FAILED.

    Optimizations run inside the API process, so after a restart nothing can
    still be running; without this a killed run stays "running" forever.
    Called at startup.
    """
    count = (
        db.query(OptimizationSession)
        .filter(OptimizationSession.status == SessionStatus.RUNNING)
        .update({OptimizationSession.status: SessionStatus.FAILED})
    )
    db.commit()
    return count


def _prior_user_feedback(
    db: Session, original_prompt: str, exclude_id: str, limit: int = 5
) -> list[str]:
    """Thumbs-down notes left on earlier runs of the same prompt, newest first."""
    rows = (
        db.query(OptimizationSession.feedback_comment)
        .filter(
            OptimizationSession.original_prompt == original_prompt,
            OptimizationSession.id != exclude_id,
            OptimizationSession.feedback_rating == "down",
            OptimizationSession.feedback_comment.isnot(None),
            OptimizationSession.feedback_comment != "",
        )
        .order_by(OptimizationSession.feedback_at.desc())
        .limit(limit)
        .all()
    )
    notes: list[str] = []
    for (comment,) in rows:
        if comment and comment.strip() not in notes:
            notes.append(comment.strip())
    return notes


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
                "id": "gepa",
                "name": "GEPA (evolve from feedback)",
                "description": "Evolves the instructions from written feedback on what the prompt gets wrong. Needs a dataset.",
                "how_it_works": (
                    "dspy.teleprompt.GEPA. The prompt runs on a few training samples; "
                    "the metric writes feedback for each miss (for example 'the label "
                    "is buried in a paragraph'); a reflection model rewrites the "
                    "instructions to address that feedback; candidates that win on "
                    "different samples stay on a Pareto front and can be merged. "
                    "Every candidate is scored on the held-out samples, and the run "
                    "returns the whole lineage."
                ),
                "best_for": (
                    "Prompts with a dataset of expected outputs, when you want a "
                    "measured gain and a visible reason for every edit."
                ),
                "returns_reasoning": True,
                "relative_speed": "slowest",
                "recommended_for": ["classification", "extraction", "qa"],
                "requires_dataset": True,
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

    thumbs_up = (
        db.query(func.count(OptimizationSession.id))
        .filter(OptimizationSession.feedback_rating == "up")
        .scalar()
        or 0
    )
    thumbs_down = (
        db.query(func.count(OptimizationSession.id))
        .filter(OptimizationSession.feedback_rating == "down")
        .scalar()
        or 0
    )

    return PerformanceMetrics(
        total_optimizations=total_optimizations,
        thumbs_up=thumbs_up,
        thumbs_down=thumbs_down,
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


@router.get("/{session_id}/result")
def get_session_result(
    session_id: str, db: Session = Depends(get_db)
) -> dict[str, Any]:
    """A past session with the full result of its last optimization.

    ``optimization_details`` is null for sessions that never completed or
    that predate result storage; the prompts and scores are still returned.
    """
    session = _get_session_or_404(db, session_id)
    details = None
    if session.result_json:
        try:
            details = json.loads(session.result_json)
        except json.JSONDecodeError:
            details = None
    return {
        "session": OptimizationSessionResponse.model_validate(session).model_dump(
            mode="json"
        ),
        "optimization_details": details,
    }


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


@router.post("/{session_id}/try", response_model=TryResponse)
async def try_session_prompts(
    session_id: str, request: TryRequest, db: Session = Depends(get_db)
) -> TryResponse:
    """Run the session's original and/or optimized prompt on one input.

    Stateless: nothing is stored. Uses the session's provider and model.
    """
    session = _get_session_or_404(db, session_id)
    prompts: dict[str, str] = {}
    for variant in request.variants:
        if variant == "original":
            prompts["original"] = session.original_prompt
        elif session.optimized_prompt:
            prompts["optimized"] = session.optimized_prompt
    if not prompts:
        raise HTTPException(
            status_code=409, detail="This session has no optimized prompt to try yet"
        )
    results = await try_prompts(
        provider=session.provider,
        model=session.model,
        input_text=request.input,
        prompts=prompts,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    return TryResponse(
        session_id=session.id,
        model=session.model,
        input=request.input,
        results=[TryResult(**r) for r in results],
    )


@router.post("/{session_id}/feedback", response_model=OptimizationSessionResponse)
def submit_feedback(
    session_id: str,
    feedback: SessionFeedback,
    db: Session = Depends(get_db),
) -> OptimizationSession:
    """Record a thumbs up/down (and optional note) on the optimized prompt.

    A null rating clears previous feedback.
    """
    from datetime import UTC, datetime

    session = _get_session_or_404(db, session_id)
    if session.optimized_prompt is None:
        raise HTTPException(
            status_code=409, detail="This session has no optimized prompt to rate yet"
        )
    session.feedback_rating = feedback.rating
    session.feedback_comment = feedback.comment if feedback.rating else None
    session.feedback_at = datetime.now(UTC) if feedback.rating else None
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


def _resolve_method(options: OptimizeRequest, override: str | None) -> str:
    method = override or options.optimization_method
    if method not in OPTIMIZATION_METHODS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown optimization method. Expected one of: {sorted(OPTIMIZATION_METHODS)}",
        )
    return method


async def _run_optimization(
    db: Session,
    session_id: str,
    options: OptimizeRequest,
    method: str,
    progress: ProgressCallback = no_progress,
) -> dict[str, Any]:
    """Optimize a session's prompt and persist the outcome.

    Shared by the synchronous route and the background job. Raises
    HTTPException with the status the client should see.
    """
    session = _get_session_or_404(db, session_id)
    dataset_samples = (
        _load_dataset_samples(db, options.dataset_id) if options.dataset_id else None
    )
    user_feedback = _prior_user_feedback(db, session.original_prompt, session.id)

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
            progress=progress,
            gepa_budget=options.gepa_budget,
            reflection_model=options.reflection_model,
            eval_strategy=options.eval_strategy,
            user_feedback=user_feedback,
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

    details = {
        "method": optimization_result["method"],
        "improvement_score": optimization_result["improvement_score"],
        "score_type": optimization_result.get("score_type", "heuristic"),
        "processing_time": optimization_result["processing_time"],
        "metadata": optimization_result.get("metadata", {}),
    }
    # Persist the whole payload so the session can be reopened later with
    # its scoreboard, chosen examples and evolution timeline intact.
    session.result_json = json.dumps(details, default=str)

    db.commit()
    db.refresh(session)

    return {
        "message": "Prompt optimized successfully",
        "session": OptimizationSessionResponse.model_validate(session).model_dump(
            mode="json"
        ),
        "optimization_details": details,
    }


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
    """Optimize a prompt and wait for the result.

    Fine for quick runs; dataset-measured runs can take minutes, so clients
    should prefer ``/optimize/start`` and poll ``/optimize/status``.
    """
    options = options or OptimizeRequest()
    method = _resolve_method(options, optimization_method)
    return await _run_optimization(db, session_id, options, method)


@router.post("/{session_id}/optimize/start", status_code=202)
async def start_optimization(
    session_id: str,
    response: Response,
    options: OptimizeRequest | None = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Start an optimization in the background; poll ``/optimize/status``.

    Returns 202 with the job snapshot, or 409 if one is already running for
    the session. The session must exist (404 otherwise) and the dataset, if
    given, is validated before the job is accepted.
    """
    options = options or OptimizeRequest()
    method = _resolve_method(options, None)
    _get_session_or_404(db, session_id)
    if options.dataset_id:
        _load_dataset_samples(db, options.dataset_id)

    # The request's session closes when this handler returns, so the job
    # opens its own on the same engine.
    session_factory = sessionmaker(bind=db.get_bind(), autoflush=False)

    async def body(job: OptimizationJob) -> dict[str, Any]:
        with session_factory() as job_db:
            return await _run_optimization(
                job_db, session_id, options, method, job.reporter()
            )

    try:
        job = job_manager.start(session_id, body)
    except JobAlreadyRunning as e:
        raise HTTPException(status_code=409, detail=str(e)) from e

    response.headers["Location"] = f"/api/v1/sessions/{session_id}/optimize/status"
    return job.snapshot()


@router.get("/{session_id}/optimize/status")
async def optimization_status(session_id: str) -> dict[str, Any]:
    """Snapshot of the session's latest background optimization."""
    job = job_manager.get(session_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail="No optimization has been started for this session"
        )
    return job.snapshot()
