"""In-process background jobs for optimizations.

One job per session at a time. Jobs live in memory for the lifetime of the
API process (a restart marks any interrupted session as failed at startup),
so this needs a single worker process, which is how the API runs.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from app.core.logging import get_logger
from app.services.progress import ProgressCallback

logger = get_logger("jobs")

JobStatus = Literal["queued", "running", "completed", "failed"]


class JobAlreadyRunning(Exception):
    pass


@dataclass
class JobProgress:
    stage: str = "queued"
    message: str = "Waiting to start"
    current: int | None = None
    total: int | None = None
    best_score: float | None = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "message": self.message,
            "current": self.current,
            "total": self.total,
            "best_score": self.best_score,
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class OptimizationJob:
    session_id: str
    status: JobStatus = "queued"
    progress: JobProgress = field(default_factory=JobProgress)
    result: dict[str, Any] | None = None
    error: str | None = None
    error_status: int | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.status in ("completed", "failed")

    def snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": self.status,
            "progress": self.progress.as_dict(),
            "history": self.history,
            "result": self.result,
            "error": self.error,
            "error_status": self.error_status,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "elapsed_seconds": round(
                (
                    (self.finished_at or datetime.now(UTC)) - self.started_at
                ).total_seconds(),
                1,
            ),
        }

    def reporter(self) -> ProgressCallback:
        """A progress callback bound to this job (safe to call from a thread)."""

        def report(
            stage: str,
            message: str = "",
            *,
            current: int | None = None,
            total: int | None = None,
            best_score: float | None = None,
        ) -> None:
            self.progress = JobProgress(
                stage=stage,
                message=message,
                current=current,
                total=total,
                best_score=(
                    best_score if best_score is not None else self.progress.best_score
                ),
            )
            self.history.append(
                {
                    "stage": stage,
                    "message": message,
                    "current": current,
                    "total": total,
                    "best_score": best_score,
                    "at": self.progress.updated_at.isoformat(),
                }
            )
            # Keep the history bounded; GEPA can emit many steps.
            if len(self.history) > 200:
                del self.history[: len(self.history) - 200]

        return report


JobBody = Callable[[OptimizationJob], Awaitable[dict[str, Any]]]


class JobManager:
    def __init__(self, max_finished: int = 100) -> None:
        self._jobs: dict[str, OptimizationJob] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._max_finished = max_finished

    def get(self, session_id: str) -> OptimizationJob | None:
        return self._jobs.get(session_id)

    def start(self, session_id: str, body: JobBody) -> OptimizationJob:
        """Schedule ``body`` for the session; one active job per session."""
        existing = self._jobs.get(session_id)
        if existing and not existing.done:
            raise JobAlreadyRunning(
                f"An optimization is already running for session {session_id}"
            )

        job = OptimizationJob(session_id=session_id)
        self._jobs[session_id] = job
        self._evict_finished()

        task = asyncio.create_task(self._run(job, body))
        self._tasks[session_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(session_id, None))
        return job

    async def wait(self, session_id: str) -> OptimizationJob | None:
        """Await the job's task (tests and the synchronous endpoint use this)."""
        task = self._tasks.get(session_id)
        if task is not None:
            await task
        return self._jobs.get(session_id)

    async def _run(self, job: OptimizationJob, body: JobBody) -> None:
        job.status = "running"
        job.progress = JobProgress(stage="starting", message="Starting optimization")
        try:
            job.result = await body(job)
            job.status = "completed"
            job.progress = JobProgress(stage="done", message="Optimization complete")
        except Exception as exc:  # the job must always reach a terminal state
            job.status = "failed"
            job.error = getattr(exc, "detail", None) or str(exc)
            job.error_status = getattr(exc, "status_code", None)
            job.progress = JobProgress(stage="failed", message=job.error)
            logger.warning(f"Optimization job for {job.session_id} failed: {job.error}")
        finally:
            job.finished_at = datetime.now(UTC)

    def _evict_finished(self) -> None:
        finished = [j for j in self._jobs.values() if j.done]
        if len(finished) <= self._max_finished:
            return
        finished.sort(key=lambda j: j.finished_at or j.started_at)
        for job in finished[: len(finished) - self._max_finished]:
            self._jobs.pop(job.session_id, None)


job_manager = JobManager()
