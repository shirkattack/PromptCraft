"""Progress reporting for long-running optimizations.

Services accept a ``ProgressCallback`` and call it at each stage. The
synchronous endpoint passes a no-op; the background job passes one that
updates the job snapshot the client polls.
"""

from typing import Protocol


class ProgressCallback(Protocol):
    def __call__(
        self,
        stage: str,
        message: str = "",
        *,
        current: int | None = None,
        total: int | None = None,
        best_score: float | None = None,
    ) -> None: ...


def no_progress(
    stage: str,
    message: str = "",
    *,
    current: int | None = None,
    total: int | None = None,
    best_score: float | None = None,
) -> None:
    """Default callback: discard progress."""
