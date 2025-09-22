"""
Progress facade
----------------
Thin wrapper around websocket_service.progress_tracker used across modules.
Provides no-op behavior when websocket layer is unavailable so callers
can `await progress.update_progress(...)` safely in tests and CLIs.
"""

from __future__ import annotations

from typing import Any, Optional, Dict


class _NoOpProgress:
    async def update_progress(self, *args: Any, **kwargs: Any):
        return None

    async def report_credibility_check(self, *args: Any, **kwargs: Any):
        return None

    async def report_deduplication(self, *args: Any, **kwargs: Any):
        return None

    async def report_search_started(self, *args: Any, **kwargs: Any):
        return None

    async def report_search_completed(self, *args: Any, **kwargs: Any):
        return None

    async def report_source_found(self, *args: Any, **kwargs: Any):
        return None

    async def report_synthesis_started(self, *args: Any, **kwargs: Any):
        return None

    async def report_synthesis_completed(self, *args: Any, **kwargs: Any):
        return None

    async def report_evidence_builder_skipped(self, *args: Any, **kwargs: Any):
        return None


def _resolve_progress() -> Any:
    try:
        from .websocket_service import progress_tracker  # type: ignore
        return progress_tracker or _NoOpProgress()
    except Exception:
        return _NoOpProgress()


# Singleton-like facade used by services
progress: Any = _resolve_progress()

async def update_progress_detailed(
    research_id: str,
    phase: str,
    step: int,
    total_steps: int,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Enhanced progress updates with step counting and % completion.

    Safe no-op when websocket layer is unavailable.
    """
    try:
        await progress.update_progress(
            research_id=research_id,
            phase=phase,
            message=f"[{step}/{total_steps}] {message}",
            custom_data={
                "step": step,
                "total_steps": total_steps,
                "percentage": (step / max(total_steps, 1)) * 100.0,
                **(details or {}),
            },
        )
    except Exception:
        return None


__all__ = ["progress", "update_progress_detailed"]
