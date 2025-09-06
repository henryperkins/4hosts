"""
Progress facade
----------------
Thin wrapper around websocket_service.progress_tracker used across modules.
Provides no-op behavior when websocket layer is unavailable so callers
can `await progress.update_progress(...)` safely in tests and CLIs.
"""

from __future__ import annotations

from typing import Any, Optional


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


def _resolve_progress() -> Any:
    try:
        from .websocket_service import progress_tracker  # type: ignore
        return progress_tracker or _NoOpProgress()
    except Exception:
        return _NoOpProgress()


# Singleton-like facade used by services
progress: Any = _resolve_progress()


__all__ = ["progress"]

