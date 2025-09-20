"""
Lightweight error logging and warning helpers without Prometheus.
Prefer structlog when available, but fall back to stdlib logging.
"""

from __future__ import annotations

from typing import Any, Dict, List
import asyncio
import traceback
from contextlib import contextmanager
from typing import Iterator, Callable, TypeVar

try:
    import structlog  # type: ignore

    _logger = structlog.get_logger(__name__)

    def _log(level: str, event: str, **fields: Any) -> None:
        log = getattr(_logger, level, _logger.warning)
        log(event, **fields)

except Exception:  # pragma: no cover - fallback when structlog missing
    import logging

    _logger = logging.getLogger(__name__)

    def _log(level: str, event: str, **fields: Any) -> None:
        msg = f"{event} | " + ", ".join(f"{k}={v}" for k, v in fields.items())
        if level == "error":
            _logger.error(msg)
        elif level == "info":
            _logger.info(msg)
        else:
            _logger.warning(msg)


def log_exception(context: str, exc: Exception, **fields: Any) -> None:
    """Log an exception with context; never raises."""
    try:
        _log("warning", context, error=str(exc), **fields)
    except Exception:
        # Avoid secondary failures during error handling
        pass


def add_warning(
    meta: Dict[str, Any] | None,
    code: str,
    message: str,
    **extra: Any
) -> Dict[str, Any]:
    """Append a warning to a metadata dict; creates the list if missing."""
    meta = meta or {}
    warnings: List[Dict[str, Any]] = list(meta.get("warnings", []) or [])
    warnings.append({"code": code, "message": message, **extra})
    meta["warnings"] = warnings
    meta["degraded"] = True
    return meta


# Consolidated error-handling helpers
T = TypeVar("T")


@contextmanager
def safely(
    context: str,
    *,
    non_fatal: bool = False,
    **fields: Any
) -> Iterator[None]:
    """Context manager that logs and re-raises by default.

    Set non_fatal=True to swallow after logging.
    """
    try:
        yield
    except Exception as exc:  # pragma: no cover - defensive
        log_exception(context, exc, **fields)
        if not non_fatal:
            raise


def safe_call(
    fn: Callable[..., T],
    *,
    context: str,
    non_fatal: bool = False,
    **fields: Any
) -> Callable[..., T]:
    """Decorator-like wrapper that logs and re-raises by default."""
    def _wrapped(*args: Any, **kwargs: Any) -> T:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            log_exception(context, exc, **fields)
            if not non_fatal:
                raise
            return None  # type: ignore[return-value]
    return _wrapped


# --------------------------------------------------------------------------- #
#                            Error classification                             #
# --------------------------------------------------------------------------- #

# Simple, extensible mapping for user-facing messages and severities.
ERROR_CLASSIFICATIONS: Dict[str, Dict[str, Any]] = {
    "rate_limit": {
        "user_message": "Search provider temporarily unavailable. Using alternative sources...",
        "action": "fallback",
        "severity": "warning",
    },
    "no_results": {
        "user_message": "No relevant results found. Trying broader search terms...",
        "action": "expand_query",
        "severity": "info",
    },
    "llm_timeout": {
        "user_message": "AI processing is taking longer than expected. Please wait...",
        "action": "retry",
        "severity": "warning",
    },
}


def identify_error_type(error: Exception) -> str:
    """Heuristic classification of common runtime errors.

    Keeps dependencies light and avoids tight coupling to provider classes.
    """
    name = type(error).__name__.lower()
    msg = str(error).lower()
    if "rate" in name and "limit" in name:
        return "rate_limit"
    if "quota" in msg or "429" in msg:
        return "rate_limit"
    if isinstance(error, asyncio.TimeoutError) or "timeout" in name or "timed out" in msg:
        return "llm_timeout"
    return "unknown"


def classify_and_log_error(
    error: Exception,
    research_id: str | None,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Classify errors and provide user-friendly feedback while logging.

    Returns a dict containing at least `user_message` and `severity`.
    """
    try:
        classification = identify_error_type(error)
        info = ERROR_CLASSIFICATIONS.get(
            classification,
            {"user_message": "An unexpected error occurred", "severity": "error"},
        )
        _log(
            "error" if info.get("severity") == "error" else "warning",
            "Classified error occurred",
            research_id=research_id,
            error_type=classification,
            error_message=str(error),
            severity=info.get("severity"),
            context=context,
            stack_trace=(traceback.format_exc() if info.get("severity") == "error" else None),
        )
        return info
    except Exception:
        # Defensive: never raise from error reporting
        return {"user_message": "An unexpected error occurred", "severity": "error"}
