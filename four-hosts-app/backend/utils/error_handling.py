"""
Lightweight error logging and warning helpers without Prometheus.
Prefer structlog when available, but fall back to stdlib logging.
"""

from __future__ import annotations

from typing import Any, Dict, List

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


def add_warning(meta: Dict[str, Any] | None, code: str, message: str, **extra: Any) -> Dict[str, Any]:
    """Append a warning to a metadata dict; creates the warnings list if missing."""
    meta = meta or {}
    warnings: List[Dict[str, Any]] = list(meta.get("warnings", []) or [])
    warnings.append({"code": code, "message": message, **extra})
    meta["warnings"] = warnings
    meta["degraded"] = True
    return meta

