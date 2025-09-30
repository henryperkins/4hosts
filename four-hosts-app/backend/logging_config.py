"""Centralised structured logging setup for the backend service.

Importing this module has the side-effect of configuring *structlog* with a
consistent, JSON-formatted pipeline that injects contextual request and
research identifiers so logs can be correlated with traces and metrics.

This helper should be imported **once** at the very top of the application
startup (e.g., inside ``core/app.py``).  Subsequent imports are effectively
no-ops because *structlog* caches its configuration.  Other modules should
call :pyfunc:`structlog.get_logger()` directly and avoid re-configuring the
library.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import structlog

# Public helpers exported by this module
__all__ = [
    "configure_logging",
    "bind_request_context",
    "get_logger",
]


def configure_logging(force: bool = False) -> None:  # noqa: D401
    """Setup structlog + stdlib bridging exactly once.

    Args:
        force: When True, reconfigure even if previously configured. Use only
               inside isolated scripts/tests that need a different renderer.
    """

    configured = getattr(structlog, "_is_configured", False)  # type: ignore[attr-defined]
    if configured and not force:
        return

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Decide renderer: JSON (default) or pretty console when LOG_PRETTY=1
    dev_mode = os.getenv("LOG_PRETTY", "0").lower() in {"1", "true", "yes"}
    if dev_mode:
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    # Processor chain shared for structlog + stdlib bridge
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Root handler uses ProcessorFormatter so stdlib logs share processors
    # Standardized processor order for consistency with structlog chain
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=[structlog.contextvars.merge_contextvars],
        processors=[
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            renderer,
        ],
    )

    # Remove existing handlers (avoid duplicates in tests / scripts)
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))

    structlog.configure(
        processors=shared_processors + [renderer],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    setattr(structlog, "_is_configured", True)  # type: ignore[attr-defined]


def bind_request_context(
    request_id: Optional[str] = None,
    research_id: Optional[str] = None,
) -> None:
    """Bind contextual IDs into structlog contextvars for subsequent logs.

    Safe to call multiple times; only provided keys are updated.
    """
    try:
        from structlog.contextvars import bind_contextvars  # type: ignore
    except ImportError:
        # structlog contextvars not available (missing extras)
        return

    payload: Dict[str, str] = {}
    if request_id:
        payload["request_id"] = request_id
    if research_id:
        payload["research_id"] = research_id
    if payload:
        try:
            bind_contextvars(**payload)
        except (TypeError, ValueError, AttributeError) as e:
            # Known safe errors during context binding - log and continue
            # Don't let context binding failures break the application
            import logging
            logging.getLogger(__name__).debug(
                "Failed to bind context: %s", e, exc_info=True
            )


def get_logger(name: Optional[str] = None):
    """Return a structlog logger; ensures configuration first."""
    configure_logging()
    return structlog.get_logger(name) if name else structlog.get_logger()


# Configure immediately on import so early log messages are captured.
configure_logging()

