"""
Error handlers for the Four Hosts Research API
"""

import structlog
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

logger = structlog.get_logger(__name__)


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    """Handle validation errors with detailed messages"""
    errors = exc.errors()
    error_messages = []

    for error in errors:
        field = ".".join(str(x) for x in error["loc"][1:])  # Skip 'body'
        msg = error["msg"]
        error_type = error["type"]

        if field == "query" and "at least 10 characters" in msg:
            error_messages.append("Query must be at least 10 characters long")
        elif field == "query" and "at most 500 characters" in msg:
            error_messages.append("Query must be at most 500 characters long")
        elif field == "query" and error_type == "value_error.missing":
            error_messages.append("Query field is required")
        elif field == "search_context_size":
            error_messages.append(
                "search_context_size must be one of: small, medium, large"
            )
        else:
            error_messages.append(f"{field}: {msg}")

    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "detail": error_messages,
            "expected_format": {
                "query": "your research query (10-500 chars required)",
                "paradigm": "dolores|teddy|bernard|maeve (optional)",
                "search_context_size": "small|medium|large (optional, default: medium)",
                "user_location": {"country": "US", "city": "NYC"}  # optional
            },
            "request_id": getattr(request.state, "request_id", "unknown"),
        },
    )


async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    # Track errors in monitoring if available
    try:
        if hasattr(request.app.state, "monitoring"):
            insights = request.app.state.monitoring.get("insights")
            if insights:
                await insights.track_error(
                    error_type="http_error",
                    severity="warning",
                    details={
                        "status_code": exc.status_code,
                        "detail": exc.detail,
                        "path": request.url.path,
                    },
                )
    except Exception:
        pass  # Don't let monitoring errors break the response

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "detail": exc.detail,
            "status_code": exc.status_code,
            "request_id": getattr(request.state, "request_id", "unknown"),
        },
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error("Unhandled exception",
                error=str(exc),
                error_type=type(exc).__name__,
                path=request.url.path,
                request_id=getattr(request.state, "request_id", "unknown"))

    # Track errors in monitoring if available
    try:
        if hasattr(request.app.state, "monitoring"):
            insights = request.app.state.monitoring.get("insights")
            if insights:
                await insights.track_error(
                    error_type="unhandled_error",
                    severity="error",
                    details={
                        "error": str(exc),
                        "type": type(exc).__name__,
                        "path": request.url.path,
                    },
                )
    except Exception:
        pass  # Don't let monitoring errors break the response

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred",
            "request_id": getattr(request.state, "request_id", "unknown"),
        },
    )
