from __future__ import annotations
"""Thin wrappers and utilities around OpenTelemetry instrumentation.

The helpers in this module intentionally avoid importing heavy
OpenTelemetry dependencies when the library is not installed so that
core business logic can run even in minimal environments (e.g., CI)
where tracing is not required.
"""

from contextlib import nullcontext
from types import TracebackType
from typing import Any, Dict, Optional, Type


def otel_span(name: str, attrs: Optional[Dict[str, Any]] = None):
    """Return an OpenTelemetry span context-manager.

    When *opentelemetry* is installed, the returned context-manager will set
    a *StatusCode.ERROR* on the span whenever the wrapped block raises an
    exception.  This ensures that back-end errors are reflected correctly in
    observability tools such as Jaeger and Honeycomb.

    If OpenTelemetry is unavailable, the function falls back to a no-op
    context-manager so that calling code does not need to guard imports.
    """

    try:
        from opentelemetry import trace  # type: ignore
        from opentelemetry.trace import Status, StatusCode  # type: ignore

        tracer = trace.get_tracer("four-hosts-research-api")

        class _SpanCtx:  # pylint: disable=too-few-public-methods
            def __init__(self) -> None:
                self._span_cm = tracer.start_as_current_span(
                    name, attributes=attrs or {}
                )
                self._span = None

            def __enter__(self):  # noqa: D401  (docstring intentionally omitted)
                self._span = self._span_cm.__enter__()
                return self._span

            def __exit__(
                self,
                exc_type: Optional[Type[BaseException]],
                exc: Optional[BaseException],
                tb: Optional[TracebackType],
            ) -> bool:  # noqa: D401
                # Flag status on error so traces reflect failures accurately
                if exc is not None and self._span is not None:
                    try:
                        self._span.set_status(Status(StatusCode.ERROR, str(exc)))
                    except Exception:
                        # Guard against API differences / missing methods
                        pass

                return self._span_cm.__exit__(exc_type, exc, tb)

        return _SpanCtx()

    except Exception:
        # OpenTelemetry not installed or failed; degrade gracefully.
        return nullcontext()  # type: ignore[return-value]
