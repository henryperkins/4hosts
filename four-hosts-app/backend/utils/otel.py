from __future__ import annotations
from contextlib import nullcontext
from typing import Any, Dict


def otel_span(name: str, attrs: Dict[str, Any] | None = None):
    """
    Return an OpenTelemetry span context-manager.
    Falls back to a no-op when OpenTelemetry is unavailable.
    """
    try:
        from opentelemetry import trace
        tracer = trace.get_tracer("four-hosts-research-api")
        return tracer.start_as_current_span(name, attributes=attrs or {})
    except Exception:
        return nullcontext()  # type: ignore[return-value]
