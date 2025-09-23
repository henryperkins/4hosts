"""metrics_facade.py
Thin shim that re-exports the `metrics` singleton from the internal
`services.metrics` module while insulating the rest of the codebase from
import-time failures when the underlying implementation is missing or
changes location.

The orchestration and telemetry layers import `services.metrics_facade`
instead of `services.metrics` directly so that:

1. Unit tests can stub or monkey-patch the facade without touching the
   real metrics implementation.
2. Optional deployments can omit the heavy metrics backend without
   breaking core application logic – the facade will gracefully degrade
   to a no-op stub.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class _MetricsProtocol(Protocol):
    """Subset of the metrics API used by the application."""

    # Global usage helpers – the concrete implementation may expose more.
    def get_latency_distributions(self) -> dict: ...  # noqa: D401,E501

    def get_fallback_rates(self) -> dict: ...

    def get_llm_usage(self) -> dict: ...

    def get_o3_usage_summary(self) -> dict: ...

    def get_paradigm_distribution(self) -> dict: ...

    def record_o3_usage(self, **kwargs: Any) -> None: ...


def _load_metrics_module() -> ModuleType | None:
    """Best-effort import of the real metrics backend."""

    try:
        # Local import to avoid hard dependency at startup.
        from services import metrics as _metrics_module  # type: ignore

        # Basic sanity check: ensure imported object looks like the interface
        if isinstance(getattr(_metrics_module, "metrics", None), _MetricsProtocol):
            return _metrics_module  # type: ignore[return-value]

        # Fallback: if the module itself implements the protocol, use it
        if isinstance(_metrics_module, _MetricsProtocol):  # type: ignore[arg-type]
            return _metrics_module
    except Exception:  # pragma: no cover – swallow import errors gracefully
        pass

    return None


# ---------------------------------------------------------------------------
# Public re-export
# ---------------------------------------------------------------------------

_mod = _load_metrics_module()

if _mod is not None:
    # Real implementation available
    metrics: _MetricsProtocol = getattr(_mod, "metrics", _mod)  # type: ignore[assignment]
else:
    # No-op stub – exposes the same attributes but does nothing.

    class _NoOpMetrics:  # noqa: D401 – simple stub
        """No-op replacement so calling code never fails."""

        def __getattr__(self, item: str) -> Any:  # noqa: D401 – dynamic attr
            def _noop(*_args: Any, **_kwargs: Any):  # noqa: D401 – ignore
                return {} if item.startswith("get_") else None

            return _noop

    metrics = _NoOpMetrics()  # type: ignore[assignment]

__all__ = ["metrics"]

