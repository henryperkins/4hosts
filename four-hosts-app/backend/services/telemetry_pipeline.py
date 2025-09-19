"""Telemetry bridge for persisting orchestrator metrics.

This module funnels per-run search metrics into durable storage
(Redis via CacheManager) and optional Prometheus instrumentation so
dashboards and alerting pipelines can consume a consistent feed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, TYPE_CHECKING

import structlog

# pylint: disable=import-error

from services.cache import cache_manager
from utils.type_coercion import as_int, as_float

# Optional dependency: expose the type only for static type checkers.
if TYPE_CHECKING:  # pragma: no cover
    from services.monitoring import PrometheusMetrics  # noqa: F401


logger = structlog.get_logger(__name__)


class TelemetryPipeline:
    """Coordinates metric persistence across supported backends."""

    def __init__(self) -> None:
        # Prometheus backend is optional; bound at runtime.
        # Prometheus backend is optional; use a generic Any to avoid runtime
        # type errors when the dependency is unavailable.
        self._prometheus: Optional[Any] = None

    def bind_prometheus(self, prometheus: Any) -> None:
        """Attach a Prometheus registry if available."""

        self._prometheus = prometheus

    async def record_search_run(self, record: Dict[str, Any]) -> None:
        """Persist a single research run's telemetry.

        The record is expected to contain serialisable primitives (ints/floats/
        strings); timestamps are normalised to ISO-8601 with UTC timezone.
        """

        ts = record.get("timestamp")
        if not isinstance(ts, str):
            record["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Always keep a copy for fallback storage to avoid mutation surprises
        safe_record = dict(record)

        # Compute deduplication_rate if absent
        # using dedup_stats when available
        try:
            if (
                "deduplication_rate" not in safe_record
                or safe_record.get("deduplication_rate") is None
            ):
                ds = safe_record.get("dedup_stats") or {}
                if isinstance(ds, dict):
                    orig = as_int(ds.get("original_count"))
                    final = as_int(ds.get("final_count"))
                    if orig > 0:
                        rate = 1.0 - (float(final) / float(orig))
                        # Clamp to [0.0, 1.0]
                        safe_record["deduplication_rate"] = max(
                            0.0, min(1.0, float(rate))
                        )
        except Exception:
            # Best-effort; ignore failures
            pass

        try:
            await cache_manager.record_search_metrics(safe_record)
        except Exception:
            logger.warning("telemetry_cache_record_failed", exc_info=True)

        try:
            self._record_prometheus_metrics(safe_record)
        except Exception:
            logger.warning("telemetry_prometheus_record_failed", exc_info=True)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _record_prometheus_metrics(self, record: Mapping[str, Any]) -> None:
        if not self._prometheus:
            return

        paradigm = str(record.get("paradigm") or "unknown").lower()
        depth = str(record.get("depth") or "standard").lower()
        labels = {"paradigm": paradigm, "depth": depth}

        total_queries = as_int(record.get("total_queries"))
        total_results = as_int(record.get("total_results"))
        processing_time = as_float(record.get("processing_time_seconds"))
        dedup_rate = as_float(record.get("deduplication_rate"))

        self._prometheus.search_runs_total.labels(**labels).inc(1)

        if total_queries:
            self._prometheus.search_queries_total.labels(**labels).inc(
                total_queries
            )

        if total_results:
            self._prometheus.search_results_total.labels(**labels).inc(
                total_results
            )

        if processing_time:
            self._prometheus.search_processing_time.labels(**labels).observe(
                processing_time
            )

        if dedup_rate is not None:
            dedup_rate = max(0.0, min(1.0, dedup_rate))
            self._prometheus.search_deduplication_rate.labels(**labels).set(
                dedup_rate
            )

        for provider in _coerce_iterable(record.get("apis_used")):
            provider_name = str(provider or "unknown").lower()
            self._prometheus.search_provider_usage.labels(
                provider=provider_name
            ).inc(1)

        for provider, cost in _iter_costs(record.get("provider_costs")):
            provider_name = str(provider or "unknown").lower()
            if cost:
                self._prometheus.search_provider_cost.labels(
                    provider=provider_name
                ).inc(cost)


# Coercion helpers moved to utils.type_coercion


# (see above)


def _coerce_iterable(value: Any) -> Iterable[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return value
    return [value]


def _iter_costs(value: Any) -> Iterable[tuple[str, float]]:
    if not isinstance(value, Mapping):
        return []
    out = []
    for key, val in value.items():
        try:
            out.append((str(key), float(val)))
        except Exception:
            continue
    return out


telemetry_pipeline = TelemetryPipeline()

