"""Telemetry bridge for persisting orchestrator metrics.

This module funnels per-run search metrics into durable storage (Redis via
CacheManager) and optional Prometheus instrumentation so dashboards and alerting
pipelines can consume a consistent feed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional

import structlog

from services.cache import cache_manager

try:  # Optional dependency: prometheus client may be unavailable in some envs
    from services.monitoring import PrometheusMetrics  # type: ignore
except Exception:  # pragma: no cover - fallback for environments without prometheus
    PrometheusMetrics = None  # type: ignore


logger = structlog.get_logger(__name__)


class TelemetryPipeline:
    """Coordinates metric persistence across supported backends."""

    def __init__(self) -> None:
        self._prometheus: Optional[PrometheusMetrics] = None

    def bind_prometheus(self, prometheus: PrometheusMetrics) -> None:
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

        total_queries = _as_int(record.get("total_queries"))
        total_results = _as_int(record.get("total_results"))
        processing_time = _as_float(record.get("processing_time_seconds"))
        dedup_rate = _as_float(record.get("deduplication_rate"))

        self._prometheus.search_runs_total.labels(**labels).inc(1)

        if total_queries:
            self._prometheus.search_queries_total.labels(**labels).inc(total_queries)

        if total_results:
            self._prometheus.search_results_total.labels(**labels).inc(total_results)

        if processing_time:
            self._prometheus.search_processing_time.labels(**labels).observe(processing_time)

        if dedup_rate is not None:
            dedup_rate = max(0.0, min(1.0, dedup_rate))
            self._prometheus.search_deduplication_rate.labels(**labels).set(dedup_rate)

        for provider in _coerce_iterable(record.get("apis_used")):
            provider_name = str(provider or "unknown").lower()
            self._prometheus.search_provider_usage.labels(provider=provider_name).inc(1)

        for provider, cost in _iter_costs(record.get("provider_costs")):
            provider_name = str(provider or "unknown").lower()
            if cost:
                self._prometheus.search_provider_cost.labels(provider=provider_name).inc(cost)


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


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

