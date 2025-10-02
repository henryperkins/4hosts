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
from utils.type_coercion import as_int, as_float, coerce_iterable

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

        # Concurrency guard so bursts of requests don't overwhelm Redis or
        # block the event-loop – make the limit configurable via env-var.
        import asyncio, os  # local to avoid top-level import for tests

        max_concurrent = int(os.getenv("TELEMETRY_MAX_PARALLEL", "8"))
        self._sem: "asyncio.Semaphore" = asyncio.Semaphore(max(1, max_concurrent))

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

        # Prevent unbounded concurrent writes that could saturate Redis under
        # high QPS.  We purposely keep the critical section small – only the
        # IO-bound cache write – so that we do not serialise the entire
        # function.
        async with self._sem:
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

        # New optional fields
        grounding_cov = as_float(record.get("grounding_coverage"))
        agent_iterations = as_int(record.get("agent_iterations"))
        agent_new_queries = as_int(record.get("agent_new_queries"))
        evidence_quotes_cnt = as_int(record.get("evidence_quotes_count"))
        evidence_docs_cnt = as_int(record.get("evidence_documents_count"))
        evidence_tokens = as_int(record.get("evidence_total_tokens"))

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

        # ------------------------------------------------------------------ #
        # Analysis telemetry instrumentation
        # ------------------------------------------------------------------ #

        analysis_duration = as_float(record.get("analysis_duration_seconds"))
        if analysis_duration:
            self._prometheus.analysis_phase_duration_seconds.labels(**labels).observe(
                analysis_duration
            )

        analysis_sources = as_int(record.get("analysis_sources_total"))
        if analysis_sources:
            self._prometheus.analysis_phase_sources_total.labels(**labels).observe(
                analysis_sources
            )

        analysis_updates = as_int(record.get("analysis_progress_updates"))
        if analysis_updates:
            self._prometheus.analysis_phase_updates_total.labels(**labels).observe(
                analysis_updates
            )

        analysis_rate = as_float(record.get("analysis_updates_per_second"))
        if analysis_rate is not None and analysis_rate >= 0:
            self._prometheus.analysis_phase_updates_per_second.labels(**labels).set(
                analysis_rate
            )

        avg_gap = as_float(record.get("analysis_avg_update_gap_seconds"))
        if avg_gap is not None and avg_gap >= 0:
            self._prometheus.analysis_phase_avg_gap_seconds.labels(**labels).set(avg_gap)

        p95_gap = as_float(record.get("analysis_p95_update_gap_seconds"))
        if p95_gap is not None and p95_gap >= 0:
            self._prometheus.analysis_phase_p95_gap_seconds.labels(**labels).set(p95_gap)

        first_gap = as_float(record.get("analysis_first_update_gap_seconds"))
        if first_gap is not None and first_gap >= 0:
            self._prometheus.analysis_phase_first_gap_seconds.labels(**labels).set(first_gap)

        last_gap = as_float(record.get("analysis_last_update_gap_seconds"))
        if last_gap is not None and last_gap >= 0:
            self._prometheus.analysis_phase_last_gap_seconds.labels(**labels).set(last_gap)

        if record.get("analysis_cancelled"):
            self._prometheus.analysis_phase_cancelled_total.labels(**labels).inc(1)

        # ------------------------------------------------------------------ #
        # New Metrics Recording
        # ------------------------------------------------------------------ #

        if grounding_cov is not None:
            grounding_cov = max(0.0, min(1.0, grounding_cov))
            self._prometheus.grounding_coverage_ratio.labels(**labels).set(
                grounding_cov
            )

        if agent_iterations:
            self._prometheus.agent_iterations_total.labels(**labels).inc(
                agent_iterations
            )

        if agent_new_queries:
            self._prometheus.agent_new_queries_total.labels(**labels).inc(
                agent_new_queries
            )

        if evidence_quotes_cnt:
            self._prometheus.evidence_quotes_total.labels(**labels).inc(
                evidence_quotes_cnt
            )

        if evidence_docs_cnt:
            self._prometheus.evidence_documents_total.labels(**labels).inc(
                evidence_docs_cnt
            )

        if evidence_tokens:
            self._prometheus.evidence_tokens_total.labels(**labels).inc(
                evidence_tokens
            )

        for provider in coerce_iterable(record.get("apis_used")):
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


# _coerce_iterable moved to utils.type_coercion.coerce_iterable


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
