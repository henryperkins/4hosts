"""
Monitoring and Logging Infrastructure for Four Hosts Research API
Phase 5: Production-Ready Features
"""

import os
import time
import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable
from collections import defaultdict, deque
import psutil

# Import logging configuration.
# In container builds the code lives at /app without a 'backend' package prefix.
# Try canonical import first, then fall back to local module path.
try:
    from backend.logging_config import configure_logging  # noqa: F401
except Exception:  # pragma: no cover - container path fallback
    from logging_config import configure_logging  # noqa: F401

import structlog
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest
from prometheus_client.core import CollectorRegistry
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from dataclasses import dataclass, field
from enum import Enum
import traceback

# Structured logger (configured globally via backend.logging_config)
logger = structlog.get_logger(__name__)

# --- Monitoring Configuration ---


class MetricType(str, Enum):
    """Types of metrics"""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MonitoringConfig:
    """Configuration for monitoring services"""

    enable_prometheus: bool = True
    enable_opentelemetry: bool = True
    prometheus_port: int = 9090  # DEPRECATED: unused
    otlp_endpoint: str = field(
        default_factory=lambda: os.getenv("OTLP_ENDPOINT", "localhost:4317")
    )
    service_name: str = "four-hosts-research-api"
    environment: str = field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development")
    )
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


# --- Prometheus Metrics ---


class PrometheusMetrics:
    """Prometheus metrics collection"""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()

        # API Metrics
        self.api_requests_total = Counter(
            "api_requests_total",
            "Total API requests",
            ["method", "endpoint", "status", "paradigm"],
            registry=self.registry,
        )

        self.api_request_duration = Histogram(
            "api_request_duration_seconds",
            "API request duration",
            ["method", "endpoint", "paradigm"],
            registry=self.registry,
        )

        # Research Metrics
        self.research_queries_total = Counter(
            "research_queries_total",
            "Total research queries",
            ["paradigm", "depth", "status"],
            registry=self.registry,
        )

        self.research_duration = Histogram(
            "research_duration_seconds",
            "Research query duration",
            ["paradigm", "depth"],
            buckets=[1, 5, 10, 30, 60, 120, 300, 600],
            registry=self.registry,
        )

        self.sources_analyzed = Histogram(
            "sources_analyzed_total",
            "Number of sources analyzed per query",
            ["paradigm"],
            buckets=[10, 25, 50, 100, 200, 500, 1000],
            registry=self.registry,
        )

        # Unified query planner metrics
        self.search_runs_total = Counter(
            "search_runs_total",
            "Total research orchestrator runs recorded",
            ["paradigm", "depth"],
            registry=self.registry,
        )

        self.search_queries_total = Counter(
            "search_queries_total",
            "Total search queries executed across runs",
            ["paradigm", "depth"],
            registry=self.registry,
        )

        self.search_results_total = Counter(
            "search_results_total",
            "Total search results collected across runs",
            ["paradigm", "depth"],
            registry=self.registry,
        )

        self.search_processing_time = Histogram(
            "search_processing_time_seconds",
            "Overall processing time per research run",
            ["paradigm", "depth"],
            buckets=[1, 5, 10, 30, 60, 120, 300, 600],
            registry=self.registry,
        )

        self.search_deduplication_rate = Gauge(
            "search_deduplication_rate",
            "Observed deduplication rate per run",
            ["paradigm", "depth"],
            registry=self.registry,
        )

        # Analysis / credibility phase telemetry — tracks smoothness of
        # incremental updates after the refactor.
        self.analysis_phase_duration_seconds = Histogram(
            "analysis_phase_duration_seconds",
            "Duration of the credibility analysis phase",
            ["paradigm", "depth"],
            buckets=[0.25, 0.5, 1, 2, 5, 10, 20, 40, 80, 160],
            registry=self.registry,
        )

        self.analysis_phase_sources_total = Histogram(
            "analysis_phase_sources_total",
            "Number of sources evaluated during analysis",
            ["paradigm", "depth"],
            buckets=[1, 5, 10, 20, 40, 80, 160, 320],
            registry=self.registry,
        )

        self.analysis_phase_updates_total = Histogram(
            "analysis_phase_updates_total",
            "Count of progress updates emitted during analysis",
            ["paradigm", "depth"],
            buckets=[1, 5, 10, 20, 40, 80, 160, 320],
            registry=self.registry,
        )

        self.analysis_phase_updates_per_second = Gauge(
            "analysis_phase_updates_per_second",
            "Rate of analysis progress updates per second",
            ["paradigm", "depth"],
            registry=self.registry,
        )

        self.analysis_phase_avg_gap_seconds = Gauge(
            "analysis_phase_avg_gap_seconds",
            "Average gap between analysis progress updates (seconds)",
            ["paradigm", "depth"],
            registry=self.registry,
        )

        self.analysis_phase_p95_gap_seconds = Gauge(
            "analysis_phase_p95_gap_seconds",
            "95th percentile gap between analysis progress updates (seconds)",
            ["paradigm", "depth"],
            registry=self.registry,
        )

        self.analysis_phase_first_gap_seconds = Gauge(
            "analysis_phase_first_gap_seconds",
            "Time from phase start to the first analysis update",
            ["paradigm", "depth"],
            registry=self.registry,
        )

        self.analysis_phase_last_gap_seconds = Gauge(
            "analysis_phase_last_gap_seconds",
            "Gap between the final two analysis updates",
            ["paradigm", "depth"],
            registry=self.registry,
        )

        self.analysis_phase_cancelled_total = Counter(
            "analysis_phase_cancelled_total",
            "Number of cancelled analysis/credibility batches",
            ["paradigm", "depth"],
            registry=self.registry,
        )

        self.search_provider_usage = Counter(
            "search_provider_queries_total",
            "Queries executed per provider",
            ["provider"],
            registry=self.registry,
        )

        self.search_provider_cost = Counter(
            "search_provider_cost_total",
            "Total spend per provider in USD",
            ["provider"],
            registry=self.registry,
        )

        # ---------------------------- New Metrics ---------------------------- #
        # These gauges/counters expose additional research-time telemetry
        # required by newer dashboards (grounding, agent loop, evidence).

        # Grounding / citation coverage ratio (0.0–1.0)
        self.grounding_coverage_ratio = Gauge(
            "grounding_coverage_ratio",
            "Ratio of sentences grounded with citations (0-1)",
            ["paradigm", "depth"],
            registry=self.registry,
        )

        # Agent loop metrics – counters so we can compute totals and rates.
        self.agent_iterations_total = Counter(
            "agent_iterations_total",
            "Total iterations executed by the agentic loop",
            ["paradigm", "depth"],
            registry=self.registry,
        )

        self.agent_new_queries_total = Counter(
            "agent_new_queries_total",
            "New search queries generated by the agentic loop",
            ["paradigm", "depth"],
            registry=self.registry,
        )

        # Evidence statistics – counters because they are naturally
        # accumulative per research run.
        self.evidence_quotes_total = Counter(
            "evidence_quotes_total",
            "Total evidence quotes extracted across runs",
            ["paradigm", "depth"],
            registry=self.registry,
        )

        self.evidence_documents_total = Counter(
            "evidence_documents_total",
            "Total unique documents contributing evidence",
            ["paradigm", "depth"],
            registry=self.registry,
        )

        self.evidence_tokens_total = Counter(
            "evidence_tokens_total",
            "Total LLM tokens used for evidence selection",
            ["paradigm", "depth"],
            registry=self.registry,
        )

        # Evidence fetch cache metrics (no labels)
        self.evidence_cache_hits_total = Counter(
            "evidence_cache_hits_total",
            "Evidence text cache hits",
            registry=self.registry,
        )
        self.evidence_cache_misses_total = Counter(
            "evidence_cache_misses_total",
            "Evidence text cache misses",
            registry=self.registry,
        )

        # System Metrics
        self.cpu_usage = Gauge(
            "system_cpu_usage_percent", "CPU usage percentage", registry=self.registry
        )

        self.memory_usage = Gauge(
            "system_memory_usage_percent",
            "Memory usage percentage",
            registry=self.registry,
        )

        self.active_connections = Gauge(
            "active_connections_total",
            "Active connections",
            ["connection_type"],
            registry=self.registry,
        )

        # Rate Limiting Metrics
        self.rate_limit_hits = Counter(
            "rate_limit_hits_total",
            "Rate limit hits",
            ["limit_type", "role"],
            registry=self.registry,
        )

        # Error Metrics
        self.errors_total = Counter(
            "errors_total",
            "Total errors",
            ["error_type", "severity"],
            registry=self.registry,
        )

        # Business Metrics
        self.paradigm_distribution = Counter(
            "paradigm_classification_total",
            "Paradigm classification distribution",
            ["primary_paradigm", "secondary_paradigm"],
            registry=self.registry,
        )

        self.synthesis_quality_score = Summary(
            "synthesis_quality_score",
            "Quality score of synthesized answers",
            ["paradigm"],
            registry=self.registry,
        )


# --- OpenTelemetry Setup ---


class OpenTelemetryService:
    """OpenTelemetry tracing and metrics service"""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.tracer_provider = None
        self.meter_provider = None
        self.tracer = None
        self.meter = None

        if config.enable_opentelemetry:
            self._setup_tracing()
            self._setup_metrics()

    def _setup_tracing(self):
        """Setup OpenTelemetry tracing"""
        # Create tracer provider
        self.tracer_provider = TracerProvider()

        # Create OTLP exporter
        insecure = os.getenv("OTLP_INSECURE", "0").lower() in {"1", "true", "yes"}
        otlp_exporter = OTLPSpanExporter(
            endpoint=self.config.otlp_endpoint,
            insecure=insecure,
        )

        # Create span processor
        span_processor = BatchSpanProcessor(otlp_exporter)
        self.tracer_provider.add_span_processor(span_processor)

        # Set tracer provider
        trace.set_tracer_provider(self.tracer_provider)

        # Get tracer
        self.tracer = trace.get_tracer(self.config.service_name, "1.0.0")

    def _setup_metrics(self):
        """Setup OpenTelemetry metrics"""
        # Create metric reader
        insecure = os.getenv("OTLP_INSECURE", "0").lower() in {"1", "true", "yes"}
        export_ms = int(os.getenv("OTLP_METRIC_EXPORT_INTERVAL_MS", "10000"))
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=self.config.otlp_endpoint, insecure=insecure),
            export_interval_millis=export_ms,  # Export interval (ms)
        )

        # Create meter provider
        self.meter_provider = MeterProvider(metric_readers=[metric_reader])

        # Set meter provider
        metrics.set_meter_provider(self.meter_provider)

        # Get meter
        self.meter = metrics.get_meter(self.config.service_name, "1.0.0")

    def create_span(self, name: str, attributes: Dict[str, Any] = None):
        """Create a new span for tracing"""
        if not self.tracer:
            return None

        return self.tracer.start_as_current_span(name, attributes=attributes or {})


# --- Performance Monitoring ---


class PerformanceMonitor:
    """Monitor application performance metrics"""

    def __init__(self):
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.current_metrics = {}
        self._monitoring_task = None

    async def start_monitoring(self, interval: int = 10):
        """Start background performance monitoring"""
        self._monitoring_task = asyncio.create_task(self._monitor_loop(interval))

    async def stop_monitoring(self):
        """Stop performance monitoring"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            await asyncio.gather(self._monitoring_task, return_exceptions=True)

    async def _monitor_loop(self, interval: int):
        """Main monitoring loop with accurate interval pacing and non-blocking CPU sampling."""
        # Prime CPU percent to enable non-blocking sampling
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

        while True:
            loop_start = time.time()
            try:
                # Collect system metrics (non-blocking CPU sample)
                cpu_sample_sec = float(os.getenv("PERF_CPU_SAMPLE_SEC", "0.0") or 0.0)
                if cpu_sample_sec > 0.0:
                    cpu_percent = psutil.cpu_percent(interval=min(cpu_sample_sec, max(0.0, interval)))
                else:
                    cpu_percent = psutil.cpu_percent(interval=None)

                memory = psutil.virtual_memory()
                disk = psutil.disk_usage("/")

                # Network I/O
                net_io = psutil.net_io_counters()

                # Process specific metrics
                process = psutil.Process()
                try:
                    conns = process.net_connections()  # psutil ≥5.9
                except Exception:
                    try:
                        conns = process.connections()  # Fallback for older psutil
                    except Exception:
                        conns = []
                process_info = {
                    "cpu_percent": process.cpu_percent(),
                    "memory_mb": process.memory_info().rss / 1024 / 1024,
                    "threads": process.num_threads(),
                    "connections": len(conns),
                }

                # Update current metrics
                self.current_metrics = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "system": {
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory.percent,
                        "memory_available_gb": memory.available / 1024 / 1024 / 1024,
                        "disk_percent": disk.percent,
                        "network_sent_mb": net_io.bytes_sent / 1024 / 1024,
                        "network_recv_mb": net_io.bytes_recv / 1024 / 1024,
                    },
                    "process": process_info,
                }

                # Store in history
                self.metrics_history["system"].append(self.current_metrics)

                # Log metrics
                logger.info("performance_metrics", **self.current_metrics)

            except Exception as e:
                logger.error("performance_monitoring_error", error=str(e))

            # Sleep the remaining time to maintain the requested interval cadence
            elapsed = max(0.0, time.time() - loop_start)
            try:
                await asyncio.sleep(max(0.0, float(interval) - elapsed))
            except Exception:
                # Defensive: never propagate sleep errors
                pass

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.current_metrics

    def get_metrics_summary(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get summary of metrics over time window"""
        if not self.metrics_history["system"]:
            return {}

        recent_metrics = list(self.metrics_history["system"])[
            -window_minutes * 6 :
        ]  # 10s intervals

        if not recent_metrics:
            return {}

        # Calculate averages
        cpu_avg = sum(m["system"]["cpu_percent"] for m in recent_metrics) / len(
            recent_metrics
        )
        memory_avg = sum(m["system"]["memory_percent"] for m in recent_metrics) / len(
            recent_metrics
        )

        return {
            "window_minutes": window_minutes,
            "samples": len(recent_metrics),
            "cpu_percent_avg": round(cpu_avg, 2),
            "memory_percent_avg": round(memory_avg, 2),
            "cpu_percent_max": max(m["system"]["cpu_percent"] for m in recent_metrics),
            "memory_percent_max": max(
                m["system"]["memory_percent"] for m in recent_metrics
            ),
        }


# --- Application Insights ---


class ApplicationInsights:
    """Track application-specific metrics and insights"""

    def __init__(self, prometheus_metrics: PrometheusMetrics):
        self.prometheus = prometheus_metrics
        self.insights_data = defaultdict(lambda: deque(maxlen=10000))

    async def track_research_query(
        self, research_id: str, query: str, paradigm: str, depth: str, start_time: float
    ):
        """Track research query metrics"""
        duration = time.time() - start_time

        # Update Prometheus metrics (schedule to avoid adding latency to request path)
        try:
            loop = asyncio.get_running_loop()
            rq = self.prometheus.research_queries_total.labels(
                paradigm=paradigm,
                depth=depth,
                status="completed",
            )
            rd = self.prometheus.research_duration.labels(
                paradigm=paradigm,
                depth=depth,
            )
            loop.call_soon(rq.inc)
            loop.call_soon(rd.observe, duration)
        except Exception:
            # Fallback to direct increment if loop not available
            self.prometheus.research_queries_total.labels(
                paradigm=paradigm, depth=depth, status="completed"
            ).inc()
            self.prometheus.research_duration.labels(
                paradigm=paradigm, depth=depth
            ).observe(duration)

        # Store insight data
        self.insights_data["research_queries"].append(
            {
                "research_id": research_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query": query[:100],  # Truncate for storage
                "paradigm": paradigm,
                "depth": depth,
                "duration_seconds": duration,
            }
        )

        # Log structured data
        logger.info(
            "research_query_completed",
            research_id=research_id,
            paradigm=paradigm,
            depth=depth,
            duration=duration,
        )

    async def track_api_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
        paradigm: Optional[str] = None,
    ):
        """Track API request metrics"""
        # Update Prometheus metrics (defer to next loop turn to avoid blocking)
        try:
            loop = asyncio.get_running_loop()
            ar = self.prometheus.api_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status=str(status_code),
                paradigm=paradigm or "none",
            )
            ad = self.prometheus.api_request_duration.labels(
                method=method,
                endpoint=endpoint,
                paradigm=paradigm or "none",
            )
            loop.call_soon(ar.inc)
            loop.call_soon(ad.observe, duration)
        except Exception:
            self.prometheus.api_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status=str(status_code),
                paradigm=paradigm or "none",
            ).inc()
            self.prometheus.api_request_duration.labels(
                method=method, endpoint=endpoint, paradigm=paradigm or "none"
            ).observe(duration)

        # Log if slow request
        if duration > 2.0:  # 2 seconds threshold
            logger.warning(
                "slow_api_request",
                method=method,
                endpoint=endpoint,
                duration=duration,
                status_code=status_code,
            )

    async def track_error(
        self, error_type: str, severity: str, details: Dict[str, Any]
    ):
        """Track application errors with controlled label cardinality."""
        # Normalise severity and error_type to avoid label explosion
        allowed_severities = {"debug", "info", "warning", "error", "critical"}
        sev = (severity or "error").lower()
        if sev not in allowed_severities:
            sev = "error"

        allowed_error_types = {
            "request_processing", "db", "cache", "search_provider", "llm",
            "webhook", "auth", "rate_limit", "validation", "unknown"
        }
        etype = (error_type or "unknown").lower()
        if etype not in allowed_error_types:
            etype = "other"

        # Update Prometheus metrics (schedule to reduce impact on request latency)
        try:
            loop = asyncio.get_running_loop()
            err = self.prometheus.errors_total.labels(error_type=etype, severity=sev)
            loop.call_soon(err.inc)
        except Exception:
            self.prometheus.errors_total.labels(
                error_type=etype, severity=sev
            ).inc()

        # Store error data (sanitised)
        self.insights_data["errors"].append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error_type": etype,
                "severity": sev,
                "details": details,
                "traceback": traceback.format_exc() if sev == "critical" else None,
            }
        )

        # Log error (structured, consistent)
        log_method = getattr(logger, sev, logger.error)
        log_method(
            "app_error",
            error_type=etype,
            severity=sev,
            **details,
        )

    def get_insights_summary(self) -> Dict[str, Any]:
        """Get summary of application insights"""
        return {
            "total_queries": len(self.insights_data["research_queries"]),
            "total_errors": len(self.insights_data["errors"]),
            "recent_queries": list(self.insights_data["research_queries"])[-10:],
            "recent_errors": list(self.insights_data["errors"])[-10:],
            "paradigm_distribution": self._calculate_paradigm_distribution(),
            "error_distribution": self._calculate_error_distribution(),
        }

    def _calculate_paradigm_distribution(self) -> Dict[str, int]:
        """Calculate distribution of paradigms from recent queries"""
        distribution = defaultdict(int)
        for query in self.insights_data["research_queries"]:
            distribution[query["paradigm"]] += 1
        return dict(distribution)

    def _calculate_error_distribution(self) -> Dict[str, int]:
        """Calculate distribution of error types"""
        distribution = defaultdict(int)
        for error in self.insights_data["errors"]:
            distribution[error["error_type"]] += 1
        return dict(distribution)


# --- Health Check Service ---


class HealthCheckService:
    """Service for health checks and readiness probes"""

    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.last_check_results: Dict[str, Dict[str, Any]] = {}

    def register_check(self, name: str, check_func: Callable):
        """Register a health check function"""
        self.checks[name] = check_func

    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks"""
        results = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {},
        }

        for name, check_func in self.checks.items():
            try:
                start_time = time.time()

                # Run check (support both sync and async)
                if asyncio.iscoroutinefunction(check_func):
                    check_result = await check_func()
                else:
                    check_result = check_func()

                duration = time.time() - start_time

                results["checks"][name] = {
                    "status": "healthy",
                    "duration_ms": round(duration * 1000, 2),
                    "details": check_result,
                }

            except Exception as e:
                results["checks"][name] = {"status": "unhealthy", "error": str(e)}
                results["status"] = "unhealthy"

        self.last_check_results = results
        return results

    async def get_readiness(self) -> Dict[str, Any]:
        """Check if service is ready to accept traffic"""
        critical_checks = ["database", "redis", "auth_service", "llm"]
        health_results = await self.run_health_checks()

        # Check critical services
        for check in critical_checks:
            if check in health_results["checks"]:
                if health_results["checks"][check]["status"] != "healthy":
                    return {
                        "ready": False,
                        "reason": f"Critical service {check} is unhealthy",
                        "checks": health_results["checks"],
                    }

        return {"ready": True, "checks": health_results["checks"]}


# --- Monitoring Middleware ---


class MonitoringMiddleware:
    """FastAPI middleware for request monitoring"""

    def __init__(
        self,
        insights: ApplicationInsights,
        otel_service: Optional[OpenTelemetryService] = None,
    ):
        self.insights = insights
        self.otel = otel_service

    async def __call__(self, request, call_next):
        """Process request with monitoring"""
        start_time = time.time()

        # Create span for tracing (use context-managed helper to preserve context)
        span = None
        span_cm = None
        if self.otel:
            span_cm = self.otel.create_span(
                f"{request.method} {request.url.path}",
                {
                    "http.method": request.method,
                    "http.url": str(request.url),
                    "http.scheme": request.url.scheme,
                    "http.host": request.url.hostname,
                    "http.target": request.url.path,
                    "user_agent": request.headers.get("user-agent", ""),
                },
            )
            try:
                span = span_cm.__enter__()
            except Exception:
                span = None
                span_cm = None

        try:
            # Process request
            response = await call_next(request)
            duration = time.time() - start_time

            # Track metrics
            await self.insights.track_api_request(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code,
                duration=duration,
            )

            # Update span
            if span:
                span.set_attribute("http.status_code", response.status_code)
                span.set_status(trace.Status(trace.StatusCode.OK))

            # Add monitoring headers
            response.headers["X-Request-ID"] = (
                request.state.request_id
                if hasattr(request.state, "request_id")
                else "unknown"
            )
            response.headers["X-Response-Time"] = f"{duration:.3f}"

            return response

        except Exception as e:
            duration = time.time() - start_time

            # Track error
            await self.insights.track_error(
                error_type="request_processing",
                severity="error",
                details={
                    "method": request.method,
                    "endpoint": request.url.path,
                    "error": str(e),
                    "duration": duration,
                },
            )

            # Update span with error
            if span:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.record_exception(e)

            raise

        finally:
            if 'span_cm' in locals() and span_cm:
                try:
                    span_cm.__exit__(None, None, None)
                except Exception:
                    pass


# --- Global Monitoring Instance ---


def create_monitoring_stack(config: MonitoringConfig) -> Dict[str, Any]:
    """Create complete monitoring stack"""
    # Create Prometheus metrics
    prometheus_metrics = PrometheusMetrics()

    # Create OpenTelemetry service
    otel_service = OpenTelemetryService(config) if config.enable_opentelemetry else None

    # Create performance monitor
    performance_monitor = PerformanceMonitor()

    # Create application insights
    app_insights = ApplicationInsights(prometheus_metrics)

    # Create health check service
    health_service = HealthCheckService()

    # Register default health checks
    health_service.register_check(
        "system",
        lambda: {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
        },
    )

    return {
        "prometheus": prometheus_metrics,
        "opentelemetry": otel_service,
        "performance": performance_monitor,
        "insights": app_insights,
        "health": health_service,
        "middleware": MonitoringMiddleware(app_insights, otel_service),
    }


# --- Middleware Factory Helper ---


def create_monitoring_middleware(
    prometheus_metrics: "PrometheusMetrics",  # noqa: F821 – forward reference
    app_insights: "ApplicationInsights",  # noqa: F821 – forward reference
    otel_service: Optional[
        "OpenTelemetryService"
    ] = None,  # noqa: F821 – forward reference
) -> MonitoringMiddleware:
    """Factory helper that returns an instance of ``MonitoringMiddleware``.

    The existing codebase occasionally expects a convenience wrapper for
    constructing the monitoring middleware instead of instantiating
    ``MonitoringMiddleware`` directly.  The underlying middleware only requires
    ``ApplicationInsights`` (and optionally an ``OpenTelemetryService``) so we
    merely forward the *insights* argument and ignore ``prometheus_metrics``
    which is already embedded inside the ``ApplicationInsights`` instance.

    Parameters
    ----------
    prometheus_metrics : PrometheusMetrics
        Prometheus metrics registry instance.  It is kept here to preserve the
        historical call-signature used throughout the project, but is not
        required for the middleware itself.
    app_insights : ApplicationInsights
        The application insights object used by the middleware to track
        request and error information.
    otel_service : OpenTelemetryService, optional
        Optional OpenTelemetry service for distributed tracing.

    Returns
    -------
    MonitoringMiddleware
        A fully-initialised monitoring middleware ready to be added to a
        FastAPI application via ``app.add_middleware``.
    """

    # Only *app_insights* is necessary for the middleware; *prometheus_metrics*
    # is accepted to maintain backwards compatibility with existing call sites.
    return MonitoringMiddleware(app_insights, otel_service)


# --- Singleton Monitoring Service for Enhanced Features ---

class EnhancedMonitoringService:
    """Monitoring service for enhanced features (self-healing, ML pipeline)"""
    
    def __init__(self):
        self.paradigm_metrics = defaultdict(lambda: {"queries": 0, "switches": 0, "satisfaction": deque(maxlen=1000)})
        self.ml_metrics = {"training_runs": 0, "model_updates": 0, "accuracy_history": deque(maxlen=1000)}
        
    async def track_paradigm_performance(self, paradigm: str, performance_record: Dict[str, Any]):
        """Track paradigm performance for self-healing system"""
        self.paradigm_metrics[paradigm]["queries"] += 1
        if performance_record.get("user_feedback"):
            self.paradigm_metrics[paradigm]["satisfaction"].append(
                performance_record["user_feedback"]
            )
    
    async def track_paradigm_switch(self, switch_decision: Dict[str, Any]):
        """Track paradigm switch events"""
        original = switch_decision.get("original_paradigm")
        if original:
            self.paradigm_metrics[original]["switches"] += 1
    
    async def track_ml_training(self, training_result: Dict[str, Any]):
        """Track ML pipeline training events"""
        self.ml_metrics["training_runs"] += 1
        if training_result.get("accuracy"):
            self.ml_metrics["accuracy_history"].append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "accuracy": training_result["accuracy"]
            })
    
    async def track_error(self, error_type: str, severity: str, details: Dict[str, Any]):
        """Track errors from enhanced features"""
        logger.bind(
            error_type=error_type,
            severity=severity,
            **details
        ).error(f"Enhanced feature error: {error_type}")


# Create singleton instance
monitoring_service = EnhancedMonitoringService()
