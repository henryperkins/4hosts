"""Lightweight Metrics Facade

Provides in-process instrumentation without forcing every caller to import
Prometheus/OpenTelemetry directly. Designed to be safe under failure:
all public methods swallow exceptions so instrumentation never breaks
the research pipeline.

Features (initial slice):
 - Stage event recording with bounded ring buffer (FIFO) for recent events
 - Percentile calculation (p50/p95/p99) over recent durations per stage
 - Simple counters (name -> int) with optional label tuples (capped)
 - Aggregation helper for system /extended-stats endpoint
 - Minimal token accounting (in/out) & fallback tracking

Future extensions can layer Prometheus export by iterating over internal
structures. We intentionally avoid external deps here; the existing
monitoring stack can scrape if desired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple, Any
from collections import deque, defaultdict
import time
import threading

MAX_EVENTS = 5000  # Ring buffer upper bound
MAX_LABEL_CARDINALITY = 50  # Prevent unbounded growth

_lock = threading.RLock()


@dataclass
class StageEvent:
    ts: float
    stage: str
    paradigm: Optional[str]
    duration_ms: float
    success: bool
    fallback: bool
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    model: Optional[str] = None
    prompt_version: Optional[str] = None


@dataclass
class O3UsageEvent:
    ts: float
    paradigm: Optional[str]
    document_count: int
    document_tokens: int
    quote_count: int
    source_count: int
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None



class MetricsFacade:
    def __init__(self):
        self._events: Deque[StageEvent] = deque(maxlen=MAX_EVENTS)
        self._counters: Dict[str, Dict[Tuple[str, ...], int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._label_values: Dict[str, set] = defaultdict(set)
        self._o3_usage: Deque[O3UsageEvent] = deque(maxlen=MAX_EVENTS)

    def record_stage(
        self,
        stage: str,
        duration_ms: float,
        paradigm: Optional[str] = None,
        success: bool = True,
        fallback: bool = False,
        tokens_in: Optional[int] = None,
        tokens_out: Optional[int] = None,
        model: Optional[str] = None,
        prompt_version: Optional[str] = None,
    ) -> None:
        try:
            if duration_ms < 0:
                duration_ms = 0
            with _lock:
                self._events.append(
                    StageEvent(
                        ts=time.time(),
                        stage=stage,
                        paradigm=paradigm,
                        duration_ms=duration_ms,
                        success=success,
                        fallback=fallback,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        model=model,
                        prompt_version=prompt_version,
                    )
                )
        except Exception:
            pass

    def increment(
        self,
        name: str,
        *label_values: str,
        amount: int = 1,
    ) -> None:
        try:
            with _lock:
                if label_values:
                    for v in label_values:
                        if (
                            len(self._label_values[name])
                            < MAX_LABEL_CARDINALITY
                        ):
                            self._label_values[name].add(v)
                        elif v not in self._label_values[name]:
                            return
                key = tuple(label_values) if label_values else tuple()
                self._counters[name][key] += amount
        except Exception:
            pass

    def record_o3_usage(
        self,
        *,
        paradigm: Optional[str],
        document_count: int,
        document_tokens: int,
        quote_count: int,
        source_count: int,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
    ) -> None:
        try:
            with _lock:
                self._o3_usage.append(
                    O3UsageEvent(
                        ts=time.time(),
                        paradigm=paradigm,
                        document_count=max(0, int(document_count)),
                        document_tokens=max(0, int(document_tokens)),
                        quote_count=max(0, int(quote_count)),
                        source_count=max(0, int(source_count)),
                        prompt_tokens=prompt_tokens if prompt_tokens is None else max(0, int(prompt_tokens)),
                        completion_tokens=completion_tokens if completion_tokens is None else max(0, int(completion_tokens)),
                    )
                )
        except Exception:
            pass

    def _percentiles(self, values: List[float]) -> Dict[str, float]:
        if not values:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        values_sorted = sorted(values)
        
        def pct(p: float) -> float:
            if not values_sorted:
                return 0.0
            k = (len(values_sorted) - 1) * (p / 100.0)
            f = int(k)
            c = min(f + 1, len(values_sorted) - 1)
            if f == c:
                return float(values_sorted[f])
            return float(
                values_sorted[f]
                + (values_sorted[c] - values_sorted[f]) * (k - f)
            )

        return {
            "p50": round(pct(50), 2),
            "p95": round(pct(95), 2),
            "p99": round(pct(99), 2),
        }

    def get_latency_distributions(self) -> Dict[str, Dict[str, float]]:
        by_stage: Dict[str, List[float]] = defaultdict(list)
        with _lock:
            for ev in self._events:
                by_stage[ev.stage].append(ev.duration_ms)
        return {
            stage: self._percentiles(vals)
            for stage, vals in by_stage.items()
        }

    def get_fallback_rates(self) -> Dict[str, float]:
        counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        with _lock:
            for ev in self._events:
                counts[ev.stage][1] += 1
                if ev.fallback:
                    counts[ev.stage][0] += 1
        return {
            stage: round((fb[0] / fb[1]) if fb[1] else 0.0, 4)
            for stage, fb in counts.items()
        }

    def get_llm_usage(self) -> Dict[str, Any]:
        usage: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"calls": 0, "tokens_in": 0, "tokens_out": 0}
        )
        with _lock:
            for ev in self._events:
                if ev.model:
                    u = usage[ev.model]
                    u["calls"] += 1
                    if ev.tokens_in:
                        u["tokens_in"] += ev.tokens_in
                    if ev.tokens_out:
                        u["tokens_out"] += ev.tokens_out
        return usage

    def get_o3_usage_summary(self) -> Dict[str, Any]:
        with _lock:
            events = list(self._o3_usage)
        if not events:
            return {}

        total_docs = sum(ev.document_count for ev in events)
        total_doc_tokens = sum(ev.document_tokens for ev in events)
        total_quotes = sum(ev.quote_count for ev in events)
        total_sources = sum(ev.source_count for ev in events)
        total_prompt_tokens = sum(ev.prompt_tokens or 0 for ev in events)
        total_completion_tokens = sum(ev.completion_tokens or 0 for ev in events)

        paradigms = defaultdict(int)
        for ev in events:
            if ev.paradigm:
                paradigms[ev.paradigm] += 1

        count = len(events)

        def _avg(value: int) -> float:
            return round((value / count), 2) if count else 0.0

        last_ts = events[-1].ts if events else time.time()

        return {
            "events": count,
            "avg_documents": _avg(total_docs),
            "avg_document_tokens": _avg(total_doc_tokens),
            "avg_quotes": _avg(total_quotes),
            "avg_sources": _avg(total_sources),
            "total_document_tokens": total_doc_tokens,
            "total_prompt_tokens": total_prompt_tokens or None,
            "total_completion_tokens": total_completion_tokens or None,
            "paradigm_counts": dict(paradigms),
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(last_ts)),
        }

    def get_paradigm_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = defaultdict(int)
        with _lock:
            for ev in self._events:
                if ev.paradigm:
                    dist[ev.paradigm] += 1
        return dict(dist)

    def get_quality_metrics(self) -> Dict[str, float]:
        return {
            "critic_avg_score": 0.0,
            "hallucination_rate": 0.0,
            "evidence_coverage_ratio": 0.0,
        }

    def get_counters(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        with _lock:
            for name, bucket in self._counters.items():
                out[name] = {"|".join(lbl): val for lbl, val in bucket.items()}
        return out

    def extended_stats_snapshot(self) -> Dict[str, Any]:
        return {
            "latency": self.get_latency_distributions(),
            "fallback_rates": self.get_fallback_rates(),
            "llm_usage": self.get_llm_usage(),
            "paradigm_distribution": self.get_paradigm_distribution(),
            "quality": self.get_quality_metrics(),
            "counters": self.get_counters(),
            "o3_usage": self.get_o3_usage_summary(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


metrics = MetricsFacade()
