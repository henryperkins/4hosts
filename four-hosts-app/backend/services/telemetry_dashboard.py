"""Helpers for building telemetry dashboard summaries."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List

from services.cache import cache_manager
from utils.date_utils import get_current_iso
from utils.type_coercion import as_float, as_int


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return 0


async def get_recent_events(limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch recent telemetry events from cache with sensible bounds."""

    if limit <= 0:
        limit = 1
    limit = min(limit, 500)
    events = await cache_manager.get_search_metrics_events(limit)
    return events or []


async def build_summary(limit: int = 50) -> Dict[str, Any]:
    """Aggregate telemetry events into a dashboard-friendly summary."""

    events = await get_recent_events(limit)
    if not events:
        return {
            "runs": 0,
            "updated_at": get_current_iso(),
            "recent_events": [],
            "totals": {},
            "paradigms": {},
            "depths": {},
            "providers": {},
            "coverage": {},
            "agent_loop": {},
            "stages": {},
        }

    runs = len(events)
    total_processing = 0.0
    total_queries = 0
    total_results = 0
    provider_costs: Dict[str, float] = defaultdict(float)
    provider_usage: Dict[str, int] = defaultdict(int)
    stage_breakdown: Dict[str, int] = defaultdict(int)
    paradigms = Counter()
    depths = Counter()
    coverage_values = []
    evidence_quotes = []
    evidence_docs = []
    agent_iterations = []
    agent_new_queries = []

    for event in events:
        total_processing += _safe_float(event.get("processing_time_seconds"))
        total_queries += _safe_int(event.get("total_queries"))
        total_results += _safe_int(event.get("total_results"))

        for provider, cost in (event.get("provider_costs") or {}).items():
            provider_costs[str(provider)] += _safe_float(cost)
        for provider in event.get("apis_used") or []:
            provider_usage[str(provider)] += 1

        for stage, count in (event.get("stage_breakdown") or {}).items():
            stage_breakdown[str(stage)] += _safe_int(count)

        paradigms[str(event.get("paradigm", "unknown"))] += 1
        depths[str(event.get("depth", "standard"))] += 1

        if event.get("grounding_coverage") is not None:
            coverage_values.append(_safe_float(event.get("grounding_coverage")))
        if event.get("evidence_quotes_count") is not None:
            evidence_quotes.append(_safe_int(event.get("evidence_quotes_count")))
        if event.get("evidence_documents_count") is not None:
            evidence_docs.append(_safe_int(event.get("evidence_documents_count")))

        if event.get("agent_iterations") is not None:
            agent_iterations.append(_safe_int(event.get("agent_iterations")))
        if event.get("agent_new_queries") is not None:
            agent_new_queries.append(_safe_int(event.get("agent_new_queries")))

    totals = {
        "runs": runs,
        "avg_processing_time_seconds": round(total_processing / runs, 2),
        "avg_total_queries": round(total_queries / runs, 2),
        "avg_total_results": round(total_results / runs, 2),
    }

    providers = {
        "costs": {k: round(v, 4) for k, v in sorted(provider_costs.items())},
        "usage": dict(sorted(provider_usage.items(), key=lambda item: item[0])),
    }

    coverage = {
        "avg_grounding": round(sum(coverage_values) / len(coverage_values), 3)
        if coverage_values
        else 0.0,
        "avg_evidence_quotes": round(sum(evidence_quotes) / len(evidence_quotes), 2)
        if evidence_quotes
        else 0.0,
        "avg_evidence_documents": round(sum(evidence_docs) / len(evidence_docs), 2)
        if evidence_docs
        else 0.0,
    }

    agent_loop = {
        "avg_iterations": round(sum(agent_iterations) / len(agent_iterations), 2)
        if agent_iterations
        else 0.0,
        "avg_new_queries": round(sum(agent_new_queries) / len(agent_new_queries), 2)
        if agent_new_queries
        else 0.0,
    }

    stages = dict(sorted(stage_breakdown.items(), key=lambda item: item[0]))

    recent = events[: min(len(events), 5)]

    return {
        "runs": runs,
        "updated_at": get_current_iso(),
        "totals": totals,
        "paradigms": dict(paradigms),
        "depths": dict(depths),
        "providers": providers,
        "coverage": coverage,
        "agent_loop": agent_loop,
        "stages": stages,
        "recent_events": recent,
    }


__all__ = ["build_summary", "get_recent_events"]
