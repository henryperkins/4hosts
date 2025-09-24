"""
System routes for SSOTA telemetry and limits
"""

from typing import Any, Dict, List
import json
from collections import defaultdict
from datetime import timedelta

import structlog
# pylint: disable=import-error

from fastapi import APIRouter, Request

from core.limits import API_RATE_LIMITS, WS_RATE_LIMITS
from services.context_engineering import context_pipeline
from services.cache import cache_manager
from services.research_store import research_store
from services.llm_client import llm_client
from services.token_manager import token_manager
from models.base import ResearchStatus, UserRole
from utils.type_coercion import as_int
from utils.date_utils import safe_parse_date, get_current_utc

router = APIRouter(prefix="/system", tags=["system"])
logger = structlog.get_logger(__name__)


@router.get("/context-metrics")
async def get_context_metrics() -> Dict[str, Any]:
    try:
        return {"context_pipeline": context_pipeline.get_pipeline_metrics()}
    except Exception as e:
        logger.error("Failed to collect context metrics: %s", e)
        return {"context_pipeline": {}}


def _serialise_api_limits() -> Dict[str, Dict[str, Any]]:
    plans: Dict[str, Dict[str, Any]] = {}
    for role, limits in API_RATE_LIMITS.items():
        role_name = role.value if isinstance(role, UserRole) else str(role)
        plans[role_name.lower()] = {
            "requests_per_minute": limits.get("requests_per_minute"),
            "requests_per_hour": limits.get("requests_per_hour"),
            "requests_per_day": limits.get("requests_per_day"),
            "concurrent_requests": limits.get("concurrent_requests"),
            "max_query_length": limits.get("max_query_length"),
            "max_sources": limits.get("max_sources"),
        }
    return plans


def _serialise_ws_limits() -> Dict[str, Dict[str, Any]]:
    plans: Dict[str, Dict[str, Any]] = {}
    for role, limits in WS_RATE_LIMITS.items():
        role_name = role.value if isinstance(role, UserRole) else str(role)
        plans[role_name.lower()] = dict(limits)
    return plans


@router.get("/limits")
async def get_limits() -> Dict[str, Any]:
    return {
        "plans": _serialise_api_limits(),
        "realtime": _serialise_ws_limits(),
    }


@router.get("/llm-ping")
async def llm_ping() -> Dict[str, str]:
    """
    Health-check endpoint: verifies LLM connectivity by issuing a minimal
    completion request (“ping”) and returning its trimmed response.
    """
    try:
        resp = await llm_client.generate_completion(
            prompt="ping",
            paradigm="bernard",
        )
        return {"status": "ok", "llm_response": resp.strip()}
    except Exception as e:
        logger.warning("LLM ping failed: %s", e)
        return {"status": "error", "detail": str(e)}


async def _collect_research_records(limit: int | None = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    try:
        if research_store.use_redis and research_store.redis_client:
            pattern = f"{research_store.key_prefix}*"
            async for key in research_store.redis_client.scan_iter(pattern):
                try:
                    data = await research_store.redis_client.get(key)
                    if not data:
                        continue
                    rec = json.loads(data)
                    records.append(rec)
                    if limit and len(records) >= limit:
                        break
                except Exception:
                    continue
        else:
            # Fallback in-memory store
            for rec in research_store.fallback_store.values():
                records.append(rec)
                if limit and len(records) >= limit:
                    break
    except Exception:
        pass
    return records


def _summarize_paradigm(rec: Dict[str, Any]) -> str | None:
    try:
        pc = rec.get("paradigm_classification") or {}
        prim = pc.get("primary")
        if isinstance(prim, dict):
            # Sometimes stored as object
            return prim.get("primary") or prim.get("paradigm")
        if isinstance(prim, str):
            return prim
        # Try final result
        res = rec.get("results") or {}
        pa = res.get("paradigm_analysis") or {}
        prim2 = (pa.get("primary") or {}).get("paradigm")
        return prim2
    except Exception:
        return None


@router.get("/stats")
async def get_system_stats(request: Request) -> Dict[str, Any]:
    try:
        # Research aggregates
        records = await _collect_research_records()
        total = len(records)
        active = 0
        dist: Dict[str, int] = {}
        proc_times = []
        for r in records:
            try:
                status = r.get("status")
                if status in [ResearchStatus.PROCESSING, ResearchStatus.IN_PROGRESS]:
                    active += 1
                # Paradigm distribution
                p = _summarize_paradigm(r)
                if p:
                    dist[p] = dist.get(p, 0) + 1
                # Processing time from completed results
                res = r.get("results") or {}
                meta = res.get("metadata") or {}
                pt = meta.get("processing_time_seconds")
                if isinstance(pt, (int, float)) and pt > 0:
                    proc_times.append(float(pt))
            except Exception:
                continue

        avg_time = round(sum(proc_times) / len(proc_times), 2) if proc_times else 0.0

        # Cache stats
        cache_stats = await cache_manager.get_cache_stats()
        hit_rate = float(cache_stats.get("hit_rate_percent", 0.0))

        rate_limiter = getattr(request.app.state, "rate_limiter", None)
        redis_components = {
            "cache": "redis" if cache_manager.redis_pool else "memory",
            "research_store": (
                "redis"
                if research_store.use_redis
                else f"fallback:{research_store.use_redis_reason}"
            ),
            "rate_limiter": "redis"
            if getattr(rate_limiter, "redis_enabled", False)
            else "memory",
            "token_manager": "redis"
            if getattr(token_manager, "redis_enabled", False)
            else "memory",
        }

        # Health status via app.state
        health_service = getattr(getattr(request.app.state, "monitoring", {}), "get", lambda k, d=None: d)("health")
        health = "healthy"
        try:
            if health_service:
                result = await health_service.run_health_checks()
                health = "healthy" if result.get("status") == "healthy" else "degraded"
        except Exception:
            health = "degraded"

        return {
            "total_queries": total,
            "active_research": active,
            "paradigm_distribution": dist,
            "average_processing_time": avg_time,
            "cache_hit_rate": hit_rate,
            "cache_hits": int(cache_stats.get("hit_count", 0)),
            "cache_misses": int(cache_stats.get("miss_count", 0)),
            "system_health": health,
            "redis_components": redis_components,
        }
    except Exception as e:
        logger.error("Failed to build system stats: %s", e)
        return {
            "total_queries": 0,
            "active_research": 0,
            "paradigm_distribution": {},
            "average_processing_time": 0,
            "cache_hit_rate": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "system_health": "degraded",
        }


@router.get("/public-stats")
async def get_public_system_stats(request: Request) -> Dict[str, Any]:
    # Public view uses the same core data but exposes only aggregate metrics
    return await get_system_stats(request)


@router.get("/extended-stats")
async def get_extended_stats() -> Dict[str, Any]:
    """Return richer metrics snapshot from in-process metrics facade.

    This is feature-gated by presence of the metrics facade; if unavailable
    returns an empty structure. Intended for internal dashboards.
    """
    try:
        from services.metrics import metrics
        snap = metrics.extended_stats_snapshot()
        return snap
    except Exception as e:
        logger.warning("Extended stats unavailable: %s", e)
        return {
            "latency": {},
            "fallback_rates": {},
            "llm_usage": {},
            "paradigm_distribution": {},
            "quality": {},
            "counters": {},
        }


@router.get("/search-metrics")
async def get_search_metrics(window_minutes: int = 60, limit: int = 720) -> Dict[str, Any]:
    """Return persisted search metrics suitable for dashboards and alerting."""

    window_minutes = max(5, min(window_minutes, 7 * 24 * 60))
    limit = max(25, min(limit, 2000))

    try:
        events = await cache_manager.get_search_metrics_events(limit=limit)
    except Exception as e:
        logger.error("Failed to retrieve search metrics: %s", e)
        events = []

    now = get_current_utc()
    cutoff = now - timedelta(minutes=window_minutes)

    timeline = {}
    provider_usage: Dict[str, int] = defaultdict(int)
    provider_costs: Dict[str, float] = defaultdict(float)
    paradigm_distribution: Dict[str, int] = defaultdict(int)

    total_queries = 0
    total_results = 0
    total_cost = 0.0
    runs = 0
    dedup_sum = 0.0
    dedup_count = 0

    for event in events:
        ts = safe_parse_date(event.get("timestamp"))
        if not ts or ts < cutoff:
            continue

        runs += 1

        tq = as_int(event.get("total_queries"))
        tr = as_int(event.get("total_results"))
        total_queries += tq
        total_results += tr

        dedup = event.get("deduplication_rate")
        if isinstance(dedup, (int, float)):
            dedup_val = float(dedup)
            dedup_sum += dedup_val
            dedup_count += 1

        depth = str(event.get("depth") or "standard").lower()
        paradigm = str(event.get("paradigm") or "unknown").lower()
        composite_key = (ts.replace(second=0, microsecond=0), paradigm, depth)
        bucket = timeline.setdefault(
            composite_key,
            {
                "timestamp": ts.replace(second=0, microsecond=0).isoformat(),
                "runs": 0,
                "total_queries": 0,
                "total_results": 0,
                "dedup_sum": 0.0,
            },
        )
        bucket["runs"] += 1
        bucket["total_queries"] += tq
        bucket["total_results"] += tr
        if isinstance(dedup, (int, float)):
            bucket["dedup_sum"] += float(dedup)

        paradigm_distribution[paradigm] += 1

        for provider in event.get("apis_used", []):
            provider_name = str(provider or "unknown").lower()
            provider_usage[provider_name] += 1

        for provider, cost in (event.get("provider_costs") or {}).items():
            try:
                val = float(cost)
            except Exception:
                continue
            pname = str(provider or "unknown").lower()
            provider_costs[pname] += val
            total_cost += val

    timeline_points = []
    for (_, paradigm, depth), payload in sorted(timeline.items(), key=lambda x: x[0]):
        runs_in_bucket = max(1, payload["runs"])
        timeline_points.append(
            {
                "timestamp": payload["timestamp"],
                "runs": payload["runs"],
                "total_queries": payload["total_queries"],
                "total_results": payload["total_results"],
                "deduplication_rate": round(payload["dedup_sum"] / runs_in_bucket, 4),
                "paradigm": paradigm,
                "depth": depth,
            }
        )

    avg_dedup = round(dedup_sum / dedup_count, 4) if dedup_count else 0.0

    return {
        "window_minutes": window_minutes,
        "runs": runs,
        "total_queries": total_queries,
        "total_results": total_results,
        "avg_deduplication_rate": avg_dedup,
        "total_cost_usd": round(total_cost, 4),
        "provider_usage": dict(provider_usage),
        "provider_costs": {k: round(v, 4) for k, v in provider_costs.items()},
        "paradigm_distribution": dict(paradigm_distribution),
        "timeline": timeline_points,
    }


# _safe_int moved to utils.type_coercion.as_int
