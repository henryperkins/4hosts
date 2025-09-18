"""
System routes for SSOTA telemetry and limits
"""

from typing import Any, Dict, List
import logging

from fastapi import APIRouter

from services.context_engineering import context_pipeline
from services.cache import cache_manager
from services.research_store import research_store
from services.llm_client import llm_client
from models.base import ResearchStatus
import json
from fastapi import Request

router = APIRouter(prefix="/system", tags=["system"])
logger = logging.getLogger(__name__)


@router.get("/context-metrics")
async def get_context_metrics() -> Dict[str, Any]:
    try:
        return {"context_pipeline": context_pipeline.get_pipeline_metrics()}
    except Exception as e:
        logger.error("Failed to collect context metrics: %s", e)
        return {"context_pipeline": {}}


@router.get("/limits")
async def get_limits() -> Dict[str, Any]:
    # Static placeholders aligned with SSOTA doc
    return {
        "plans": {
            "free": {"requests_per_hour": 10, "concurrent": 1, "max_sources": 50},
            "basic": {"requests_per_hour": 100, "concurrent": 5, "max_sources": 200},
            "pro": {"requests_per_hour": 1000, "concurrent": 20, "max_sources": 1000},
            "enterprise": {"requests_per_hour": None, "concurrent": None, "max_sources": None},
        }
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
