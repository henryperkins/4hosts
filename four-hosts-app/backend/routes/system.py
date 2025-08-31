"""
System routes for SSOTA telemetry and limits
"""

from typing import Any, Dict
import logging

from fastapi import APIRouter

from services.context_engineering import context_pipeline

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

