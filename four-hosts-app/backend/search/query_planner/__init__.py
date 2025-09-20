"""Unified query planner package."""

# Prefer consolidated types from services; fallback to local wrapper for tools.
try:
    from services.query_planning.types import (  # type: ignore
        QueryCandidate,
        PlannerConfig,
        StageName,
    )
except Exception:  # pragma: no cover
    from .types import QueryCandidate, PlannerConfig, StageName

from .planner import QueryPlanner

__all__ = ["QueryCandidate", "PlannerConfig", "StageName", "QueryPlanner"]
