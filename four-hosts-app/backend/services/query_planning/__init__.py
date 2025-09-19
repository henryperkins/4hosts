"""Query planning utilities shared across planner integrations."""

from .types import QueryCandidate, PlannerConfig, StageName
from .cleaner import canon_query, jaccard_similarity, is_duplicate
from .variator import (
    RuleBasedStage,
    LLMVariationsStage,
    ParadigmStage,
    ContextStage,
    AgenticFollowupsStage,
)
from .config import build_planner_config
from .optimizer import QueryOptimizer

__all__ = [
    "QueryCandidate",
    "PlannerConfig",
    "StageName",
    "RuleBasedStage",
    "LLMVariationsStage",
    "ParadigmStage",
    "ContextStage",
    "AgenticFollowupsStage",
    "canon_query",
    "jaccard_similarity",
    "is_duplicate",
    "build_planner_config",
    "QueryOptimizer",
]
