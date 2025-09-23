from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

StageName = Literal["rule_based", "paradigm", "llm", "agentic", "context"]


@dataclass(slots=True)
class QueryCandidate:
    """Represents a single search query candidate emitted by the planner."""

    query: str
    stage: StageName
    label: str
    weight: float = 1.0
    source_filter: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class PlannerConfig:
    """Configuration knobs shared across planner integrations."""

    max_candidates: int = 12
    enable_llm: bool = False
    enable_agentic: bool = True
    stage_order: List[StageName] = field(
        default_factory=lambda: ["rule_based", "paradigm", "llm"]
    )
    per_stage_caps: Dict[StageName, int] = field(
        default_factory=lambda: {
            "rule_based": 6,
            "paradigm": 6,
            "llm": 4,
            "agentic": 6,
            "context": 6,
        }
    )
    dedup_jaccard: float = 0.92

    # Relative importance of each stage when ranking.  This was previously
    # a hard-coded constant inside `planner.py`; exposing it here makes the
    # weights configurable at runtime and simplifies tuning experiments.
    stage_prior: Dict[StageName, float] = field(
        default_factory=lambda: {
            "paradigm": 1.0,
            "rule_based": 0.96,
            "llm": 0.9,
            "context": 0.88,
            "agentic": 0.86,
        }
    )
