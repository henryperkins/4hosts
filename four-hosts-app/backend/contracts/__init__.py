"""contracts
============

Canonical data contracts shared across Four Hosts backend services.

This module intentionally contains **zero** runtime side-effects – it simply
defines Pydantic models / enums that act as typed, version-controlled
interfaces between independent layers (searcher → orchestrator → answer
generator, etc.).

The goal for PR0 is to introduce these objects without changing existing
behaviour.  Down-stream services will gradually migrate to them in subsequent
PR slices.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, HttpUrl, Field

# ---------------------------------------------------------------------------
# Enums / Constants
# ---------------------------------------------------------------------------


class ResearchStatus(str, Enum):
    """Unified status vocabulary for research execution lifecycle."""

    OK = "OK"
    FAILED_NO_SOURCES = "FAILED_NO_SOURCES"
    TOOL_ERROR = "TOOL_ERROR"
    TIMEOUT = "TIMEOUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# ---------------------------------------------------------------------------
# Core transfer objects
# ---------------------------------------------------------------------------


class Source(BaseModel):
    """Minimal representation of a web/document source used across layers."""

    url: HttpUrl
    title: str = Field(..., max_length=512)
    snippet: Optional[str] = Field(None, max_length=2_048)
    score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Search engine response bundled with diagnostics."""

    query: str
    sources: List[Source] = Field(default_factory=list)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


class ResearchBundle(BaseModel):
    """Aggregated search data + context passed into answer generation."""

    query: str
    sources: List[Source]
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


class GeneratedAnswer(BaseModel):
    """Final answer returned by `AnswerGenerator` implementations."""

    status: ResearchStatus
    content_md: str
    citations: List[Source] = Field(default_factory=list)
    quality_score: Optional[float] = None
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# No-op adapter stubs (filled out in later PR slices)
# ---------------------------------------------------------------------------


def to_source(legacy_obj: Any) -> Source:  # type: ignore[valid-type]
    """Temporary helper to cast an existing *legacy* search result object into
    the new `Source` contract.

    This performs best-effort field mapping and should **not** introduce any
    side effects. It will be replaced by dedicated adapter modules once the
    migration reaches PR2/PR3.
    """

    data = {
        "url": getattr(legacy_obj, "url", "http://invalid.local"),
        "title": getattr(legacy_obj, "title", ""),
        "snippet": getattr(legacy_obj, "snippet", None),
        "score": getattr(legacy_obj, "credibility_score", None),
        "metadata": getattr(legacy_obj, "metadata", {}),
    }
    return Source.parse_obj(data)


__all__ = [
    "ResearchStatus",
    "Source",
    "SearchResult",
    "ResearchBundle",
    "GeneratedAnswer",
    "to_source",
]
