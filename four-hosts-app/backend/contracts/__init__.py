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


__all__ = [
    "ResearchStatus",
    "Source",
    "SearchResult",
    "ResearchBundle",
    "GeneratedAnswer",
]
