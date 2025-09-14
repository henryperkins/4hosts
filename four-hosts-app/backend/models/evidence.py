"""
Canonical Evidence models used across context → orchestrator → synthesis.

These Pydantic models replace ad-hoc dicts and dataclasses so that all
layers share one contract. The bundle is carried on SynthesisContext and
serialized in results metadata for UI/exports.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, HttpUrl


class EvidenceQuote(BaseModel):
    id: str = Field(..., description="Stable ID used in prompts and UI")
    url: HttpUrl | str
    title: str = ""
    domain: str = ""
    quote: str
    start: int | None = Field(default=None, description="Start offset in source text if known")
    end: int | None = Field(default=None, description="End offset in source text if known")
    published_date: Optional[str] = None
    credibility_score: Optional[float] = None
    suspicious: Optional[bool] = None
    doc_summary: Optional[str] = None
    source_type: Optional[str] = None


class EvidenceMatch(BaseModel):
    domain: str
    fragments: List[str] = []


class EvidenceBundle(BaseModel):
    """Unified evidence payload produced by CE and enriched by the orchestrator."""

    quotes: List[EvidenceQuote] = []
    matches: List[EvidenceMatch] = []
    by_domain: Dict[str, int] = {}
    focus_areas: List[str] = []

    class Config:
        json_schema_extra = {
            "example": {
                "quotes": [
                    {
                        "id": "q001",
                        "url": "https://example.com/post",
                        "title": "Example",
                        "domain": "example.com",
                        "quote": "Key sentence used as evidence.",
                        "credibility_score": 0.82
                    }
                ],
                "matches": [
                    {"domain": "example.com", "fragments": ["theme-aligned fragment here"]}
                ],
                "by_domain": {"example.com": 2},
                "focus_areas": ["market size", "regulatory risk"]
            }
        }

