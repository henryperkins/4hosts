"""
Canonical Evidence models used across context → orchestrator → synthesis.

These Pydantic models replace ad-hoc dicts and dataclasses so that all
layers share one contract. The bundle is carried on SynthesisContext and
serialized in results metadata for UI/exports.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
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
    # Optional short context window around the quote for better grounding
    context_window: Optional[str] = None


class EvidenceMatch(BaseModel):
    domain: str
    fragments: List[str] = []


class EvidenceDocument(BaseModel):
    """Full-text document payload for high-token o3 synthesis."""

    id: str = Field(..., description="Stable ID referenced in prompts (e.g. d001)")
    url: HttpUrl | str
    title: str = ""
    domain: str = ""
    content: str = Field(..., description="Full or truncated document content")
    token_count: int = Field(0, description="Approximate tokens contributed to the prompt")
    word_count: int = Field(0, description="Approximate words for analytics")
    truncated: bool = Field(False, description="Whether the content was trimmed to fit budget")
    credibility_score: Optional[float] = None
    published_date: Optional[str] = None
    source_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidenceBundle(BaseModel):
    """Unified evidence payload produced by CE and enriched by the orchestrator."""

    quotes: List[EvidenceQuote] = []
    matches: List[EvidenceMatch] = []
    by_domain: Dict[str, int] = {}
    focus_areas: List[str] = []
    documents: List[EvidenceDocument] = []
    documents_token_count: int = Field(0, description="Total tokens consumed by full-document context")

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
                "focus_areas": ["market size", "regulatory risk"],
                "documents": [
                    {
                        "id": "d001",
                        "url": "https://example.com/full-report",
                        "title": "Example Report",
                        "domain": "example.com",
                        "content": "Full report content trimmed for brevity...",
                        "token_count": 950,
                        "word_count": 3600,
                        "truncated": True,
                        "credibility_score": 0.88
                    }
                ],
                "documents_token_count": 950
            }
        }
