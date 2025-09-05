"""
Research-related Pydantic models
"""

from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field

from models.base import (
    ResearchDepth,
    ResearchStatus,
    Paradigm,
    SourceResult,
    ParadigmClassification
)


class ResearchOptions(BaseModel):
    depth: ResearchDepth = ResearchDepth.STANDARD
    paradigm_override: Optional[Paradigm] = None
    include_secondary: bool = True
    max_sources: int = Field(default=50, ge=10, le=200)
    language: str = "en"
    region: str = "us"
    enable_real_search: bool = True
    # Optional feature flag expected by frontend
    enable_ai_classification: Optional[bool] = False
    # Optional deep-research tuning parameters (accepted but optional)
    search_context_size: Optional[str] = Field(
        default=None, pattern="^(small|medium|large)$"
    )
    user_location: Optional[Dict[str, str]] = None


class ResearchQuery(BaseModel):
    query: str = Field(..., min_length=10, max_length=500)
    options: ResearchOptions = ResearchOptions()


class ResearchResult(BaseModel):
    research_id: str
    query: str
    status: ResearchStatus
    paradigm_analysis: Dict[str, Any]
    answer: Dict[str, Any]
    sources: List[SourceResult]
    metadata: Dict[str, Any]
    cost_info: Optional[Dict[str, float]] = None


class ParadigmOverrideRequest(BaseModel):
    """Request model to force a different paradigm for research job"""
    research_id: str
    paradigm: Paradigm
    reason: Optional[str] = None


class ClassifyRequest(BaseModel):
    query: str


class ResearchDeepQuery(BaseModel):
    query: str = Field(..., min_length=10, max_length=500)
    paradigm: Optional[Paradigm] = None
    search_context_size: Optional[str] = Field(
        default="medium", pattern="^(small|medium|large)$"
    )
    user_location: Optional[Dict[str, str]] = None
