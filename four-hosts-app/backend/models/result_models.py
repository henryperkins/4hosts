"""
API response contracts for research results.
Lightweight and backward-compatible (allows extra fields).
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from models.base import ResearchStatus, Paradigm, SourceResult


class AnswerSection(BaseModel):
    title: str = ""
    paradigm: Paradigm
    content: str = ""
    confidence: float = 0.0
    sources_count: int = 0
    citations: List[Any] = []
    key_insights: List[str] = []

    class Config:
        extra = "allow"


class AnswerPayload(BaseModel):
    summary: str = ""
    sections: List[AnswerSection] = []
    action_items: List[Dict[str, Any]] = []
    citations: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}

    class Config:
        extra = "allow"


class ParadigmSummary(BaseModel):
    paradigm: Paradigm
    confidence: float = 0.0
    approach: Optional[str] = None
    focus: Optional[str] = None

    class Config:
        extra = "allow"


class ParadigmAnalysis(BaseModel):
    primary: ParadigmSummary
    secondary: Optional[ParadigmSummary] = None

    class Config:
        extra = "allow"


class ResearchFinalResult(BaseModel):
    research_id: str
    query: str
    status: ResearchStatus
    paradigm_analysis: ParadigmAnalysis
    answer: AnswerPayload
    integrated_synthesis: Optional[Dict[str, Any]] = None
    mesh_synthesis: Optional[Dict[str, Any]] = None
    sources: List[SourceResult] = []
    metadata: Dict[str, Any] = {}
    cost_info: Dict[str, Any] = {}
    export_formats: Dict[str, str] = {}

    class Config:
        extra = "allow"

