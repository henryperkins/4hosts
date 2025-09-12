from __future__ import annotations

from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pydantic import BaseModel, Field


class ClassificationOriginal(BaseModel):
    primary: str = Field(..., description="Primary paradigm at time of classification")
    secondary: Optional[str] = Field(None, description="Secondary paradigm if any")
    distribution: Dict[str, float] = Field(
        default_factory=dict,
        description="Paradigm probability distribution (e.g., {'bernard': 0.6, ...})",
    )
    confidence: Optional[float] = Field(
        None, description="Overall classification confidence"
    )


class ClassificationFeedbackRequest(BaseModel):
    research_id: Optional[str] = Field(
        None, description="Associated research id if available"
    )
    query: str = Field(..., description="The original user query")
    original: ClassificationOriginal = Field(
        ..., description="Original classification snapshot"
    )
    user_correction: Optional[str] = Field(
        None,
        description="User's corrected paradigm (e.g., 'maeve'); leave None if user agrees",
    )
    rationale: Optional[str] = Field(
        None, description="Optional free-text rationale for the correction"
    )


class AnswerFeedbackRequest(BaseModel):
    research_id: str = Field(..., description="Associated research id")
    rating: Union[int, float] = Field(
        ..., ge=0, le=1, description="Normalized rating [0,1]"
    )
    reason: Optional[str] = Field(None, description="Optional free-text reason")
    improvements: Optional[List[str]] = Field(
        default=None,
        description="Optional list of short improvement suggestions",
    )
    helpful: Optional[bool] = Field(
        default=None, description="Whether the answer was helpful"
    )


class FeedbackEvent(BaseModel):
    id: str
    user_id: str
    type: str  # 'classification' | 'answer'
    payload: Dict[str, Any]
    timestamp: datetime