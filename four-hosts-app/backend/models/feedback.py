from __future__ import annotations

from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator


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
    query: str = Field(
        ..., description="The original user query", max_length=1000
    )
    original: ClassificationOriginal = Field(
        ..., description="Original classification snapshot"
    )
    user_correction: Optional[str] = Field(
        None,
        description="User's corrected paradigm (e.g., 'maeve'); leave None if user agrees",
        max_length=50,
    )
    rationale: Optional[str] = Field(
        None, description="Optional free-text rationale for the correction", max_length=500
    )


class AnswerFeedbackRequest(BaseModel):
    research_id: str = Field(..., description="Associated research id")
    rating: Union[int, float] = Field(
        ..., ge=0, le=1, description="Normalized rating [0,1]"
    )
    reason: Optional[str] = Field(None, description="Optional free-text reason", max_length=500)
    improvements: Optional[List[str]] = Field(
        default=None,
        description="Optional list of short improvement suggestions (<= 8 items, each <= 500 chars)",
    )
    helpful: Optional[bool] = Field(
        default=None, description="Whether the answer was helpful"
    )

    @validator("improvements")
    def _validate_improvements_list(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return v
        if len(v) > 8:
            raise ValueError("At most 8 improvement suggestions are allowed")
        return v

    @validator("improvements", each_item=True)
    def _validate_improvement_item(cls, v: str) -> str:
        if v is None:
            return v
        if isinstance(v, str) and len(v) > 500:
            raise ValueError("Each improvement suggestion must be <= 500 characters")
        return v


class FeedbackEvent(BaseModel):
    id: str
    user_id: str
    type: str  # 'classification' | 'answer'
    payload: Dict[str, Any]
    timestamp: datetime