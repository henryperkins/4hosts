"""
Context Models for Four Hosts Research Application
Defines data structures for context engineering and search results
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

# Import HostParadigm from classification_engine
from services.classification_engine import HostParadigm, QueryFeatures as QueryFeaturesBase


class QueryFeaturesSchema(BaseModel):
    """Schema for query features used in classification"""
    text: str
    tokens: List[str] = []
    entities: List[str] = []
    intent_signals: List[str] = []
    domain: Optional[str] = None
    urgency_score: float = 0.0
    complexity_score: float = 0.0
    emotional_valence: float = 0.0

    class Config:
        json_schema_extra = {
            "example": {
                "text": "What are the latest breakthroughs in quantum computing?",
                "tokens": ["latest", "breakthroughs", "quantum", "computing"],
                "entities": ["quantum computing"],
                "intent_signals": ["research", "discovery"],
                "domain": "technology",
                "urgency_score": 0.3,
                "complexity_score": 0.8,
                "emotional_valence": 0.1
            }
        }


class ClassificationResultSchema(BaseModel):
    """Schema for paradigm classification results"""
    query: str
    primary_paradigm: HostParadigm
    secondary_paradigm: Optional[HostParadigm] = None
    distribution: Dict[HostParadigm, float]
    confidence: float
    features: QueryFeaturesSchema
    reasoning: Dict[HostParadigm, List[str]]
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "query": "How can we expose corporate corruption?",
                "primary_paradigm": "revolutionary",
                "secondary_paradigm": "analytical",
                "distribution": {
                    "revolutionary": 0.7,
                    "analytical": 0.2,
                    "strategic": 0.08,
                    "devotion": 0.02
                },
                "confidence": 0.85,
                "features": {
                    "text": "How can we expose corporate corruption?",
                    "tokens": ["expose", "corporate", "corruption"],
                    "entities": ["corporate corruption"],
                    "intent_signals": ["expose", "reveal"],
                    "domain": "social_justice",
                    "urgency_score": 0.7,
                    "complexity_score": 0.6,
                    "emotional_valence": -0.4
                },
                "reasoning": {
                    "revolutionary": ["Contains 'expose' keyword", "Focus on revealing corruption"],
                    "analytical": ["Seeks investigative approach"],
                    "strategic": ["Implies need for methodology"],
                    "devotion": ["No community care signals"]
                },
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class ContextEngineeredQuerySchema(BaseModel):
    """Schema for context-engineered query results"""
    original_query: str
    refined_query: str
    paradigm: HostParadigm
    enhancements: List[str]
    context_additions: List[str]
    focus_areas: List[str]
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "original_query": "climate change solutions",
                "refined_query": "innovative technological solutions for climate change mitigation and adaptation strategies",
                "paradigm": "analytical",
                "enhancements": ["Added specificity", "Included both mitigation and adaptation"],
                "context_additions": ["technological", "strategies"],
                "focus_areas": ["innovation", "mitigation", "adaptation"],
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class UserContextSchema(BaseModel):
    """Schema for user context information"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    query_history: List[str] = []
    paradigm_preferences: Dict[HostParadigm, float] = {}
    domain_interests: List[str] = []
    location: Optional[str] = None
    language: str = "en"
    device_type: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "session_id": "session456",
                "query_history": ["AI ethics", "machine learning bias"],
                "paradigm_preferences": {
                    "analytical": 0.6,
                    "revolutionary": 0.3,
                    "strategic": 0.1,
                    "devotion": 0.0
                },
                "domain_interests": ["technology", "ethics", "ai"],
                "location": "San Francisco, CA",
                "language": "en",
                "device_type": "desktop",
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class SearchResultSchema(BaseModel):
    """Schema for search results"""
    title: str
    url: str
    snippet: str
    source: str
    timestamp: Optional[datetime] = None
    credibility_score: Optional[float] = None
    relevance_score: float
    paradigm_alignment: Dict[HostParadigm, float] = {}
    metadata: Dict[str, Any] = {}

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Breaking: New Evidence in Corporate Scandal",
                "url": "https://example.com/article",
                "snippet": "Recent investigations have uncovered...",
                "source": "investigative-news.com",
                "timestamp": "2024-01-15T10:30:00",
                "credibility_score": 0.85,
                "relevance_score": 0.92,
                "paradigm_alignment": {
                    "revolutionary": 0.8,
                    "analytical": 0.15,
                    "strategic": 0.05,
                    "devotion": 0.0
                },
                "metadata": {
                    "author": "Jane Doe",
                    "publication_date": "2024-01-15",
                    "content_type": "article"
                }
            }
        }