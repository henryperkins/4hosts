"""
Context Models for Four Hosts Application
Provides Pydantic schemas for consistent serialization across Redis, WebSocket, and API responses
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
import uuid


# Re-export enums for convenience
from services.classification_engine import HostParadigm


class SerializableEnum(str, Enum):
    """Base class for serializable enums"""
    
    def __str__(self):
        return self.value
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v, values=None):
        if isinstance(v, cls):
            return v
        if isinstance(v, str):
            return cls(v)
        raise ValueError(f'Unable to convert {v} to {cls}')


class ResearchDepth(SerializableEnum):
    """Research depth options"""
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class ResearchStatus(SerializableEnum):
    """Research status tracking"""
    PENDING = "pending"
    PROCESSING = "processing"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QueryFeaturesSchema(BaseModel):
    """Schema for query features"""
    text: str
    tokens: List[str]
    entities: List[str]
    intent_signals: List[str]
    domain: Optional[str] = None
    urgency_score: float
    complexity_score: float
    emotional_valence: float


class ClassificationResultSchema(BaseModel):
    """Schema for classification results with proper serialization"""
    model_config = ConfigDict(use_enum_values=False)
    
    query: str
    primary_paradigm: HostParadigm
    secondary_paradigm: Optional[HostParadigm] = None
    distribution: Dict[HostParadigm, float]
    confidence: float
    features: QueryFeaturesSchema
    reasoning: Dict[HostParadigm, List[str]]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    def to_redis_dict(self) -> Dict[str, Any]:
        """Convert to Redis-storable dictionary"""
        data = self.model_dump(mode='json')
        # Ensure enums are stored as values
        data['primary_paradigm'] = self.primary_paradigm.value
        if self.secondary_paradigm:
            data['secondary_paradigm'] = self.secondary_paradigm.value
        # Convert distribution keys to strings
        data['distribution'] = {k.value: v for k, v in self.distribution.items()}
        data['reasoning'] = {k.value: v for k, v in self.reasoning.items()}
        return data
    
    @classmethod
    def from_redis_dict(cls, data: Dict[str, Any]) -> 'ClassificationResultSchema':
        """Reconstruct from Redis dictionary"""
        # Convert string keys back to enums
        data['primary_paradigm'] = HostParadigm(data['primary_paradigm'])
        if data.get('secondary_paradigm'):
            data['secondary_paradigm'] = HostParadigm(data['secondary_paradigm'])
        
        # Reconstruct enum keys in dictionaries
        if 'distribution' in data:
            data['distribution'] = {
                HostParadigm(k): v for k, v in data['distribution'].items()
            }
        if 'reasoning' in data:
            data['reasoning'] = {
                HostParadigm(k): v for k, v in data['reasoning'].items()
            }
        
        return cls(**data)


class SearchResultSchema(BaseModel):
    """Schema for search results with origin tracking"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    url: str
    title: str
    snippet: str
    full_content: Optional[str] = None
    source_api: str
    credibility_score: float = 0.5
    credibility_explanation: Optional[str] = None
    origin_query: Optional[str] = None
    origin_query_id: Optional[str] = None
    paradigm_alignment: Optional[HostParadigm] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        data = self.model_dump(mode='json')
        if self.paradigm_alignment:
            data['paradigm_alignment'] = self.paradigm_alignment.value
        return data


class ContextLayerDebugInfo(BaseModel):
    """Debug information from context engineering layers"""
    layer_name: str
    processing_time_ms: float
    input_state: Dict[str, Any]
    output_state: Dict[str, Any]
    reasoning: List[str]
    removed_elements: Optional[List[str]] = None
    added_elements: Optional[List[str]] = None
    transformations: Optional[Dict[str, Any]] = None


class ContextEngineeredQuerySchema(BaseModel):
    """Schema for context-engineered queries with full debug info"""
    original_query: str
    classification: ClassificationResultSchema
    
    # Layer outputs
    write_layer_output: Dict[str, Any]
    select_layer_output: Dict[str, Any]
    compress_layer_output: Dict[str, Any]
    isolate_layer_output: Dict[str, Any]
    
    # Debug information
    debug_info: Optional[List[ContextLayerDebugInfo]] = None
    
    # Final outputs
    refined_queries: List[str]
    search_strategy: Dict[str, Any]
    context_metadata: Dict[str, Any]
    
    def get_paradigm_context(self) -> Dict[str, Any]:
        """Extract full paradigm context for answer generation"""
        return {
            "classification": self.classification.to_redis_dict(),
            "layer_outputs": {
                "write": self.write_layer_output,
                "select": self.select_layer_output,
                "compress": self.compress_layer_output,
                "isolate": self.isolate_layer_output
            },
            "debug_info": [d.model_dump() for d in self.debug_info] if self.debug_info else [],
            "search_strategy": self.search_strategy,
            "context_metadata": self.context_metadata
        }


class UserContextSchema(BaseModel):
    """Schema for user context that flows through the pipeline"""
    user_id: str
    role: str
    preferences: Dict[str, Any] = Field(default_factory=dict)
    location: Optional[str] = None
    language: str = "en"
    timezone: str = "UTC"
    max_sources: int = 10
    default_paradigm: Optional[HostParadigm] = None
    verbosity_preference: str = "balanced"  # minimal, balanced, detailed
    
    @property
    def is_pro_user(self) -> bool:
        return self.role in ["PRO", "ENTERPRISE", "ADMIN"]
    
    @property
    def source_limit(self) -> int:
        """Get source limit based on user role"""
        role_limits = {
            "FREE": 5,
            "BASIC": 10,
            "PRO": 20,
            "ENTERPRISE": 50,
            "ADMIN": 100
        }
        return role_limits.get(self.role, 5)


class ResearchRequestSchema(BaseModel):
    """Schema for research requests"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    user_context: UserContextSchema
    options: Dict[str, Any] = Field(default_factory=dict)
    depth: ResearchDepth = ResearchDepth.STANDARD
    classification: Optional[ClassificationResultSchema] = None
    status: ResearchStatus = ResearchStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def to_redis_dict(self) -> Dict[str, Any]:
        """Convert to Redis-storable dictionary"""
        data = self.model_dump(mode='json')
        if self.classification:
            data['classification'] = self.classification.to_redis_dict()
        return data


class WebSocketMessageSchema(BaseModel):
    """Enhanced WebSocket message with sequence tracking"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sequence_number: int
    type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    research_id: Optional[str] = None
    data: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_json(self) -> str:
        """Convert to JSON with proper datetime handling"""
        return self.model_dump_json()


class ResearchResultSchema(BaseModel):
    """Complete research result with all context preserved"""
    research_id: str
    query: str
    classification: ClassificationResultSchema
    context_engineering: ContextEngineeredQuerySchema
    search_results: List[SearchResultSchema]
    answer: str
    paradigm_tone: HostParadigm
    sources_used: List[Dict[str, Any]]
    credibility_summary: Dict[str, Any]
    user_context: UserContextSchema
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    def to_api_response(self, include_debug: bool = False) -> Dict[str, Any]:
        """Convert to API response format"""
        response = {
            "research_id": self.research_id,
            "query": self.query,
            "answer": self.answer,
            "paradigm": self.paradigm_tone.value,
            "sources": self.sources_used,
            "credibility": self.credibility_summary,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp.isoformat()
        }
        
        if include_debug:
            response["debug"] = {
                "classification": self.classification.to_redis_dict(),
                "context_engineering": self.context_engineering.get_paradigm_context(),
                "search_results_count": len(self.search_results),
                "user_context": self.user_context.model_dump()
            }
        
        return response