"""
Base models and common types for the Four Hosts Research API
"""

from enum import Enum
from typing import Optional, Dict
from pydantic import BaseModel

# Import compatibility types
from services.classification_engine import HostParadigm
from database.models import (
    UserRole as _DBUserRole,
    ParadigmType as _DBParadigm
)


class ResearchDepth(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"
    DEEP_RESEARCH = "deep_research"  # Uses o3-deep-research model


class ResearchStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Paradigm(str, Enum):
    DOLORES = "dolores"
    TEDDY = "teddy"
    BERNARD = "bernard"
    MAEVE = "maeve"


class UserRole(str, Enum):
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"


class SourceResult(BaseModel):
    title: str
    url: str
    snippet: str
    domain: str
    credibility_score: float
    published_date: Optional[str] = None
    source_type: str = "web"


class ParadigmClassification(BaseModel):
    primary: Paradigm
    secondary: Optional[Paradigm]
    distribution: Dict[str, float]
    confidence: float
    explanation: Dict[str, str]

    class Config:
        use_enum_values = True


# Mapping for backward compatibility
HOST_TO_MAIN_PARADIGM = {
    HostParadigm.DOLORES: Paradigm.DOLORES,
    HostParadigm.TEDDY: Paradigm.TEDDY,
    HostParadigm.BERNARD: Paradigm.BERNARD,
    HostParadigm.MAEVE: Paradigm.MAEVE,
}

# Re-export canonical definitions (for backward compatibility)
DBUserRole = _DBUserRole
DBParadigm = _DBParadigm
