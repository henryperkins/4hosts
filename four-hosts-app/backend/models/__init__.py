"""
Models package for Four Hosts Research API
"""

from models.base import (
    ResearchDepth,
    ResearchStatus,
    Paradigm,
    UserRole,
    SourceResult,
    ParadigmClassification,
    HOST_TO_MAIN_PARADIGM,
    DBUserRole,
    DBParadigm
)

from models.research import (
    ResearchOptions,
    ResearchQuery,
    ResearchResult,
    ParadigmOverrideRequest,
    ClassifyRequest,
    ResearchDeepQuery
)

from models.auth import (
    UserCreate,
    UserLogin,
    Token,
    RefreshTokenRequest,
    LogoutRequest,
    PreferencesPayload
)

__all__ = [
    # Base models
    "ResearchDepth",
    "ResearchStatus",
    "Paradigm",
    "UserRole",
    "SourceResult",
    "ParadigmClassification",
    "HOST_TO_MAIN_PARADIGM",
    "DBUserRole",
    "DBParadigm",
    # Research models
    "ResearchOptions",
    "ResearchQuery",
    "ResearchResult",
    "ParadigmOverrideRequest",
    "ClassifyRequest",
    "ResearchDeepQuery",
    # Auth models
    "UserCreate",
    "UserLogin",
    "Token",
    "RefreshTokenRequest",
    "LogoutRequest",
    "PreferencesPayload"
]
