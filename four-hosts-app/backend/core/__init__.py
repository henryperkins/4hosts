"""
Core package for Four Hosts Research API.

This module re-exports the public configuration values, feature flags,
helper functions, and dependency utilities so that other parts of the
application (and external packages) can import them from `core`
without needing to know the internal module layout.

Prefer importing from `core` rather than submodules to keep the public
API stable.
"""

# Avoid importing create_app here to prevent circular imports
# from core.app import create_app

from core.config import (
    # Environment / host helpers
    TRUSTED_ORIGINS,
    PARADIGM_EXPLANATIONS,
    get_environment,
    get_allowed_hosts,
    is_production,

    # Auth / security basics
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,

    # Synthesis sizing defaults
    SYNTHESIS_BASE_WORDS,
    SYNTHESIS_BASE_TOKENS,
    SYNTHESIS_MAX_LENGTH_DEFAULT,

    # Evidence handling defaults
    EVIDENCE_MAX_QUOTES_DEFAULT,
    EVIDENCE_BUDGET_TOKENS_DEFAULT,
    EVIDENCE_MAX_DOCS_DEFAULT,
    EVIDENCE_QUOTES_PER_DOC_DEFAULT,
    EVIDENCE_QUOTE_MAX_CHARS,
    EVIDENCE_SEMANTIC_SCORING,
    EVIDENCE_INCLUDE_SUMMARIES,

    # Feature flags: Feedback
    ENABLE_FEEDBACK_RATE_LIMIT,
    FEEDBACK_RATE_LIMIT_PER_MINUTE,
    ENABLE_FEEDBACK_RECONCILE,
    FEEDBACK_RECONCILE_WINDOW_MINUTES,

    # Feature flags: Mesh Network
    ENABLE_MESH_NETWORK,
    MESH_MIN_PROBABILITY,
    MESH_MAX_PARADIGMS,
)

from core.dependencies import (
    get_current_user,
    get_current_user_optional,
    require_role,
    get_request_id,
)

from core.error_handlers import (
    validation_exception_handler,
    http_exception_handler,
    general_exception_handler,
)

__all__ = [
    # Environment / host helpers
    "TRUSTED_ORIGINS",
    "PARADIGM_EXPLANATIONS",
    "get_environment",
    "get_allowed_hosts",
    "is_production",

    # Auth / security basics
    "ALGORITHM",
    "ACCESS_TOKEN_EXPIRE_MINUTES",

    # Synthesis sizing defaults
    "SYNTHESIS_BASE_WORDS",
    "SYNTHESIS_BASE_TOKENS",
    "SYNTHESIS_MAX_LENGTH_DEFAULT",

    # Evidence handling defaults
    "EVIDENCE_MAX_QUOTES_DEFAULT",
    "EVIDENCE_BUDGET_TOKENS_DEFAULT",
    "EVIDENCE_MAX_DOCS_DEFAULT",
    "EVIDENCE_QUOTES_PER_DOC_DEFAULT",
    "EVIDENCE_QUOTE_MAX_CHARS",
    "EVIDENCE_SEMANTIC_SCORING",
    "EVIDENCE_INCLUDE_SUMMARIES",

    # Feature flags: Feedback
    "ENABLE_FEEDBACK_RATE_LIMIT",
    "FEEDBACK_RATE_LIMIT_PER_MINUTE",
    "ENABLE_FEEDBACK_RECONCILE",
    "FEEDBACK_RECONCILE_WINDOW_MINUTES",

    # Feature flags: Mesh Network
    "ENABLE_MESH_NETWORK",
    "MESH_MIN_PROBABILITY",
    "MESH_MAX_PARADIGMS",

    # Dependency helpers
    "get_current_user",
    "get_current_user_optional",
    "require_role",
    "get_request_id",

    # Error handlers
    "validation_exception_handler",
    "http_exception_handler",
    "general_exception_handler",
]
