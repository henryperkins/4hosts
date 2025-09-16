"""
Core configuration and settings for Four Hosts Research API

This module centralizes tunable knobs for synthesis length, token budgets,
and evidence handling so we can relax previous hard caps without scattering
magic numbers throughout the codebase. Values can be overridden via env vars
to balance cost/perf by environment.
"""

import os
import secrets
from typing import Dict, Any, List

# CSRF Protection
CSRF_TOKEN_SECRET = secrets.token_urlsafe(32)

# Configuration
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Enhanced CORS and security middleware (locked-down)
TRUSTED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://api.4hosts.ai",
    "https://app.lakefrontdigital.io",
]

# Paradigm explanation data
PARADIGM_EXPLANATIONS: Dict[str, Dict[str, Any]] = {
    "dolores": {
        "paradigm": "dolores",
        "name": "Revolutionary Paradigm",
        "description": (
            "Expose systemic issues and empower transformative change."
        ),
        "approach_suggestion": "Focus on exposing systemic issues and empowering resistance",
    },
    "teddy": {
        "paradigm": "teddy",
        "name": "Devotion Paradigm",
        "description": "Provide compassionate support and protective measures.",
        "approach_suggestion": "Prioritize community support and protective measures",
    },
    "bernard": {
        "paradigm": "bernard",
        "name": "Analytical Paradigm",
        "description": "Focus on empirical, data-driven analysis.",
        "approach_suggestion": "Emphasize empirical research and data-driven analysis",
    },
    "maeve": {
        "paradigm": "maeve",
        "name": "Strategic Paradigm",
        "description": "Deliver actionable strategies and optimization.",
        "approach_suggestion": "Develop strategic frameworks and actionable plans",
    },
}


def get_environment() -> str:
    """Get current environment"""
    return os.getenv("ENVIRONMENT", "production")


def get_allowed_hosts() -> List[str]:
    """Get allowed hosts for trusted host middleware"""
    return os.getenv(
        "ALLOWED_HOSTS",
        "localhost,127.0.0.1,api.4hosts.ai"
    ).split(",")


def is_production() -> bool:
    """Check if running in production"""
    return get_environment() == "production"


# ────────────────────────────────────────────────────────────
#  Synthesis & Evidence Defaults (env‑overridable)
# ────────────────────────────────────────────────────────────

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except Exception:
        return default


# Baseline words per entire answer (allocated per section by weight)
# UPDATED FOR O3: Increased from 5000 to 50000 to utilize o3's 100k output capacity
SYNTHESIS_BASE_WORDS: int = _env_int("SYNTHESIS_BASE_WORDS", 50000)

# Baseline max output tokens per entire answer (allocated per section)
# UPDATED FOR O3: Increased from 8000 to 80000 to utilize o3's 100k output capacity
SYNTHESIS_BASE_TOKENS: int = _env_int("SYNTHESIS_BASE_TOKENS", 80000)

# Default max_length for SynthesisContext when callers omit it
# UPDATED FOR O3: Increased from 5000 to 50000 for comprehensive synthesis
SYNTHESIS_MAX_LENGTH_DEFAULT: int = _env_int("SYNTHESIS_MAX_LENGTH_DEFAULT", 50000)

# Evidence block budgets in prompts
# UPDATED FOR O3: Increased from 30 to 200 to include more evidence
EVIDENCE_MAX_QUOTES_DEFAULT: int = _env_int("EVIDENCE_MAX_QUOTES_DEFAULT", 200)
# UPDATED FOR O3: Increased from 2000 to 95000 to utilize o3's 100k input capacity
EVIDENCE_BUDGET_TOKENS_DEFAULT: int = _env_int("EVIDENCE_BUDGET_TOKENS_DEFAULT", 95000)

# Evidence extraction limits
# UPDATED FOR O3: Increased from 20 to 100 to analyze 5x more documents
EVIDENCE_MAX_DOCS_DEFAULT: int = _env_int("EVIDENCE_MAX_DOCS_DEFAULT", 100)
# UPDATED FOR O3: Increased from 3 to 10 for richer evidence extraction
EVIDENCE_QUOTES_PER_DOC_DEFAULT: int = _env_int("EVIDENCE_QUOTES_PER_DOC_DEFAULT", 10)
EVIDENCE_QUOTE_MAX_CHARS: int = _env_int("EVIDENCE_QUOTE_MAX_CHARS", 360)

# Enable lightweight semantic scoring for quote selection
EVIDENCE_SEMANTIC_SCORING: bool = (os.getenv("EVIDENCE_SEMANTIC_SCORING", "1").lower() in {"1", "true", "yes"})

# Include short per‑source summaries alongside quotes in prompts
EVIDENCE_INCLUDE_SUMMARIES: bool = (os.getenv("EVIDENCE_INCLUDE_SUMMARIES", "1").lower() in {"1", "true", "yes"})


# ────────────────────────────────────────────────────────────
#  Feature Flags
# ────────────────────────────────────────────────────────────

# Feedback
ENABLE_FEEDBACK_RATE_LIMIT: bool = (os.getenv("ENABLE_FEEDBACK_RATE_LIMIT", "1").lower() in {"1", "true", "yes"})
FEEDBACK_RATE_LIMIT_PER_MINUTE: int = _env_int("FEEDBACK_RATE_LIMIT_PER_MINUTE", 30)

ENABLE_FEEDBACK_RECONCILE: bool = (os.getenv("ENABLE_FEEDBACK_RECONCILE", "0").lower() in {"1", "true", "yes"})
FEEDBACK_RECONCILE_WINDOW_MINUTES: int = _env_int("FEEDBACK_RECONCILE_WINDOW_MINUTES", 10)

# Mesh Network
ENABLE_MESH_NETWORK: bool = (os.getenv("ENABLE_MESH_NETWORK", "0").lower() in {"1", "true", "yes"})
MESH_MIN_PROBABILITY: float = float(os.getenv("MESH_MIN_PROBABILITY", "0.25") or 0.25)
MESH_MAX_PARADIGMS: int = _env_int("MESH_MAX_PARADIGMS", 3)
