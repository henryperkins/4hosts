"""
Core configuration and settings for Four Hosts Research API
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
        )
    },
    "teddy": {
        "paradigm": "teddy",
        "name": "Devotion Paradigm",
        "description": "Provide compassionate support and protective measures."
    },
    "bernard": {
        "paradigm": "bernard",
        "name": "Analytical Paradigm",
        "description": "Focus on empirical, data-driven analysis."
    },
    "maeve": {
        "paradigm": "maeve",
        "name": "Strategic Paradigm",
        "description": "Deliver actionable strategies and optimization."
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
