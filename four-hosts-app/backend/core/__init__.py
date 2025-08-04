"""
Core package for Four Hosts Research API
"""

from core.app import create_app
from core.config import (
    TRUSTED_ORIGINS,
    PARADIGM_EXPLANATIONS,
    get_environment,
    get_allowed_hosts,
    is_production
)
from core.dependencies import (
    get_current_user,
    get_current_user_optional,
    require_role,
    get_request_id
)
from core.error_handlers import (
    validation_exception_handler,
    http_exception_handler,
    general_exception_handler
)

__all__ = [
    "create_app",
    "TRUSTED_ORIGINS",
    "PARADIGM_EXPLANATIONS",
    "get_environment",
    "get_allowed_hosts",
    "is_production",
    "get_current_user",
    "get_current_user_optional",
    "require_role",
    "get_request_id",
    "validation_exception_handler",
    "http_exception_handler",
    "general_exception_handler"
]
