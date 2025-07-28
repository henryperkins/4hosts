"""
Legacy compatibility layer.

`main.py` (and a few other modules) still import::

    from services.auth_service import AuthService

The real implementation now lives in `services.auth`.  To avoid touching all
call-sites we re-export the relevant symbols here.
"""

from .auth import (
    AuthService,
    TokenData,
    create_access_token,
    hash_password,  # convenience
    verify_password,  # convenience
)

__all__ = [
    "AuthService",
    "TokenData",
    "create_access_token",
    "hash_password",
    "verify_password",
]
