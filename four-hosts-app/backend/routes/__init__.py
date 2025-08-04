"""
Routes package for Four Hosts Research API
"""

from routes.auth import router as auth_router
from routes.research import router as research_router
from routes.paradigms import router as paradigms_router

__all__ = [
    "auth_router",
    "research_router",
    "paradigms_router"
]
