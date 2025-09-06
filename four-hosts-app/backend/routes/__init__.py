"""
Routes package for Four Hosts Research API
"""

from routes.auth import router as auth_router
from routes.research import router as research_router
from routes.paradigms import router as paradigms_router
from routes.search import router as search_router
from routes.users import router as users_router
from routes.system import router as system_router
from routes.responses import router as responses_router

__all__ = [
    "auth_router",
    "research_router",
    "paradigms_router",
    "search_router",
    "users_router",
    "system_router",
    "responses_router",
]
