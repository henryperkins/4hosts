"""
Routes package for Four Hosts Research API.

This module aggregates all FastAPI APIRouter instances so they can be
imported centrally. It also provides helper utilities to register every
router on an application with a consistent versioned prefix.

NOTE:
Routers in this project define prefixes like "/auth", "/research", etc.
The application (core.app / setup code) is expected to mount them under
a version path (e.g. "/v1"). Avoid putting the version segment inside
individual route modules to prevent duplicated paths ("/v1/v1/...").
"""

from __future__ import annotations
from typing import Iterable, Tuple
from fastapi import FastAPI

# Individual routers
from routes.auth import router as auth_router
from routes.research import router as research_router
from routes.paradigms import router as paradigms_router
from routes.search import router as search_router
from routes.users import router as users_router
from routes.system import router as system_router
from routes.responses import router as responses_router
from routes.feedback import router as feedback_router  # Newly added router

# Public re-export list
__all__ = [
    "auth_router",
    "research_router",
    "paradigms_router",
    "search_router",
    "users_router",
    "system_router",
    "responses_router",
    "feedback_router",
    "all_routers",
    "iter_routers",
    "register_all_routers",
]

# Ordered collection (order can matter for overlapping paths or dependency init)
all_routers = [
    auth_router,
    research_router,
    paradigms_router,
    search_router,
    users_router,
    system_router,
    responses_router,
    feedback_router,
]


def iter_routers() -> Iterable:
    """Yield all router objects (simple iterator helper)."""
    yield from all_routers


def register_all_routers(
    app: FastAPI,
    *,
    version_prefix: str = "/v1",
    include_feedback: bool = True,
) -> None:
    """
    Register all routers on the provided FastAPI application.

    Args:
        app: FastAPI application instance.
        version_prefix: Base prefix under which feature routers are mounted.
        include_feedback: Allow excluding feedback endpoints if a deployment
                          wants to disable that feature quickly.
    """
    for router in all_routers:
        if not include_feedback and router is feedback_router:
            continue
        # Each router already has its own prefix (e.g. "/auth"), so we layer
        # the version prefix in front.
        app.include_router(router, prefix=version_prefix)


def list_routes_summary() -> list[Tuple[str, list[str]]]:
    """
    Return a lightweight summary of (prefix, tags) for introspection,
    diagnostics or documentation generation.
    """
    summary: list[Tuple[str, list[str]]] = []
    for r in all_routers:
        prefix = getattr(r, "prefix", "")
        tags = getattr(r, "tags", [])
        summary.append((prefix, list(tags)))
    return summary
