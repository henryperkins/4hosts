"""
Four Hosts services package public API

This module consolidates the most commonly used service entrypoints behind a
stable import surface, while keeping imports lazy to avoid circular
dependencies and unnecessary import cost.

Usage examples:
    from services import (
        research_store, initialize_cache, cache_manager,
        initialize_research_system, initialise_llm_client,
        create_search_manager, BraveSearchAPI, SearchConfig,
        classification_engine, HostParadigm,
        self_healing_system, WebhookManager, ExportService,
    )

Note: Internals remain accessible via explicit submodule imports
      (e.g., `from services.research_orchestrator import research_orchestrator`).
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

# Map public names to (module, attribute) for lazy access
_EXPORTS: dict[str, tuple[str, str]] = {
    # Data store and caching
    "research_store": ("research_store", "research_store"),
    "initialize_cache": ("cache", "initialize_cache"),
    "cache_manager": ("cache", "cache_manager"),

    # Research orchestration
    "research_orchestrator": ("research_orchestrator", "research_orchestrator"),
    "initialize_research_system": ("research_orchestrator", "initialize_research_system"),

    # LLM client
    "llm_client": ("llm_client", "llm_client"),
    "initialise_llm_client": ("llm_client", "initialise_llm_client"),

    # Search
    "create_search_manager": ("search_apis", "create_search_manager"),
    "BraveSearchAPI": ("search_apis", "BraveSearchAPI"),
    "SearchConfig": ("search_apis", "SearchConfig"),

    # Classification
    "classification_engine": ("classification_engine", "classification_engine"),
    "HostParadigm": ("classification_engine", "HostParadigm"),

    # Systems and utilities
    "self_healing_system": ("self_healing_system", "self_healing_system"),
    "WebhookManager": ("webhook_manager", "WebhookManager"),
    "ExportService": ("export_service", "ExportService"),
    "background_llm_manager": ("background_llm", "background_llm_manager"),
    "task_registry": ("task_registry", "task_registry"),

    # Auth essentials (kept minimal here to avoid heavy dependency fan-in)
    "auth_service": ("auth", "auth_service"),
    "get_current_user": ("auth", "get_current_user"),
    "decode_token": ("auth", "decode_token"),
    "TokenData": ("auth", "TokenData"),
    "UserRole": ("auth", "UserRole"),
    "require_role": ("auth", "require_role"),
}

__all__ = sorted(_EXPORTS.keys())


def __getattr__(name: str):  # PEP 562 lazy attribute access
    if name in _EXPORTS:
        module_name, attr = _EXPORTS[name]
        module = import_module(f".{module_name}", __name__)
        value = getattr(module, attr)
        globals()[name] = value  # cache for future access
        return value
    raise AttributeError(f"module 'services' has no attribute '{name}'")


def __dir__():
    return sorted(list(globals().keys()) + list(__all__))


if TYPE_CHECKING:
    # These imports are for type checkers and IDEs only; at runtime we rely on lazy loading.
    from .cache import cache_manager, initialize_cache  # noqa: F401
    from .classification_engine import HostParadigm, classification_engine  # noqa: F401
    from .export_service import ExportService  # noqa: F401
    from .llm_client import llm_client, initialise_llm_client  # noqa: F401
    from .research_orchestrator import (
        initialize_research_system, research_orchestrator,
    )  # noqa: F401
    from .research_store import research_store  # noqa: F401
    from .search_apis import BraveSearchAPI, SearchConfig, create_search_manager  # noqa: F401
    from .self_healing_system import self_healing_system  # noqa: F401
    from .task_registry import task_registry  # noqa: F401
    from .webhook_manager import WebhookManager  # noqa: F401
    from .auth import (
        auth_service, decode_token, get_current_user, require_role, TokenData, UserRole,
    )  # noqa: F401
