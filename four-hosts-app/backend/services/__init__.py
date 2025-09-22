"""
Four Hosts services package public API.

This module now exposes (almost) every public symbol defined in any module
under the services package so that they can be imported directly, e.g.:

    from services import (
        research_store, initialize_cache, cache_manager,
        initialize_research_system, initialise_llm_client,
        create_search_manager, BraveSearchAPI, SearchConfig,
        classification_engine, HostParadigm,
        self_healing_system, WebhookManager, ExportService,
        answer_generator, deep_research_service,  # module aliases
        AnswerGenerator, DeepResearchService,     # example symbols inside modules
    )

Export Strategy
---------------
1. Explicit Core Map (_EXPLICIT_EXPORTS)
   A hand‑curated, backwards‑compatible set of stable entrypoints kept for
   clarity and to guarantee lazy loading semantics. These ALWAYS take
   precedence over any automatically discovered symbol.

2. Automatic Discovery (optional at import time)
   Unless the environment variable FOUR_HOSTS_SERVICES_FAST_INIT=1 is set,
   every submodule (and subpackage) is imported once at package import time
   to enumerate its public symbols:
       * If a module defines __all__, those names are considered public.
       * Otherwise any attribute not starting with '_' is considered public.
   Each discovered symbol is added to the public export surface unless it
   collides with an explicit export (explicit always wins) or an earlier
   discovered symbol (first discovery wins).

   In addition, each module itself is made importable by name:
       from services import answer_generator
   (i.e. the module object is exported under its module basename.)

3. Fast Init / Lazy Fallback Mode
   If FOUR_HOSTS_SERVICES_FAST_INIT=1 is set, the automatic eager scan is
   skipped to minimize startup cost. In that mode, when an unknown attribute
   is accessed, a lazy resolution pass attempts to locate the symbol by
   importing modules on demand (still respecting explicit overrides).

Collision Handling
------------------
Order of precedence (highest first):
    1. Explicit core exports (_EXPLICIT_EXPORTS)
    2. First discovered symbol during eager scan (if enabled)
    3. First module found lazily during on-demand resolution (FAST_INIT mode)

Thread Safety
-------------
A simple threading.Lock protects mutation of the dynamic export registry
during lazy resolution.

Environment Variables
---------------------
FOUR_HOSTS_SERVICES_FAST_INIT=1
    Skip the eager discovery scan; rely on lazy, on-demand symbol resolution.

Performance Notes
-----------------
The eager scan necessarily imports each module once. If this proves too
costly in certain deployment contexts (e.g. cold starts), enable FAST_INIT.

Developer Guidance
------------------
- To deliberately hide a symbol from automatic export, prefix it with '_'.
- To explicitly control which symbols are exported from a module, define
  __all__ = ['PublicName', ...] in that module.
- Add new high-value stable entrypoints to _EXPLICIT_EXPORTS for clarity.

"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Dict, Tuple
import os
import pkgutil
import pathlib
import threading

# -----------------------------------------------------------------------------
# 1. Hand‑curated explicit exports (backwards compatibility + stable surface)
# -----------------------------------------------------------------------------
_EXPLICIT_EXPORTS: Dict[str, Tuple[str, str]] = {
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

    # Auth essentials
    "auth_service": ("auth", "auth_service"),
    "get_current_user": ("auth", "get_current_user"),
    "decode_token": ("auth", "decode_token"),
    "TokenData": ("auth", "TokenData"),
    "UserRole": ("auth", "UserRole"),
    "require_role": ("auth", "require_role"),
}

# Working export registry (mutable; begins with explicit)
_EXPORTS: Dict[str, Tuple[str, str]] = dict(_EXPLICIT_EXPORTS)

# Track which modules we have fully loaded during eager discovery
_LOADED_MODULES: set[str] = set()
# Lock for thread-safe lazy resolution
_RESOLUTION_LOCK = threading.Lock()

# Environment flag
_FAST_INIT = os.environ.get("FOUR_HOSTS_SERVICES_FAST_INIT") == "1"


def _is_public_name(name: str) -> bool:
    return not name.startswith("_")


def _iter_service_modules():
    """
    Yield fully-qualified module names under this package.

    Uses pkgutil.walk_packages to include subpackages.
    """
    pkg_path = pathlib.Path(__file__).parent
    for module_info in pkgutil.walk_packages(
        [str(pkg_path)], prefix=__name__ + "."
    ):
        if module_info.name == __name__:
            continue
        # Skip compiled / namespace anomalies just in case
        yield module_info.name


def _import_module(module_fq_name: str):
    """
    Import a module, returning (module, short_name) or (None, short_name) if fails.
    """
    short_name = module_fq_name.rsplit(".", 1)[-1]
    try:
        module = import_module(module_fq_name)
        _LOADED_MODULES.add(module_fq_name)
        return module, short_name
    except Exception:
        # Intentionally swallow to avoid breaking import; unresolved modules
        # can still be resolved lazily if/when requested.
        return None, short_name


def _collect_public_symbols(module, short_name: str):
    """
    Collect and register public symbols from a module.

    1. Register the module object itself under its short name (if not colliding).
    2. For each public symbol, add to _EXPORTS if not already present.
    """
    # Export the module by its short name (module alias) if safe
    if short_name not in _EXPORTS and short_name not in globals():
        globals()[short_name] = module  # cache module object directly
        # Mark with a sentinel mapping so __getattr__ bypasses import
        _EXPORTS[short_name] = ("__module_alias__", short_name)

    if hasattr(module, "__all__"):
        public_names = [n for n in getattr(module, "__all__") if _is_public_name(n)]
    else:
        public_names = [n for n in dir(module) if _is_public_name(n)]

    for name in public_names:
        if name in _EXPORTS:  # do not override explicit or already registered
            continue
        attr = getattr(module, name, None)
        if attr is None:
            continue
        _EXPORTS[name] = (short_name, name)
        # Pre-cache the attribute to avoid repeated getattr if eager mode
        globals()[name] = attr


def _auto_collect_exports():
    """
    Eagerly import all service submodules and collect their public exports.

    Skips this process entirely if FAST_INIT mode is enabled.
    """
    if _FAST_INIT:
        return

    for module_fq_name in _iter_service_modules():
        module, short_name = _import_module(module_fq_name)
        if module is None:
            continue
        # Never re-process explicitly enumerated modules twice
        _collect_public_symbols(module, short_name)


def _lazy_resolve_symbol(name: str):
    """
    Attempt to locate 'name' by importing modules lazily (FAST_INIT path).

    Returns the resolved object or raises AttributeError.
    """
    # Double-checked locking pattern
    if name in globals():
        return globals()[name]

    with _RESOLUTION_LOCK:
        if name in globals():
            return globals()[name]

        # First search already loaded modules
        for module_fq_name in list(_LOADED_MODULES):
            module = import_module(module_fq_name)
            if hasattr(module, name):
                short = module_fq_name.rsplit(".", 1)[-1]
                if name not in _EXPORTS:
                    _EXPORTS[name] = (short, name)
                value = getattr(module, name)
                globals()[name] = value
                return value

        # Walk remaining modules
        for module_fq_name in _iter_service_modules():
            if module_fq_name not in _LOADED_MODULES:
                module, short_name = _import_module(module_fq_name)
                if module is None:
                    continue
                if hasattr(module, name):
                    if name not in _EXPORTS:
                        _EXPORTS[name] = (short_name, name)
                    value = getattr(module, name)
                    globals()[name] = value
                    # Also ensure module alias exported if not present
                    if short_name not in _EXPORTS and short_name not in globals():
                        globals()[short_name] = module
                        _EXPORTS[short_name] = ("__module_alias__", short_name)
                    return value

    raise AttributeError(f"module 'services' has no attribute '{name}'")


# Perform eager discovery unless FAST_INIT requested
_auto_collect_exports()

# Public symbol list (sorted for determinism)
__all__ = sorted(set(_EXPORTS.keys()))


def __getattr__(name: str):
    """
    PEP 562 lazy attribute access.

    Resolution order:
      1. If name registered in _EXPORTS:
            a. If module alias sentinel -> return cached module object.
            b. Import underlying module (if not yet imported) and return attribute.
      2. If FAST_INIT mode: attempt lazy on-demand resolution across modules.
      3. Otherwise raise AttributeError.
    """
    if name in _EXPORTS:
        module_name, attr = _EXPORTS[name]

        # Module alias sentinel path
        if module_name == "__module_alias__":
            # Should already be in globals (set during collection)
            value = globals().get(name)
            if value is not None:
                return value
            # Fallback: attempt real import (edge case)
            full_module_name = f"{__name__}.{attr}"
            value = import_module(full_module_name)
            globals()[name] = value
            return value

        module = import_module(f".{module_name}", __name__)
        value = getattr(module, attr)
        globals()[name] = value  # cache
        return value

    if _FAST_INIT:
        return _lazy_resolve_symbol(name)

    raise AttributeError(f"module 'services' has no attribute '{name}'")


def __dir__():
    return sorted(list(set(globals().keys()) | set(__all__)))


# -----------------------------------------------------------------------------
# Type checking support (only retains explicit core exports; dynamically
# discovered names are not enumerated here to avoid import side-effects)
# -----------------------------------------------------------------------------
if TYPE_CHECKING:
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
