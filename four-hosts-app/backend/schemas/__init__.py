"""Python packages mapping to JSON schemas.

Currently exposes `research` sub-module to provide legacy symbols
(`SearchResult`, `HostParadigm`) expected by a handful of historical
tests that have not yet migrated to the new contracts package.  We
re-export the canonical models so no duplicate definitions creep in.
"""

from importlib import import_module as _imp

# Lazily create `schemas.research` when first imported so that the
# symbol table is populated from the authoritative locations.

import types as _types
import sys as _sys


def _create_research_module() -> None:
    """Build a surrogate module that proxies to real implementations."""

    mod = _types.ModuleType("schemas.research")

    # HostParadigm is defined in services.classification_engine; we import
    # the Enum directly to avoid subtle duplicates.
    HostParadigm = _imp("services.classification_engine").HostParadigm

    # SearchResult lives in `contracts`; re-use that Pydantic model.
    SearchResult = _imp("contracts").SearchResult

    mod.HostParadigm = HostParadigm
    mod.SearchResult = SearchResult

    # Make module discoverable via sys.modules so `import schemas.research`
    # works transparently.
    _sys.modules[mod.__name__] = mod


# Eagerly register upon package import so downstream calls succeed.
_create_research_module()

# Clean up helper names
del _imp, _types, _sys, _create_research_module
