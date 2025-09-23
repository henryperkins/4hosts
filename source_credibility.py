"""Thin wrapper module exporting credibility utilities from backend services.

This provides a stable, top-level import path (`import source_credibility`) for
external callers that expect the public API described in the credibility engine
design docs.
"""

from __future__ import annotations

import sys
import pathlib

# Ensure backend package directory is on `sys.path` so that relative imports
# resolve correctly from the implementation module.
_BASE_DIR = pathlib.Path(__file__).resolve().parent
_BACKEND_DIR = _BASE_DIR / "four-hosts-app" / "backend"
# Insert **after** the repo root so that top-level helpers (e.g. `utils`)
# are not shadowed by the backend package of the same name.  This preserves
# existing behaviour for backend consumers while avoiding accidental import
# collisions.
if str(_BACKEND_DIR) not in sys.path:
    sys.path.append(str(_BACKEND_DIR))

# Import directly from the actual implementation
from services.credibility import (
    DomainAuthorityChecker,
    CredibilityScore,
    ControversyDetector,
    SourceReputationDatabase,
    analyze_source_credibility_batch,
    get_source_credibility,
)

# Make `from source_credibility import *` work as expected.
__all__ = [
    "DomainAuthorityChecker",
    "CredibilityScore",
    "ControversyDetector",
    "SourceReputationDatabase",
    "analyze_source_credibility_batch",
    "get_source_credibility",
]
