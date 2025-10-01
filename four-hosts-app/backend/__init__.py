"""Backend package marker and compatibility shims.

This module ensures both ``backend.services`` *and* the historical
``services`` import path resolve to the same package so that relative imports
defined inside ``backend.services`` (e.g. ``from ..utils import …``) work
correctly during tests.
"""

from __future__ import annotations

import sys


def _alias_services_package() -> None:
    """Expose ``backend.services`` under the top-level ``services`` name.

    Some older modules and tests import ``services`` directly. Without this
    alias those imports resolve to a top-level package which breaks relative
    imports (``..utils``) inside the services modules, triggering
    ``ImportError: attempted relative import beyond top-level package`` during
    collection.
    """

    try:
        from . import services as _services_pkg  # noqa: WPS433 - limited scope
    except Exception:  # pragma: no cover - best effort aliasing
        return

    sys.modules.setdefault("services", _services_pkg)

    # Adjust spec metadata so relative imports treat the package as
    # ``backend.services`` even when accessed via the legacy ``services`` name.
    try:
        _services_pkg.__package__ = "backend.services"
        if getattr(_services_pkg, "__spec__", None):
            _services_pkg.__spec__.name = "backend.services"  # type: ignore[attr-defined]
            _services_pkg.__spec__.parent = "backend"  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - defensive guard
        pass


_alias_services_package()

__all__ = ["_alias_services_package"]
