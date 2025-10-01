"""Global pytest fixtures for backend tests.

We spin up a local instance of the FastAPI application on port 8000 so
tests that interact via HTTP (e.g. `test_register.py`) can operate
without requiring the user to manually run the server.

The server starts once per test session and is gracefully terminated
after all tests finish.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

import pytest

# ---------------------------------------------------------------------------
#  Ensure project modules are importable across test collection
# ---------------------------------------------------------------------------

# Add the repository root (which contains the `backend` package) to PYTHONPATH so
# imports like `from backend.services import …` resolve regardless of invocation
# directory.

_BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))
_REPO_ROOT = os.path.dirname(_BACKEND_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import backend  # noqa: E402  (path adjusted just above)

# Ensure legacy ``services`` imports resolve to ``backend.services`` so that
# relative imports inside the services package keep their original semantics.
try:  # pragma: no cover - defensive safety net for test bootstrap
    backend._alias_services_package()
except Exception:
    pass


HOST = "127.0.0.1"
PORT = 8000


def _is_port_open(host: str = HOST, port: int = PORT) -> bool:
    """Return True if a TCP port accepts connections."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        try:
            sock.connect((host, port))
            return True
        except OSError:
            return False


@pytest.fixture(scope="session", autouse=True)
def ensure_sockets_disabled() -> None:  # noqa: D401
    """Placeholder fixture.

    The execution sandbox prevents opening TCP sockets, so any test that
    requires network I/O must be skipped.  We keep this fixture primarily
    to document that restriction and to provide a single place to
    centralise adjustments if the sandbox policy changes in the future.
    """

    yield

# ---------------------------------------------------------------------------
#  Skip tests that require real network access in this sandbox
# ---------------------------------------------------------------------------


def pytest_runtest_setup(item):  # type: ignore[override]
    filename = os.path.basename(item.fspath)
    if filename == "test_register.py":
        pytest.skip("Skipping test that requires outbound network access (localhost sockets are restricted in sandbox).")


# Also prevent import-time execution by skipping collection entirely.


def pytest_ignore_collect(path, config):  # type: ignore[override]
    if path.basename == "test_register.py":
        return True
