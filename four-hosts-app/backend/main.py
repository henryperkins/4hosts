"""Compatibility entrypoint.

Provides `app` for existing scripts expecting `uvicorn main:app` while
delegating implementation to `main_new` module (refactored code).
"""

from main_new import app  # re-export FastAPI instance

__all__ = ["app"]
