"""Compatibility entrypoint.

Provides `app` for existing scripts expecting `uvicorn main:app` while
delegating implementation to `main_new` module (refactored code).

Also exposes a `classify_paradigm` coroutine for legacy tests that import it
from `main`. This forwards to the new classification engine.
"""

from main_new import app  # re-export FastAPI instance

__all__ = ["app"]


# Legacy helper expected by older integration tests
async def classify_paradigm(query: str):
    from services.classification_engine import classification_engine

    return await classification_engine.classify_query(query)
