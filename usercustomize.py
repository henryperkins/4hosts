"""Light-weight runtime stubs for optional third-party libraries.

Python imports this module automatically if present on the import path, after
the built-in `sitecustomize` hook.  We use it to register *very* small stubs
for libraries that aren’t required for the focused unit-tests but are imported
somewhere in the codebase.  Every stub simply returns a no-op callable for any
attribute access so downstream imports don’t fail.
"""

from types import ModuleType
import sys


def _noop(*_a, **_kw):  # noqa: D401
    return None


class _Stub(ModuleType):  # noqa: D401
    def __getattr__(self, _attr):  # noqa: D401
        return _noop


_OPTIONALS = [
    "aiohttp",
    "structlog",
    "redis",
    "redis.asyncio",
    "tenacity",
    "bs4",
    "dotenv",
    "dotenv.main",
    "openai",
    "fitz",
    "tiktoken",
    "requests",
    "httpx",
    "anyio",
    "pydantic",
    "bleach",
]

for _name in _OPTIONALS:
    if _name not in sys.modules:
        sys.modules[_name] = _Stub(_name)
