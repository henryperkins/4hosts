"""Utility helpers for running blocking functions in the async event loop.

These helpers centralise the anyio thread-offloading logic so that
all synchronous, CPU-bound or IO-bound calls (bcrypt, redis-py, jwt, subprocess)
can be executed without blocking the main event-loop.
"""

from __future__ import annotations

from typing import Callable, TypeVar, Any

import anyio

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


async def run_in_thread(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Execute *func* in a worker thread and await the result.

    anyio.to_thread.run_sync only forwards *positional* arguments, therefore
    we capture kwargs in a closure to preserve full signature compatibility.
    """

    if kwargs:
        return await anyio.to_thread.run_sync(lambda: func(*args, **kwargs))
    return await anyio.to_thread.run_sync(func, *args)
