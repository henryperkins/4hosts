"""Shared helpers for creating Redis clients with consistent settings."""

from __future__ import annotations

import os
from typing import Optional

import redis.asyncio as redis

DEFAULT_REDIS_URL = "redis://localhost:6379"


def get_redis_url(env_var: str = "REDIS_URL", default: str = DEFAULT_REDIS_URL) -> str:
    """Return the Redis URL from environment or a sane default."""
    candidate = os.getenv(env_var, "").strip()
    return candidate or default


def create_async_redis(
    url: Optional[str] = None,
    *,
    decode_responses: bool = True,
    max_connections: Optional[int] = None,
    **kwargs,
) -> redis.Redis:
    """Create an async Redis client using a shared configuration."""
    connection_url = url or get_redis_url()
    client = redis.from_url(
        connection_url,
        decode_responses=decode_responses,
        max_connections=max_connections,
        **kwargs,
    )
    return client


async def ping(client: Optional[redis.Redis]) -> bool:
    """Best-effort ping helper that returns False when client is unavailable."""
    if client is None:
        return False
    try:
        await client.ping()
        return True
    except Exception:
        return False
