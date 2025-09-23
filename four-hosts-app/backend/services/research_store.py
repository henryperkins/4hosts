"""
Research Store Service
Provides Redis-backed research data storage with fallback to in-memory storage
"""

import json
import structlog
from typing import Dict, Optional, List, Any
import os
from utils.date_utils import get_current_utc

from logging_config import configure_logging

configure_logging()
logger = structlog.get_logger(__name__)


class ResearchStore:
    """Research data storage with Redis backend and in-memory fallback"""

    def __init__(self, redis_url: Optional[str] = None):
        # Keep line length under flake8 limits
        env_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_url = redis_url or env_url
        self.redis_client = None
        self.key_prefix = "research:"
        self.fallback_store: Dict[str, Dict] = {}  # In-memory fallback
        # Track expirations for fallback items to approximate Redis TTL
        self.fallback_store_exp: Dict[str, float] = {}
        self.use_redis = False
        self.use_redis_reason: str = "uninitialized"

    async def initialize(self):
        """Initialize Redis connection with fallback to in-memory"""
        try:
            import redis.asyncio as redis

            # from_url is not awaitable; configure decode_responses to get str
            # configure decode_responses to get str values from redis
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True
            )
            # Test connection
            await self.redis_client.ping()
            self.use_redis = True
            self.use_redis_reason = "ok"
            logger.info("✓ Redis research store initialized")
        except Exception as e:
            self.use_redis = False
            self.use_redis_reason = str(e)
            logger.warning(
                "Redis unavailable, using in-memory store: %s (url=%s)",
                str(e),
                self.redis_url,
            )

    def _purge_fallback_expired(self):
        """Remove expired entries from fallback store."""
        if not self.fallback_store_exp:
            return
        now_ts = get_current_utc().timestamp()
        expired = [
            k for k, ts in self.fallback_store_exp.items()
            if ts <= now_ts
        ]
        for k in expired:
            self.fallback_store.pop(k, None)
            self.fallback_store_exp.pop(k, None)

    async def set(self, research_id: str, data: Dict):
        """Store research data"""
        if self.use_redis and self.redis_client:
            try:
                key = f"{self.key_prefix}{research_id}"
                serialized_data = json.dumps(data, default=str)
                # 24h TTL
                await self.redis_client.set(
                    key,
                    serialized_data,
                    ex=86400
                )
                return
            except Exception as e:
                logger.error("Redis set error: %s", str(e))

        # Fallback to in-memory with TTL approximation (24h) and timestamp
        self._purge_fallback_expired()
        data = dict(data or {})
        data["_updated_at"] = get_current_utc().isoformat()
        self.fallback_store[research_id] = data
        self.fallback_store_exp[research_id] = (
            get_current_utc().timestamp() + 86400
        )

    async def get(self, research_id: str) -> Optional[Dict]:
        """Get research data"""
        if self.use_redis and self.redis_client:
            try:
                key = f"{self.key_prefix}{research_id}"
                data = await self.redis_client.get(key)
                if not data:
                    return None
                # decode_responses=True ensures str here
                record = json.loads(data)
                return record
            except Exception as e:
                logger.error("Redis get error: %s", str(e))

        # Fallback to in-memory
        self._purge_fallback_expired()
        record = self.fallback_store.get(research_id)
        if record:
            # Warn when using in-memory, non-persistent state
            logger.warning(
                "ResearchStore: resuming from in-memory fallback; "
                "state may be non-persistent"
            )
        return record

    async def exists(self, research_id: str) -> bool:
        """Check if research exists"""
        if self.use_redis and self.redis_client:
            try:
                key = f"{self.key_prefix}{research_id}"
                return bool(await self.redis_client.exists(key))
            except Exception as e:
                logger.error("Redis exists error: %s", str(e))

        # Fallback to in-memory
        self._purge_fallback_expired()
        return research_id in self.fallback_store

    async def update_field(self, research_id: str, field: str, value: Any):
        """Update a specific field (delegates to update_fields)."""
        await self.update_fields(research_id, {field: value})

    async def update_fields(self, research_id: str, patch: Dict[str, Any]):
        """
        Atomically update multiple fields by performing a single read/merge/write
        of the full research record. Adds/bumps a monotonically increasing
        `version` and updates `_updated_at` timestamp on every write.

        For Redis, this results in a single SET of the merged JSON payload which
        is atomic at the key level. For the in-memory fallback it updates the
        single object reference.
        """
        if not isinstance(patch, dict):
            patch = {"_patch": patch}

        # Read existing record (or start a new one if missing)
        current = await self.get(research_id) or {}

        # Merge and bump version
        try:
            prev_version = int(current.get("version", 0) or 0)
        except Exception:
            prev_version = 0

        merged = dict(current)
        merged.update(patch)
        merged["version"] = prev_version + 1
        merged["_updated_at"] = get_current_utc().isoformat()

        # Write back once
        await self.set(research_id, merged)

    async def get_user_research(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[Dict]:
        """Get all research for a user"""
        results = []

        if self.use_redis and self.redis_client:
            try:
                # Scan Redis keys
                pattern = f"{self.key_prefix}*"
                async for key in self.redis_client.scan_iter(pattern):
                    if len(results) >= limit:
                        break
                    data = await self.redis_client.get(key)
                    if data:
                        research = json.loads(data)
                        if research.get("user_id") == user_id:
                            results.append(research)
                return results
            except Exception as e:
                logger.error("Redis scan error: %s", str(e))

        # Fallback to in-memory
        self._purge_fallback_expired()
        for research in self.fallback_store.values():
            if research.get("user_id") == user_id:
                results.append(research)
                if len(results) >= limit:
                    break

        return results

    def values(self):
        """Get all values - for backward compatibility"""
        if self.use_redis:
            logger.warning(
                "values() method not efficient with Redis - "
                "use get_user_research instead"
            )
        return self.fallback_store.values()

    def __contains__(self, research_id: str) -> bool:
        """Check if research_id exists - for backward compatibility"""
        # This is synchronous for backward compatibility
        # For async usage, use exists() method
        if self.use_redis:
            logger.warning(
                "Synchronous contains check not available with Redis"
            )
            return False
        return research_id in self.fallback_store

    def __getitem__(self, research_id: str) -> Dict:
        """Get item - for backward compatibility"""
        # This is synchronous for backward compatibility
        # For async usage, use get() method
        if self.use_redis:
            logger.warning("Synchronous item access not available with Redis")
            return {}
        return self.fallback_store.get(research_id, {})

    def __setitem__(self, research_id: str, data: Dict):
        """Set item - for backward compatibility"""
        # This is synchronous for backward compatibility
        # For async usage, use set() method
        if self.use_redis:
            logger.warning("Synchronous item setting not available with Redis")
            return
        self.fallback_store[research_id] = data


# Create global instance
research_store = ResearchStore()
