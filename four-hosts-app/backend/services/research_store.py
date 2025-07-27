"""
Research Store Service
Provides Redis-backed research data storage with fallback to in-memory storage
"""

import json
import logging
from typing import Dict, Optional, List, Any
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class ResearchStore:
    """Research data storage with Redis backend and in-memory fallback"""

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_client = None
        self.key_prefix = "research:"
        self.fallback_store: Dict[str, Dict] = {}  # In-memory fallback
        self.use_redis = False

    async def initialize(self):
        """Initialize Redis connection with fallback to in-memory"""
        try:
            import redis.asyncio as redis
            self.redis_client = await redis.from_url(self.redis_url)
            # Test connection
            await self.redis_client.ping()
            self.use_redis = True
            logger.info("✓ Redis research store initialized")
        except Exception as e:
            logger.warning(f"Redis unavailable, using in-memory store: {str(e)}")
            self.use_redis = False

    async def set(self, research_id: str, data: Dict):
        """Store research data"""
        if self.use_redis and self.redis_client:
            try:
                key = f"{self.key_prefix}{research_id}"
                serialized_data = json.dumps(data, default=str)
                await self.redis_client.set(key, serialized_data, ex=86400)  # 24h TTL
                return
            except Exception as e:
                logger.error(f"Redis set error: {str(e)}")

        # Fallback to in-memory
        self.fallback_store[research_id] = data

    async def get(self, research_id: str) -> Optional[Dict]:
        """Get research data"""
        if self.use_redis and self.redis_client:
            try:
                key = f"{self.key_prefix}{research_id}"
                data = await self.redis_client.get(key)
                return json.loads(data) if data else None
            except Exception as e:
                logger.error(f"Redis get error: {str(e)}")

        # Fallback to in-memory
        return self.fallback_store.get(research_id)

    async def exists(self, research_id: str) -> bool:
        """Check if research exists"""
        if self.use_redis and self.redis_client:
            try:
                key = f"{self.key_prefix}{research_id}"
                return await self.redis_client.exists(key)
            except Exception as e:
                logger.error(f"Redis exists error: {str(e)}")

        # Fallback to in-memory
        return research_id in self.fallback_store

    async def update_field(self, research_id: str, field: str, value: Any):
        """Update a specific field"""
        data = await self.get(research_id)
        if data:
            data[field] = value
            await self.set(research_id, data)

    async def get_user_research(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get all research for a user"""
        results = []

        if self.use_redis and self.redis_client:
            try:
                # Scan Redis keys
                async for key in self.redis_client.scan_iter(f"{self.key_prefix}*"):
                    if len(results) >= limit:
                        break
                    data = await self.redis_client.get(key)
                    if data:
                        research = json.loads(data)
                        if research.get("user_id") == user_id:
                            results.append(research)
                return results
            except Exception as e:
                logger.error(f"Redis scan error: {str(e)}")

        # Fallback to in-memory
        for research in self.fallback_store.values():
            if research.get("user_id") == user_id:
                results.append(research)
                if len(results) >= limit:
                    break

        return results

    def values(self):
        """Get all values - for backward compatibility"""
        if self.use_redis:
            logger.warning("values() method not efficient with Redis - use get_user_research instead")
        return self.fallback_store.values()

    def __contains__(self, research_id: str) -> bool:
        """Check if research_id exists - for backward compatibility"""
        # This is synchronous for backward compatibility
        # For async usage, use exists() method
        if self.use_redis:
            logger.warning("Synchronous contains check not available with Redis")
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
