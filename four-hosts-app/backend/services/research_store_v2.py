"""
Enhanced Research Store Service V2
Provides Redis-backed research data storage with proper serialization and memory management
"""

import json
import logging
from typing import Dict, Optional, List, Any, Set
import os
from datetime import datetime, timedelta
import asyncio
from collections import OrderedDict
import weakref

import redis.asyncio as redis
from cachetools import TTLCache
import pickle

from models.context_models import (
    ResearchRequestSchema, ClassificationResultSchema, 
    ResearchResultSchema, ResearchStatus
)

logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe LRU cache with TTL support"""
    
    def __init__(self, maxsize: int = 1000, ttl: int = 3600):
        self._cache = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl
        self._lock = asyncio.Lock()
        self._access_times = {}
    
    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key in self._cache:
                # Check TTL
                if (datetime.utcnow() - self._access_times[key]).seconds > self._ttl:
                    del self._cache[key]
                    del self._access_times[key]
                    return None
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return self._cache[key]
            return None
    
    async def set(self, key: str, value: Any):
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._maxsize:
                    # Remove oldest
                    oldest = next(iter(self._cache))
                    del self._cache[oldest]
                    del self._access_times[oldest]
                
            self._cache[key] = value
            self._access_times[key] = datetime.utcnow()
    
    async def delete(self, key: str):
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._access_times[key]
    
    async def clear_expired(self):
        """Remove expired entries"""
        async with self._lock:
            now = datetime.utcnow()
            expired_keys = [
                k for k, t in self._access_times.items()
                if (now - t).seconds > self._ttl
            ]
            for key in expired_keys:
                del self._cache[key]
                del self._access_times[key]


class ResearchStoreV2:
    """Enhanced Research data storage with proper serialization and memory management"""
    
    def __init__(self, redis_url: str = None, enable_cache: bool = True):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_client = None
        self.key_prefix = "research:v2:"
        self.use_redis = False
        
        # Memory management
        self.enable_cache = enable_cache
        self._local_cache = LRUCache(maxsize=500, ttl=900) if enable_cache else None
        
        # Cleanup tracking
        self._active_research: Set[str] = set()
        self._cleanup_task = None
        
        # Metrics
        self._metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "redis_errors": 0,
            "serialization_errors": 0
        }
    
    async def initialize(self):
        """Initialize Redis connection with fallback to in-memory"""
        try:
            self.redis_client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False  # We'll handle decoding
            )
            await self.redis_client.ping()
            self.use_redis = True
            logger.info("✓ Redis research store V2 initialized")
            
            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            
        except Exception as e:
            logger.warning(f"Redis unavailable: {str(e)}")
            self.use_redis = False
    
    async def close(self):
        """Clean shutdown"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        if self.redis_client:
            await self.redis_client.close()
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of expired data"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Clean local cache
                if self._local_cache:
                    await self._local_cache.clear_expired()
                
                # Clean completed research older than 24 hours
                if self.use_redis:
                    await self._cleanup_old_research()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    async def _cleanup_old_research(self):
        """Remove old completed research from Redis"""
        try:
            pattern = f"{self.key_prefix}*"
            async for key in self.redis_client.scan_iter(pattern):
                data = await self.redis_client.get(key)
                if data:
                    try:
                        research = json.loads(data)
                        created_at = datetime.fromisoformat(research.get("created_at", ""))
                        status = research.get("status", "")
                        
                        # Remove if completed and older than 24 hours
                        if status in ["completed", "failed", "cancelled"]:
                            if (datetime.utcnow() - created_at) > timedelta(hours=24):
                                await self.redis_client.delete(key)
                                
                    except Exception:
                        pass
                        
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    async def store_research_request(self, request: ResearchRequestSchema) -> str:
        """Store a new research request"""
        research_id = request.id
        self._active_research.add(research_id)
        
        # Serialize with schema
        data = request.to_redis_dict()
        
        # Store in cache
        if self._local_cache:
            await self._local_cache.set(research_id, data)
        
        # Store in Redis
        if self.use_redis:
            try:
                key = f"{self.key_prefix}{research_id}"
                serialized = json.dumps(data, default=str)
                await self.redis_client.set(
                    key, 
                    serialized, 
                    ex=86400  # 24 hour TTL
                )
            except Exception as e:
                self._metrics["redis_errors"] += 1
                logger.error(f"Redis store error: {e}")
        
        return research_id
    
    async def store_classification(
        self, 
        research_id: str, 
        classification: ClassificationResultSchema
    ):
        """Store classification result with research"""
        data = await self.get_research(research_id)
        if data:
            data["classification"] = classification.to_redis_dict()
            await self._update_research(research_id, data)
    
    async def get_research(self, research_id: str) -> Optional[Dict[str, Any]]:
        """Get research data with proper deserialization"""
        # Check cache first
        if self._local_cache:
            cached = await self._local_cache.get(research_id)
            if cached:
                self._metrics["cache_hits"] += 1
                return cached
        
        self._metrics["cache_misses"] += 1
        
        # Get from Redis
        if self.use_redis:
            try:
                key = f"{self.key_prefix}{research_id}"
                data = await self.redis_client.get(key)
                if data:
                    deserialized = json.loads(data)
                    
                    # Update cache
                    if self._local_cache:
                        await self._local_cache.set(research_id, deserialized)
                    
                    return deserialized
                    
            except Exception as e:
                self._metrics["redis_errors"] += 1
                logger.error(f"Redis get error: {e}")
        
        return None
    
    async def update_status(
        self, 
        research_id: str, 
        status: ResearchStatus,
        error_message: Optional[str] = None
    ):
        """Update research status"""
        data = await self.get_research(research_id)
        if data:
            data["status"] = status.value
            data["updated_at"] = datetime.utcnow().isoformat()
            if error_message:
                data["error_message"] = error_message
            
            await self._update_research(research_id, data)
            
            # Remove from active if terminal state
            if status in [ResearchStatus.COMPLETED, ResearchStatus.FAILED, ResearchStatus.CANCELLED]:
                self._active_research.discard(research_id)
    
    async def _update_research(self, research_id: str, data: Dict[str, Any]):
        """Update research data"""
        # Update cache
        if self._local_cache:
            await self._local_cache.set(research_id, data)
        
        # Update Redis
        if self.use_redis:
            try:
                key = f"{self.key_prefix}{research_id}"
                serialized = json.dumps(data, default=str)
                
                # Get remaining TTL
                ttl = await self.redis_client.ttl(key)
                if ttl > 0:
                    await self.redis_client.set(key, serialized, ex=ttl)
                else:
                    await self.redis_client.set(key, serialized, ex=86400)
                    
            except Exception as e:
                self._metrics["redis_errors"] += 1
                logger.error(f"Redis update error: {e}")
    
    async def store_result(self, result: ResearchResultSchema):
        """Store complete research result"""
        research_id = result.research_id
        
        # Get existing data
        data = await self.get_research(research_id)
        if not data:
            data = {"id": research_id}
        
        # Add result
        data["result"] = result.to_api_response(include_debug=True)
        data["status"] = ResearchStatus.COMPLETED.value
        data["completed_at"] = datetime.utcnow().isoformat()
        
        await self._update_research(research_id, data)
    
    async def get_user_research(
        self, 
        user_id: str, 
        limit: int = 50,
        status_filter: Optional[List[ResearchStatus]] = None
    ) -> List[Dict[str, Any]]:
        """Get research history for a user"""
        results = []
        
        if self.use_redis:
            try:
                pattern = f"{self.key_prefix}*"
                async for key in self.redis_client.scan_iter(pattern):
                    if len(results) >= limit:
                        break
                    
                    data = await self.redis_client.get(key)
                    if data:
                        try:
                            research = json.loads(data)
                            
                            # Check user and status
                            user_match = research.get("user_context", {}).get("user_id") == user_id
                            status_match = True
                            if status_filter:
                                status_match = research.get("status") in [s.value for s in status_filter]
                            
                            if user_match and status_match:
                                results.append(research)
                                
                        except Exception:
                            pass
                            
            except Exception as e:
                logger.error(f"Redis scan error: {e}")
        
        # Sort by created_at descending
        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return results[:limit]
    
    async def exists(self, research_id: str) -> bool:
        """Check if research exists"""
        # Check cache first
        if self._local_cache:
            cached = await self._local_cache.get(research_id)
            if cached:
                return True
        
        # Check Redis
        if self.use_redis:
            try:
                key = f"{self.key_prefix}{research_id}"
                return bool(await self.redis_client.exists(key))
            except Exception:
                pass
        
        return False
    
    async def delete(self, research_id: str):
        """Delete research data"""
        # Remove from cache
        if self._local_cache:
            await self._local_cache.delete(research_id)
        
        # Remove from Redis
        if self.use_redis:
            try:
                key = f"{self.key_prefix}{research_id}"
                await self.redis_client.delete(key)
            except Exception as e:
                logger.error(f"Redis delete error: {e}")
        
        # Remove from active set
        self._active_research.discard(research_id)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get store metrics"""
        return {
            **self._metrics,
            "active_research_count": len(self._active_research),
            "cache_enabled": self.enable_cache,
            "redis_enabled": self.use_redis
        }


# Singleton instance
research_store_v2 = ResearchStoreV2()