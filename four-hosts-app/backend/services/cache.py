"""
Redis-based caching layer for Four Hosts Research Application
Implements intelligent caching for search results, API responses, and paradigm classifications
"""

import redis.asyncio as redis
import json
import logging
import hashlib
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
from dataclasses import asdict
import asyncio
from contextlib import asynccontextmanager

from .search_apis import SearchResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheManager:
    """Manages Redis caching with intelligent TTL and cost optimization"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_pool = None
        self.hit_count = 0
        self.miss_count = 0

        # TTL configurations (in seconds)
        self.ttl_config = {
            "search_results": 24 * 3600,  # 24 hours
            "paradigm_classification": 7 * 24 * 3600,  # 7 days
            "source_credibility": 30 * 24 * 3600,  # 30 days
            "api_cost_tracking": 24 * 3600,  # 24 hours
            "user_preferences": 30 * 24 * 3600,  # 30 days
        }

    async def initialize(self):
        """Initialize Redis connection pool"""
        try:
            self.redis_pool = redis.ConnectionPool.from_url(
                self.redis_url, max_connections=20, decode_responses=True
            )
            # Test connection
            redis_client = redis.Redis(connection_pool=self.redis_pool)
            await redis_client.ping()
            logger.info("Redis connection established successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            return False

    async def close(self):
        """Close Redis connection pool"""
        if self.redis_pool:
            await self.redis_pool.disconnect()

    @asynccontextmanager
    async def get_client(self):
        """Context manager for Redis client"""
        if not self.redis_pool:
            await self.initialize()

        client = redis.Redis(connection_pool=self.redis_pool)
        try:
            yield client
        finally:
            await client.aclose()

    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a consistent cache key"""
        # Combine all arguments into a single string
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
        key_string = ":".join(key_parts)

        # Create hash for long keys
        if len(key_string) > 200:
            key_hash = hashlib.md5(key_string.encode()).hexdigest()
            return f"{prefix}:hash:{key_hash}"

        return f"{prefix}:{key_string}"

    async def get_search_results(
        self, query: str, config_dict: Dict[str, Any], paradigm: str
    ) -> Optional[List[SearchResult]]:
        """Get cached search results"""
        cache_key = self._generate_cache_key("search", query, paradigm, **config_dict)

        try:
            async with self.get_client() as client:
                cached_data = await client.get(cache_key)

                if cached_data:
                    self.hit_count += 1
                    results_data = json.loads(cached_data)

                    # Convert back to SearchResult objects
                    results = []
                    for result_dict in results_data:
                        # Handle datetime conversion
                        if result_dict.get("published_date"):
                            result_dict["published_date"] = datetime.fromisoformat(
                                result_dict["published_date"].replace("Z", "+00:00")
                            )
                        results.append(SearchResult(**result_dict))

                    logger.info(f"Cache HIT for search: {query[:50]}...")
                    return results
                else:
                    self.miss_count += 1
                    logger.info(f"Cache MISS for search: {query[:50]}...")
                    return None

        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            return None

    async def set_search_results(
        self,
        query: str,
        config_dict: Dict[str, Any],
        paradigm: str,
        results: List[SearchResult],
    ):
        """Cache search results"""
        cache_key = self._generate_cache_key("search", query, paradigm, **config_dict)

        try:
            # Convert SearchResult objects to dict for JSON serialization
            results_data = []
            for result in results:
                result_dict = asdict(result)
                # Handle datetime serialization
                if result_dict.get("published_date"):
                    result_dict["published_date"] = result_dict[
                        "published_date"
                    ].isoformat()
                results_data.append(result_dict)

            async with self.get_client() as client:
                await client.setex(
                    cache_key,
                    self.ttl_config["search_results"],
                    json.dumps(results_data, default=str),
                )

                logger.info(
                    f"Cached {len(results)} search results for: {query[:50]}..."
                )

        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")

    async def get_paradigm_classification(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached paradigm classification"""
        cache_key = self._generate_cache_key("paradigm", query)

        try:
            async with self.get_client() as client:
                cached_data = await client.get(cache_key)

                if cached_data:
                    self.hit_count += 1
                    logger.info(f"Cache HIT for paradigm: {query[:50]}...")
                    return json.loads(cached_data)
                else:
                    self.miss_count += 1
                    return None

        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            return None

    async def set_paradigm_classification(
        self, query: str, classification: Dict[str, Any]
    ):
        """Cache paradigm classification"""
        cache_key = self._generate_cache_key("paradigm", query)

        try:
            async with self.get_client() as client:
                await client.setex(
                    cache_key,
                    self.ttl_config["paradigm_classification"],
                    json.dumps(classification, default=str),
                )

        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")

    async def get_source_credibility(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get cached source credibility data"""
        cache_key = self._generate_cache_key("credibility", domain)

        try:
            async with self.get_client() as client:
                cached_data = await client.get(cache_key)

                if cached_data:
                    return json.loads(cached_data)
                return None

        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            return None

    async def set_source_credibility(
        self, domain: str, credibility_data: Dict[str, Any]
    ):
        """Cache source credibility data"""
        cache_key = self._generate_cache_key("credibility", domain)

        try:
            async with self.get_client() as client:
                await client.setex(
                    cache_key,
                    self.ttl_config["source_credibility"],
                    json.dumps(credibility_data, default=str),
                )

        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")

    async def track_api_cost(self, api_name: str, cost: float, calls: int = 1):
        """Track API costs for monitoring"""
        today = datetime.now().strftime("%Y-%m-%d")
        cost_key = self._generate_cache_key("cost", api_name, today)
        calls_key = self._generate_cache_key("calls", api_name, today)

        try:
            async with self.get_client() as client:
                # Increment cost and calls atomically
                pipe = client.pipeline()
                pipe.incrbyfloat(cost_key, cost)
                pipe.incr(calls_key, calls)
                pipe.expire(cost_key, self.ttl_config["api_cost_tracking"])
                pipe.expire(calls_key, self.ttl_config["api_cost_tracking"])
                await pipe.execute()

        except Exception as e:
            logger.error(f"Cost tracking error: {str(e)}")

    async def get_daily_api_costs(
        self, date: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Get API costs for a specific date"""
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        try:
            async with self.get_client() as client:
                # Get all cost keys for the date
                cost_pattern = f"cost:*:{date}"
                calls_pattern = f"calls:*:{date}"

                cost_keys = await client.keys(cost_pattern)
                calls_keys = await client.keys(calls_pattern)

                results = {}

                # Get costs
                if cost_keys:
                    cost_values = await client.mget(*cost_keys)
                    for key, value in zip(cost_keys, cost_values):
                        api_name = key.split(":")[1]  # Extract API name
                        if api_name not in results:
                            results[api_name] = {}
                        results[api_name]["cost"] = float(value) if value else 0.0

                # Get call counts
                if calls_keys:
                    call_values = await client.mget(*calls_keys)
                    for key, value in zip(calls_keys, call_values):
                        api_name = key.split(":")[1]  # Extract API name
                        if api_name not in results:
                            results[api_name] = {}
                        results[api_name]["calls"] = int(value) if value else 0

                return results

        except Exception as e:
            logger.error(f"Cost retrieval error: {str(e)}")
            return {}

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0

        try:
            async with self.get_client() as client:
                info = await client.info("memory")

                return {
                    "hit_count": self.hit_count,
                    "miss_count": self.miss_count,
                    "hit_rate_percent": round(hit_rate, 2),
                    "memory_used": info.get("used_memory_human", "Unknown"),
                    "connected_clients": info.get("connected_clients", 0),
                }
        except Exception as e:
            logger.error(f"Stats retrieval error: {str(e)}")
            return {
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_rate_percent": round(hit_rate, 2),
                "error": str(e),
            }

    async def clear_expired_keys(self, pattern: str = "*"):
        """Clear expired keys (maintenance function)"""
        try:
            async with self.get_client() as client:
                keys = await client.keys(pattern)
                deleted = 0

                for key in keys:
                    ttl = await client.ttl(key)
                    if ttl == -1:  # Key exists but has no expiry
                        # Set a default expiry based on key type
                        key_type = key.split(":")[0]
                        if key_type in self.ttl_config:
                            await client.expire(key, self.ttl_config[key_type])
                    elif ttl == -2:  # Key doesn't exist
                        deleted += 1

                logger.info(
                    f"Maintenance: processed {len(keys)} keys, {deleted} were expired"
                )
                return {"processed": len(keys), "expired": deleted}

        except Exception as e:
            logger.error(f"Maintenance error: {str(e)}")
            return {"error": str(e)}

    async def invalidate_search_cache(self, query_pattern: str = "*"):
        """Invalidate search cache for specific patterns"""
        try:
            async with self.get_client() as client:
                pattern = f"search:*{query_pattern}*"
                keys = await client.keys(pattern)

                if keys:
                    deleted = await client.delete(*keys)
                    logger.info(f"Invalidated {deleted} search cache entries")
                    return deleted
                return 0

        except Exception as e:
            logger.error(f"Cache invalidation error: {str(e)}")
            return 0


# Global cache manager instance
cache_manager = CacheManager()


async def initialize_cache():
    """Initialize the global cache manager"""
    success = await cache_manager.initialize()
    if success:
        logger.info("Cache system initialized successfully")
    else:
        logger.warning("Cache system initialization failed - running without cache")
    return success


async def cleanup_cache():
    """Cleanup cache connections"""
    await cache_manager.close()


# Convenience functions for easy use
async def get_cached_search_results(
    query: str, config_dict: Dict[str, Any], paradigm: str
) -> Optional[List[SearchResult]]:
    """Convenience function to get cached search results"""
    return await cache_manager.get_search_results(query, config_dict, paradigm)


async def cache_search_results(
    query: str, config_dict: Dict[str, Any], paradigm: str, results: List[SearchResult]
):
    """Convenience function to cache search results"""
    await cache_manager.set_search_results(query, config_dict, paradigm, results)


# Example usage and testing
async def test_cache_system():
    """Test the cache system"""
    print("Testing cache system...")

    # Initialize cache
    success = await initialize_cache()
    if not success:
        print("Cache initialization failed")
        return

    # Test paradigm classification caching
    test_query = "How can small businesses compete with Amazon?"
    test_classification = {
        "primary": "maeve",
        "secondary": "dolores",
        "confidence": 0.78,
        "distribution": {"maeve": 0.4, "dolores": 0.25, "bernard": 0.2, "teddy": 0.15},
    }

    print("Testing paradigm classification cache...")

    # Set classification
    await cache_manager.set_paradigm_classification(test_query, test_classification)

    # Get classification
    cached_classification = await cache_manager.get_paradigm_classification(test_query)

    if cached_classification:
        print("✓ Paradigm classification cached successfully")
        print(f"  Primary: {cached_classification['primary']}")
        print(f"  Confidence: {cached_classification['confidence']}")
    else:
        print("✗ Failed to retrieve cached classification")

    # Test search results caching
    print("\nTesting search results cache...")

    test_results = [
        SearchResult(
            title="Test Article 1",
            url="https://example.com/1",
            snippet="This is a test snippet",
            source="test_api",
            domain="example.com",
        ),
        SearchResult(
            title="Test Article 2",
            url="https://example.com/2",
            snippet="Another test snippet",
            source="test_api",
            domain="example.com",
        ),
    ]

    config_dict = {"max_results": 50, "language": "en", "region": "us"}

    # Cache results
    await cache_manager.set_search_results(
        test_query, config_dict, "maeve", test_results
    )

    # Retrieve results
    cached_results = await cache_manager.get_search_results(
        test_query, config_dict, "maeve"
    )

    if cached_results and len(cached_results) == 2:
        print("✓ Search results cached successfully")
        print(f"  Cached {len(cached_results)} results")
        print(f"  First result: {cached_results[0].title}")
    else:
        print("✗ Failed to retrieve cached search results")

    # Test cost tracking
    print("\nTesting cost tracking...")
    await cache_manager.track_api_cost("google", 0.05, 1)
    await cache_manager.track_api_cost("bing", 0.03, 1)

    costs = await cache_manager.get_daily_api_costs()
    if costs:
        print("✓ Cost tracking working")
        for api, data in costs.items():
            print(f"  {api}: ${data.get('cost', 0):.3f} ({data.get('calls', 0)} calls)")

    # Get cache stats
    stats = await cache_manager.get_cache_stats()
    print(f"\nCache Stats:")
    print(f"  Hit rate: {stats['hit_rate_percent']}%")
    print(f"  Memory used: {stats.get('memory_used', 'Unknown')}")

    # Cleanup
    await cleanup_cache()
    print("\nCache test completed")


if __name__ == "__main__":
    asyncio.run(test_cache_system())
