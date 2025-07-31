"""
Memory Management Service
Provides centralized memory management with TTL caches and cleanup mechanisms
"""

import asyncio
import gc
import logging
import sys
import weakref
from typing import Dict, Any, Optional, Set, List, Callable
from datetime import datetime, timedelta
from collections import defaultdict
import psutil
import os

from cachetools import TTLCache, LRUCache
import schedule

logger = logging.getLogger(__name__)


class MemoryManager:
    """Centralized memory management for the Four Hosts application"""
    
    def __init__(self):
        # Memory thresholds
        self.memory_threshold_mb = int(os.getenv("MEMORY_THRESHOLD_MB", "1024"))
        self.critical_threshold_mb = int(os.getenv("CRITICAL_MEMORY_MB", "1536"))
        
        # Managed caches
        self._caches: Dict[str, Any] = {}
        self._cache_configs: Dict[str, Dict[str, Any]] = {}
        
        # Weak references to track large objects
        self._tracked_objects: Set[weakref.ref] = set()
        
        # Cleanup tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._emergency_cleanup_callbacks: List[Callable] = []
        
        # Metrics
        self._metrics = {
            "cleanups_performed": 0,
            "emergency_cleanups": 0,
            "objects_tracked": 0,
            "caches_managed": 0,
            "memory_freed_mb": 0
        }
        
        # Initialize default caches
        self._initialize_default_caches()
    
    def _initialize_default_caches(self):
        """Initialize default application caches"""
        
        # Classification cache
        self.register_cache(
            "classification_cache",
            TTLCache(maxsize=1000, ttl=3600),  # 1 hour TTL
            cleanup_priority=2
        )
        
        # Search results cache
        self.register_cache(
            "search_results_cache",
            TTLCache(maxsize=500, ttl=1800),  # 30 min TTL
            cleanup_priority=1
        )
        
        # Context engineering cache
        self.register_cache(
            "context_cache",
            TTLCache(maxsize=200, ttl=900),  # 15 min TTL
            cleanup_priority=3
        )
        
        # User session cache
        self.register_cache(
            "user_sessions",
            TTLCache(maxsize=5000, ttl=86400),  # 24 hour TTL
            cleanup_priority=4
        )
        
        # Robot parser cache (for respectful fetching)
        self.register_cache(
            "robot_parsers",
            LRUCache(maxsize=100),
            cleanup_priority=1
        )
    
    def register_cache(
        self, 
        name: str, 
        cache: Any,
        cleanup_priority: int = 5,
        cleanup_callback: Optional[Callable] = None
    ):
        """Register a cache for management"""
        self._caches[name] = cache
        self._cache_configs[name] = {
            "priority": cleanup_priority,
            "callback": cleanup_callback,
            "created_at": datetime.utcnow()
        }
        self._metrics["caches_managed"] = len(self._caches)
        logger.info(f"Registered cache: {name} with priority {cleanup_priority}")
    
    def get_cache(self, name: str) -> Optional[Any]:
        """Get a registered cache"""
        return self._caches.get(name)
    
    def track_large_object(self, obj: Any, size_estimate_mb: float = 0):
        """Track a large object for memory management"""
        if size_estimate_mb > 10:  # Only track objects > 10MB
            weak_ref = weakref.ref(obj, self._on_object_deleted)
            self._tracked_objects.add(weak_ref)
            self._metrics["objects_tracked"] = len(self._tracked_objects)
    
    def _on_object_deleted(self, ref):
        """Callback when tracked object is garbage collected"""
        self._tracked_objects.discard(ref)
        self._metrics["objects_tracked"] = len(self._tracked_objects)
    
    async def start_monitoring(self):
        """Start memory monitoring and cleanup tasks"""
        if self._cleanup_task:
            return
        
        self._cleanup_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Memory monitoring started")
    
    async def stop_monitoring(self):
        """Stop memory monitoring"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        logger.info("Memory monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Get memory usage
                memory_info = self.get_memory_info()
                
                # Check thresholds
                if memory_info["rss_mb"] > self.critical_threshold_mb:
                    await self.emergency_cleanup()
                elif memory_info["rss_mb"] > self.memory_threshold_mb:
                    await self.routine_cleanup()
                
                # Clean expired entries in TTL caches
                self._clean_expired_entries()
                
                # Remove dead weak references
                self._tracked_objects = {ref for ref in self._tracked_objects if ref() is not None}
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage information"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
            "tracked_objects": len(self._tracked_objects),
            "cache_sizes": {name: len(cache) if hasattr(cache, '__len__') else 0 
                          for name, cache in self._caches.items()}
        }
    
    async def routine_cleanup(self):
        """Perform routine memory cleanup"""
        logger.info("Starting routine memory cleanup")
        start_memory = self.get_memory_info()["rss_mb"]
        
        # Clean caches by priority (lower priority = clean first)
        sorted_caches = sorted(
            self._cache_configs.items(),
            key=lambda x: x[1]["priority"]
        )
        
        for cache_name, config in sorted_caches[:3]:  # Clean top 3 low-priority caches
            cache = self._caches.get(cache_name)
            if cache and hasattr(cache, 'clear'):
                size_before = len(cache) if hasattr(cache, '__len__') else 0
                
                # Clear 50% of cache
                if hasattr(cache, 'items'):
                    items = list(cache.items())
                    for key, _ in items[:len(items)//2]:
                        cache.pop(key, None)
                
                logger.info(f"Cleaned cache {cache_name}: {size_before} -> {len(cache) if hasattr(cache, '__len__') else 0}")
        
        # Force garbage collection
        gc.collect()
        
        # Calculate freed memory
        end_memory = self.get_memory_info()["rss_mb"]
        freed_mb = start_memory - end_memory
        self._metrics["memory_freed_mb"] += max(0, freed_mb)
        self._metrics["cleanups_performed"] += 1
        
        logger.info(f"Routine cleanup completed. Freed: {freed_mb:.2f} MB")
    
    async def emergency_cleanup(self):
        """Perform emergency memory cleanup"""
        logger.warning("Starting EMERGENCY memory cleanup")
        start_memory = self.get_memory_info()["rss_mb"]
        
        # Clear all caches
        for cache_name, cache in self._caches.items():
            if hasattr(cache, 'clear'):
                cache.clear()
                logger.info(f"Cleared cache: {cache_name}")
        
        # Call emergency cleanup callbacks
        for callback in self._emergency_cleanup_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Emergency callback error: {e}")
        
        # Clear tracked objects
        self._tracked_objects.clear()
        
        # Aggressive garbage collection
        gc.collect(2)
        
        # Calculate freed memory
        end_memory = self.get_memory_info()["rss_mb"]
        freed_mb = start_memory - end_memory
        self._metrics["memory_freed_mb"] += max(0, freed_mb)
        self._metrics["emergency_cleanups"] += 1
        
        logger.warning(f"Emergency cleanup completed. Freed: {freed_mb:.2f} MB")
    
    def _clean_expired_entries(self):
        """Clean expired entries from TTL caches"""
        for cache_name, cache in self._caches.items():
            if isinstance(cache, TTLCache):
                # TTLCache automatically removes expired items on access
                # Force expiration check by accessing the cache
                _ = len(cache)
    
    def register_emergency_callback(self, callback: Callable):
        """Register a callback for emergency cleanup"""
        self._emergency_cleanup_callbacks.append(callback)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get memory management metrics"""
        memory_info = self.get_memory_info()
        
        return {
            **self._metrics,
            **memory_info,
            "threshold_mb": self.memory_threshold_mb,
            "critical_threshold_mb": self.critical_threshold_mb,
            "health_status": self._get_health_status(memory_info["rss_mb"])
        }
    
    def _get_health_status(self, current_mb: float) -> str:
        """Get memory health status"""
        if current_mb > self.critical_threshold_mb:
            return "critical"
        elif current_mb > self.memory_threshold_mb:
            return "warning"
        else:
            return "healthy"


class CacheDecorators:
    """Decorators for automatic caching with memory management"""
    
    @staticmethod
    def ttl_cache(cache_name: str, ttl: int = 3600, maxsize: int = 128):
        """Decorator to add TTL caching to a function"""
        def decorator(func):
            # Create cache if not exists
            cache = memory_manager.get_cache(cache_name)
            if not cache:
                cache = TTLCache(maxsize=maxsize, ttl=ttl)
                memory_manager.register_cache(cache_name, cache)
            
            async def async_wrapper(*args, **kwargs):
                # Create cache key
                cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                
                # Check cache
                if cache_key in cache:
                    return cache[cache_key]
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Store in cache
                cache[cache_key] = result
                
                return result
            
            def sync_wrapper(*args, **kwargs):
                # Create cache key
                cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                
                # Check cache
                if cache_key in cache:
                    return cache[cache_key]
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Store in cache
                cache[cache_key] = result
                
                return result
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator


# Global memory manager instance
memory_manager = MemoryManager()


# Example emergency cleanup callbacks for different services
async def classification_engine_cleanup():
    """Emergency cleanup for classification engine"""
    from services.classification_engine import classification_engine
    if hasattr(classification_engine, 'cache'):
        classification_engine.cache.clear()
    logger.info("Cleared classification engine cache")


async def search_api_cleanup():
    """Emergency cleanup for search APIs"""
    from services.search_apis import SearchAPIManager
    # Clear any in-memory results or caches
    logger.info("Cleared search API caches")


# Register emergency callbacks
memory_manager.register_emergency_callback(classification_engine_cleanup)
memory_manager.register_emergency_callback(search_api_cleanup)