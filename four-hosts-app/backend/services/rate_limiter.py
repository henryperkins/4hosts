"""
Rate Limiting Service for Four Hosts Research API
Phase 5: Production-Ready Features
"""

import time
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Tuple
from collections import defaultdict, deque
import redis
import os
import re

from utils.async_utils import run_in_thread
from utils.security import ip_validator, token_validator
import json
import logging
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

try:
    # Heavy server-side auth dependencies; may not be available in simple client-only contexts
    from services.auth_service import UserRole, RATE_LIMITS, get_api_key_info, decode_token
except Exception:  # pragma: no cover - fallback for lightweight imports
    from enum import Enum

    class UserRole(str, Enum):  # minimal stub for client-side contexts
        FREE = "free"

    RATE_LIMITS = {
        UserRole.FREE: {
            "concurrent_requests": 5,
            "requests_per_minute": 60,
            "requests_per_hour": 3600,
            "requests_per_day": 100000,
        }
    }

    async def get_api_key_info(*_args, **_kwargs):  # type: ignore[empty-body]
        return None

    async def decode_token(*_args, **_kwargs):  # type: ignore[empty-body]
        return {}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimitExceeded(HTTPException):
    """Custom exception for rate limit exceeded"""

    def __init__(self, retry_after: int, limit_type: str, limit: int):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Rate limit exceeded",
                "limit_type": limit_type,
                "limit": limit,
                "retry_after": retry_after,
            },
            headers={"Retry-After": str(retry_after)},
        )


class RateLimiter:
    """
    Token bucket algorithm implementation for rate limiting
    Supports multiple time windows and concurrent request limits
    """

    def __init__(self, redis_url: Optional[str] = None):
        """Initialize rate limiter with optional Redis backend"""
        self.redis_url = redis_url
        self.redis_client = None

        # In-memory fallback storage
        self.buckets: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.request_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10000)
        )
        self.concurrent_requests: Dict[str, int] = defaultdict(int)

        # Try to connect to Redis if URL provided
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("Connected to Redis for rate limiting")
            except Exception as e:
                logger.warning(
                    f"Failed to connect to Redis: {e}. Using in-memory storage."
                )
                self.redis_client = None

    async def check_rate_limit(
        self, identifier: str, role: UserRole, request_type: str = "api"
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if request is within rate limits
        Returns: (is_allowed, limit_info)
        """
        limits = RATE_LIMITS.get(role, RATE_LIMITS[UserRole.FREE])
        current_time = time.time()

        # Check concurrent requests
        if not await self._check_concurrent_limit(
            identifier, limits["concurrent_requests"]
        ):
            return False, {
                "limit_type": "concurrent_requests",
                "limit": limits["concurrent_requests"],
                "retry_after": 1,
            }

        # Check per-minute limit
        if not await self._check_time_window_limit(
            identifier, "minute", 60, limits["requests_per_minute"], current_time
        ):
            return False, {
                "limit_type": "requests_per_minute",
                "limit": limits["requests_per_minute"],
                "retry_after": 60,
            }

        # Check per-hour limit
        if not await self._check_time_window_limit(
            identifier, "hour", 3600, limits["requests_per_hour"], current_time
        ):
            return False, {
                "limit_type": "requests_per_hour",
                "limit": limits["requests_per_hour"],
                "retry_after": 3600,
            }

        # Check per-day limit
        if not await self._check_time_window_limit(
            identifier, "day", 86400, limits["requests_per_day"], current_time
        ):
            return False, {
                "limit_type": "requests_per_day",
                "limit": limits["requests_per_day"],
                "retry_after": 86400,
            }

        # All checks passed
        await self._record_request(identifier, current_time)
        return True, None

    async def _check_concurrent_limit(self, identifier: str, limit: int) -> bool:
        """Check concurrent request limit"""
        if self.redis_client:
            key = f"concurrent:{identifier}"
            current = await self._redis_get_int(key, 0)
            return current < limit
        else:
            return self.concurrent_requests[identifier] < limit

    async def _check_time_window_limit(
        self,
        identifier: str,
        window_name: str,
        window_seconds: int,
        limit: int,
        current_time: float,
    ) -> bool:
        """Check rate limit for a specific time window"""
        if self.redis_client:
            return await self._check_redis_window_limit(
                identifier, window_name, window_seconds, limit, current_time
            )
        else:
            return self._check_memory_window_limit(
                identifier, window_name, window_seconds, limit, current_time
            )

    def _check_memory_window_limit(
        self,
        identifier: str,
        window_name: str,
        window_seconds: int,
        limit: int,
        current_time: float,
    ) -> bool:
        """Check rate limit using in-memory storage"""
        bucket_key = f"{identifier}:{window_name}"
        bucket = self.buckets[bucket_key]

        # Initialize bucket if needed
        if "tokens" not in bucket:
            bucket["tokens"] = limit
            bucket["last_refill"] = current_time

        # Refill tokens based on time passed
        time_passed = current_time - bucket["last_refill"]
        tokens_to_add = (time_passed / window_seconds) * limit
        bucket["tokens"] = min(limit, bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = current_time

        # Check if token available
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True

        return False

    async def _check_redis_window_limit(
        self,
        identifier: str,
        window_name: str,
        window_seconds: int,
        limit: int,
        current_time: float,
    ) -> bool:
        """Check rate limit using Redis sliding window"""
        def _sync_op() -> bool:
            key_local = f"rate_limit:{identifier}:{window_name}"
            window_start_local = current_time - window_seconds

            pipe_local = self.redis_client.pipeline()
            pipe_local.zremrangebyscore(key_local, 0, window_start_local)
            pipe_local.zcard(key_local)
            pipe_local.zadd(key_local, {str(current_time): current_time})
            pipe_local.expire(key_local, window_seconds + 60)
            results_local = pipe_local.execute()
            count_local = results_local[1]

            if count_local >= limit:
                self.redis_client.zrem(key_local, str(current_time))
                return False

            return True

        return await run_in_thread(_sync_op)

    async def _record_request(self, identifier: str, timestamp: float):
        """Record a request for tracking"""
        self.request_history[identifier].append(timestamp)

        if self.redis_client:
            # Record in Redis for distributed tracking
            key = f"request_history:{identifier}"
            await run_in_thread(self.redis_client.lpush, key, timestamp)
            await run_in_thread(self.redis_client.ltrim, key, 0, 999)  # Keep last 1000
            await run_in_thread(self.redis_client.expire, key, 86400)  # 24 hour expiry

    async def increment_concurrent(self, identifier: str):
        """Increment concurrent request counter"""
        if self.redis_client:
            key = f"concurrent:{identifier}"
            await run_in_thread(self.redis_client.incr, key)
            await run_in_thread(self.redis_client.expire, key, 300)  # 5 minute expiry
        else:
            self.concurrent_requests[identifier] += 1

    async def decrement_concurrent(self, identifier: str):
        """Decrement concurrent request counter"""
        if self.redis_client:
            key = f"concurrent:{identifier}"
            current = await self._redis_get_int(key, 0)
            if current > 0:
                await run_in_thread(self.redis_client.decr, key)
        else:
            if self.concurrent_requests[identifier] > 0:
                self.concurrent_requests[identifier] -= 1

    async def _redis_get_int(self, key: str, default: int = 0) -> int:
        """Get integer value from Redis with default"""
        value = await run_in_thread(self.redis_client.get, key)
        return int(value) if value else default

    def get_usage_stats(self, identifier: str, role: UserRole) -> Dict[str, Any]:
        """Get current usage statistics for an identifier"""
        limits = RATE_LIMITS.get(role, RATE_LIMITS[UserRole.FREE])
        current_time = time.time()

        stats = {"limits": limits, "usage": {}}

        # Calculate usage for each time window
        for window_name, window_seconds, limit_key in [
            ("minute", 60, "requests_per_minute"),
            ("hour", 3600, "requests_per_hour"),
            ("day", 86400, "requests_per_day"),
        ]:
            if self.redis_client:
                key = f"rate_limit:{identifier}:{window_name}"
                window_start = current_time - window_seconds
                count = self.redis_client.zcount(key, window_start, current_time)
            else:
                # Count from in-memory history
                window_start = current_time - window_seconds
                count = sum(
                    1 for ts in self.request_history[identifier] if ts > window_start
                )

            stats["usage"][limit_key] = {
                "used": count,
                "limit": limits[limit_key],
                "remaining": max(0, limits[limit_key] - count),
                "reset_in": window_seconds,
            }

        # Add concurrent usage
        if self.redis_client:
            concurrent = self.redis_client.get(f"concurrent:{identifier}") or 0
        else:
            concurrent = self.concurrent_requests[identifier]

        stats["usage"]["concurrent_requests"] = {
            "used": int(concurrent),
            "limit": limits["concurrent_requests"],
            "remaining": max(0, limits["concurrent_requests"] - int(concurrent)),
        }

        return stats


# --- Rate Limiting Middleware ---


class RateLimitMiddleware:
    """FastAPI middleware for rate limiting"""

    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.enforce_identifier = os.getenv("RATE_LIMIT_ENFORCE_IDENTIFIER", "1").lower() in ("1", "true", "yes")
        self.default_anon_limit = int(os.getenv("RATE_LIMIT_ANON_RPM", "10"))

    async def __call__(self, request: Request, call_next):
        """Process request with rate limiting"""
        # Skip rate limiting for OPTIONS requests
        if request.method == "OPTIONS":
            return await call_next(request)

        # Skip rate limiting for certain paths
        skip_paths = [
            "/health",
            "/ready",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/",
            "/auth/register",
            "/auth/login",
            "/auth/refresh",
            "/auth/user",
            "/api/csrf-token",  # CSRF token endpoint must be accessible without auth
        ]
        if request.url.path in skip_paths:
            return await call_next(request)

        # Extract identifier (API key or user ID from auth)
        identifier = await self._extract_identifier(request)
        if not identifier:
            if self.enforce_identifier:
                # Reject requests without valid identifier
                return JSONResponse(
                    status_code=400,
                    content={"error": "Authentication required for API access"}
                )
            else:
                # Allow with strict anonymous limits
                client_ip = self._get_client_ip(request)
                if not client_ip:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Unable to identify request source"}
                    )
                identifier = f"anon:{client_ip}"

        # Extract role from auth
        role = await self._extract_role(request) if identifier and not identifier.startswith("anon:") else UserRole.FREE

        # Check rate limit
        is_allowed, limit_info = await self.rate_limiter.check_rate_limit(
            identifier, role, "api"
        )

        if not is_allowed:
            raise RateLimitExceeded(
                retry_after=limit_info["retry_after"],
                limit_type=limit_info["limit_type"],
                limit=limit_info["limit"],
            )

        # Track concurrent request
        await self.rate_limiter.increment_concurrent(identifier)

        try:
            # Add rate limit headers to response
            response = await call_next(request)

            # Add rate limit info headers
            stats = self.rate_limiter.get_usage_stats(identifier, role)
            response.headers["X-RateLimit-Limit"] = str(
                stats["limits"]["requests_per_minute"]
            )
            response.headers["X-RateLimit-Remaining"] = str(
                stats["usage"]["requests_per_minute"]["remaining"]
            )
            response.headers["X-RateLimit-Reset"] = str(
                int(time.time()) + stats["usage"]["requests_per_minute"]["reset_in"]
            )

            return response
        finally:
            # Always decrement concurrent counter
            await self.rate_limiter.decrement_concurrent(identifier)

    async def _extract_identifier(self, request: Request) -> Optional[str]:
        """Extract identifier from request (API key or user ID)"""
        try:
            # Priority 1: Check request state for already authenticated user
            if hasattr(request.state, "user") and request.state.user:
                user_id = getattr(request.state.user, "id", None)
                if user_id:
                    return f"user:{user_id}"

            # Priority 2: Check for API key in header
            api_key = request.headers.get("X-API-Key", "").strip()
            if api_key and len(api_key) < 256:  # Sanity check
                try:
                    api_key_info = await get_api_key_info(api_key)
                    if api_key_info:
                        # Store API key info in request state for role extraction
                        request.state.api_key_info = api_key_info
                        return f"api_key:{api_key_info.user_id}"
                except Exception as e:
                    logger.warning(f"API key validation error: {e}")

            # Priority 3: Check for Bearer token
            auth_header = request.headers.get("Authorization", "").strip()
            token = token_validator.validate_bearer_header(auth_header)
            if token:
                try:
                    from services.auth_service import decode_token
                    payload = await decode_token(token)
                    # Store token data in request state for role extraction
                    request.state.token_data = payload
                    user_id = payload.get("user_id")
                    if user_id:
                        return f"user:{user_id}"
                except Exception as e:
                    logger.debug(f"Bearer token validation error: {e}")

            # No valid authentication found
            return None

        except Exception as e:
            logger.error(f"Error extracting identifier: {e}")
            return None

    def _get_client_ip(self, request: Request) -> Optional[str]:
        """Extract client IP address from request"""
        try:
            # Check X-Forwarded-For header with validation
            forwarded = request.headers.get("X-Forwarded-For", "").strip()
            if forwarded:
                extracted_ip = ip_validator.extract_from_forwarded(forwarded)
                if extracted_ip:
                    return extracted_ip

            # Check X-Real-IP header
            real_ip = request.headers.get("X-Real-IP", "").strip()
            if real_ip and ip_validator.validate_ip(real_ip):
                return real_ip

            # Fall back to direct client IP
            if request.client and request.client.host:
                client_host = str(request.client.host)
                if ip_validator.validate_ip(client_host):
                    return client_host

            return None

        except Exception as e:
            logger.error(f"Error extracting client IP: {e}")
            return None

    async def _extract_role(self, request: Request) -> UserRole:
        """Extract user role from request"""
        # Check if we have API key info
        if hasattr(request.state, "api_key_info"):
            return request.state.api_key_info.role

        # Check if we have token data
        if hasattr(request.state, "token_data"):
            role_str = request.state.token_data.get("role", "free")
            return UserRole(role_str)

        # Default to free tier
        return UserRole.FREE


# --- Adaptive Rate Limiting ---


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter that adjusts limits based on system load
    """

    def __init__(self, base_limiter: RateLimiter):
        self.base_limiter = base_limiter
        self.load_factor = 1.0  # 1.0 = normal, <1.0 = restricted, >1.0 = relaxed
        self.metrics = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "response_time_p95": 0.0,
            "error_rate": 0.0,
        }

    async def update_metrics(self, metrics: Dict[str, float]):
        """Update system metrics for adaptive limiting"""
        self.metrics.update(metrics)

        # Calculate load factor based on metrics
        if self.metrics["cpu_usage"] > 80 or self.metrics["memory_usage"] > 85:
            self.load_factor = 0.5  # Restrict to 50%
        elif self.metrics["response_time_p95"] > 2000:  # 2 seconds
            self.load_factor = 0.7  # Restrict to 70%
        elif self.metrics["error_rate"] > 0.05:  # 5% error rate
            self.load_factor = 0.8  # Restrict to 80%
        else:
            self.load_factor = 1.0  # Normal operation

        logger.info(f"Adaptive rate limit factor: {self.load_factor}")

    async def check_rate_limit(
        self, identifier: str, role: UserRole, request_type: str = "api"
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check rate limit with adaptive adjustment"""
        # Get base limits and adjust
        limits = RATE_LIMITS.get(role, RATE_LIMITS[UserRole.FREE])
        adjusted_limits = {
            k: int(v * self.load_factor) if isinstance(v, (int, float)) else v
            for k, v in limits.items()
        }

        # Temporarily override limits
        original_limits = RATE_LIMITS[role]
        RATE_LIMITS[role] = adjusted_limits

        try:
            return await self.base_limiter.check_rate_limit(
                identifier, role, request_type
            )
        finally:
            # Restore original limits
            RATE_LIMITS[role] = original_limits


# --- Cost-Based Rate Limiting ---


class CostBasedRateLimiter:
    """
    Rate limiter based on operation costs
    Different operations consume different amounts of quota
    """

    OPERATION_COSTS = {
        "search": 1,
        "classify": 1,
        "research_simple": 5,
        "research_standard": 10,
        "research_deep": 25,
        "synthesis": 15,
        "export_pdf": 5,
        "export_data": 2,
    }

    def __init__(self, base_limiter: RateLimiter):
        self.base_limiter = base_limiter
        self.quotas: Dict[str, Dict[str, float]] = defaultdict(dict)

    def get_operation_cost(self, operation: str, params: Dict[str, Any] = None) -> int:
        """Calculate cost for an operation"""
        base_cost = self.OPERATION_COSTS.get(operation, 1)

        # Adjust cost based on parameters
        if operation == "research_simple" and params:
            if params.get("max_sources", 0) > 100:
                base_cost *= 1.5
            if params.get("include_secondary", False):
                base_cost *= 1.2

        return int(base_cost)

    async def check_quota(
        self,
        identifier: str,
        role: UserRole,
        operation: str,
        params: Dict[str, Any] = None,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check if user has enough quota for operation"""
        cost = self.get_operation_cost(operation, params)
        limits = RATE_LIMITS.get(role, RATE_LIMITS[UserRole.FREE])

        # Get daily quota (simplified: requests_per_day)
        daily_quota = limits["requests_per_day"]

        # Check current usage
        usage_key = f"{identifier}:daily"
        current_usage = self.quotas.get(usage_key, {}).get("used", 0)

        if current_usage + cost > daily_quota:
            return False, {
                "quota_exceeded": True,
                "operation": operation,
                "cost": cost,
                "remaining_quota": daily_quota - current_usage,
                "reset_time": self._get_next_reset_time(),
            }

        # Deduct quota
        if usage_key not in self.quotas:
            self.quotas[usage_key] = {
                "used": 0,
                "reset_at": self._get_next_reset_time(),
            }

        self.quotas[usage_key]["used"] += cost

        return True, {
            "operation": operation,
            "cost": cost,
            "remaining_quota": daily_quota - self.quotas[usage_key]["used"],
        }

    def _get_next_reset_time(self) -> datetime:
        """Get next daily reset time (midnight UTC)"""
        now = datetime.now(timezone.utc)
        tomorrow = now + timedelta(days=1)
        return tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)


# --- Client-Side Token Bucket (for outbound calls) ---


class ClientRateLimiter:
    """
    Lightweight async token-bucket limiter for client/outbound usage.

    Purpose:
    - Provide simple, per-process smoothing for outbound provider calls
      (e.g., search/fetch) without touching the production API limiter.
    - Avoid confusion with the server-side RateLimiter by using a distinct name.

    Notes:
    - Uses `asyncio` and in-process state only (no Redis, no headers).
    - Tokens refill continuously; capacity defaults to `calls_per_minute`.
    - `.wait()` blocks until 1 token is available (configurable via `tokens`).
    """

    def __init__(
        self,
        calls_per_minute: int = 60,
        *,
        burst: Optional[int] = None,
        clock: Optional[callable] = None,
    ) -> None:
        if calls_per_minute <= 0:
            raise ValueError("calls_per_minute must be > 0")
        self._rate_per_sec: float = calls_per_minute / 60.0
        self._capacity: float = float(burst if burst is not None else calls_per_minute)
        self._tokens: float = self._capacity
        self._clock = clock or time.monotonic
        self._updated: float = self._clock()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._updated)
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate_per_sec)
            self._updated = now

    async def wait(self, tokens: float = 1.0) -> None:
        if tokens <= 0:
            return
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                needed = tokens - self._tokens
                # Compute sleep outside the lock to let other coroutines progress
                sleep_for = needed / self._rate_per_sec if self._rate_per_sec > 0 else 0.05
            await asyncio.sleep(max(0.001, sleep_for))
