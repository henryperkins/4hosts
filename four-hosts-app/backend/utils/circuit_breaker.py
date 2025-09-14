"""Circuit Breaker pattern implementation for resilient external service calls.

Provides automatic failure detection and recovery for external APIs, databases,
and other services that may experience temporary failures.
"""

import asyncio
import time
import logging
from typing import Callable, Any, Optional, Dict, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import functools

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitStats:
    """Statistics for circuit breaker monitoring."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changes: List[tuple] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate percentage."""
        if self.total_calls == 0:
            return 0.0
        return (self.failed_calls / self.total_calls) * 100


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    Args:
        name: Identifier for this circuit breaker
        failure_threshold: Number of consecutive failures before opening
        recovery_timeout: Seconds to wait before attempting recovery
        expected_exception: Exception type(s) to catch (default: Exception)
        success_threshold: Successful calls needed to close from half-open
        failure_rate_threshold: Failure rate percentage to trigger open state
        sample_size: Number of recent calls to consider for failure rate
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
        success_threshold: int = 2,
        failure_rate_threshold: float = 50.0,
        sample_size: int = 10
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.success_threshold = success_threshold
        self.failure_rate_threshold = failure_rate_threshold
        self.sample_size = sample_size

        self.state = CircuitState.CLOSED
        self.stats = CircuitStats()
        self._half_open_calls = 0
        self._recent_results: List[bool] = []
        self._lock = asyncio.Lock()

    def _record_success(self):
        """Record a successful call."""
        self.stats.successful_calls += 1
        self.stats.total_calls += 1
        self.stats.consecutive_failures = 0
        self.stats.last_success_time = time.time()
        self._recent_results.append(True)

        if len(self._recent_results) > self.sample_size:
            self._recent_results.pop(0)

    def _record_failure(self):
        """Record a failed call."""
        self.stats.failed_calls += 1
        self.stats.total_calls += 1
        self.stats.consecutive_failures += 1
        self.stats.last_failure_time = time.time()
        self._recent_results.append(False)

        if len(self._recent_results) > self.sample_size:
            self._recent_results.pop(0)

    def _should_open(self) -> bool:
        """Check if circuit should open based on failure conditions."""
        # Check consecutive failures
        if self.stats.consecutive_failures >= self.failure_threshold:
            return True

        # Check failure rate
        if len(self._recent_results) >= self.sample_size:
            recent_failures = self._recent_results.count(False)
            failure_rate = (recent_failures / len(self._recent_results)) * 100
            if failure_rate >= self.failure_rate_threshold:
                return True

        return False

    def _change_state(self, new_state: CircuitState):
        """Change circuit state and log the transition."""
        old_state = self.state
        self.state = new_state
        self.stats.state_changes.append((time.time(), old_state, new_state))

        logger.info(
            f"Circuit breaker '{self.name}' changed state: {old_state.value} -> {new_state.value}"
        )

        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0

    async def _attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.state != CircuitState.OPEN:
            return False

        if self.stats.last_failure_time is None:
            return True

        time_since_failure = time.time() - self.stats.last_failure_time
        if time_since_failure >= self.recovery_timeout:
            async with self._lock:
                if self.state == CircuitState.OPEN:  # Double-check
                    self._change_state(CircuitState.HALF_OPEN)
                    return True

        return False

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func if successful

        Raises:
            CircuitOpenError: If circuit is open
            Original exception: If func fails and circuit allows
        """
        # Check if we should attempt recovery
        if self.state == CircuitState.OPEN:
            if not await self._attempt_reset():
                raise CircuitOpenError(
                    f"Circuit breaker '{self.name}' is open. "
                    f"Next retry in {self.recovery_timeout - (time.time() - self.stats.last_failure_time):.1f}s"
                )

        # Half-open state: limited calls allowed
        if self.state == CircuitState.HALF_OPEN:
            async with self._lock:
                if self._half_open_calls >= self.success_threshold:
                    # Already testing, reject additional calls
                    raise CircuitOpenError(
                        f"Circuit breaker '{self.name}' is testing recovery"
                    )
                self._half_open_calls += 1

        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Record success
            self._record_success()

            # Handle state transitions on success
            if self.state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.success_threshold:
                    self._change_state(CircuitState.CLOSED)

            return result

        except self.expected_exception as e:
            # Record failure
            self._record_failure()

            # Handle state transitions on failure
            if self.state == CircuitState.CLOSED:
                if self._should_open():
                    self._change_state(CircuitState.OPEN)
            elif self.state == CircuitState.HALF_OPEN:
                self._change_state(CircuitState.OPEN)

            # Re-raise the exception
            raise e

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self.stats.total_calls,
            "successful_calls": self.stats.successful_calls,
            "failed_calls": self.stats.failed_calls,
            "failure_rate": self.stats.failure_rate,
            "consecutive_failures": self.stats.consecutive_failures,
            "last_failure": datetime.fromtimestamp(self.stats.last_failure_time).isoformat()
                          if self.stats.last_failure_time else None,
            "last_success": datetime.fromtimestamp(self.stats.last_success_time).isoformat()
                          if self.stats.last_success_time else None,
        }

    def reset(self):
        """Manually reset the circuit breaker to closed state."""
        self._change_state(CircuitState.CLOSED)
        self.stats.consecutive_failures = 0
        self._half_open_calls = 0
        self._recent_results.clear()


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open and rejecting calls."""
    pass


class CircuitBreakerManager:
    """Manage multiple circuit breakers for different services."""

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}

    def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        **kwargs
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                **kwargs
            )
        return self._breakers[name]

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all circuit breakers."""
        return {
            name: breaker.get_stats()
            for name, breaker in self._breakers.items()
        }

    def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()


# Global manager instance
circuit_manager = CircuitBreakerManager()


def with_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type = Exception
):
    """
    Decorator to wrap async functions with circuit breaker protection.

    Usage:
        @with_circuit_breaker("external_api", failure_threshold=3)
        async def call_external_api():
            ...
    """
    def decorator(func):
        breaker = circuit_manager.get_or_create(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception
        )

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)

        # Attach breaker for testing/monitoring
        wrapper.circuit_breaker = breaker

        return wrapper
    return decorator