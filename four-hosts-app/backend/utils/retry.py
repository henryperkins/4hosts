"""
Centralized retry and backoff configuration for the Four-Hosts app.
Consolidates retry, backoff, jitter, and rate limit handling.
"""

import asyncio
import os
import random
import logging
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = structlog.get_logger(__name__)


class RetryConfig:
    """Centralized retry configuration loaded from environment."""

    # General retry settings
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

    # LLM-specific settings
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
    LLM_BACKOFF_MIN_SEC = float(os.getenv("LLM_BACKOFF_MIN_SEC", "2"))
    LLM_BACKOFF_MAX_SEC = float(os.getenv("LLM_BACKOFF_MAX_SEC", "8"))

    # Search/fetch rate limit settings
    RATE_LIMIT_BASE_DELAY = float(
        os.getenv("SEARCH_RATE_LIMIT_BASE_DELAY", "1")
    )
    RATE_LIMIT_BACKOFF_FACTOR = float(
        os.getenv("SEARCH_RATE_LIMIT_BACKOFF_FACTOR", "2")
    )
    RATE_LIMIT_MAX_DELAY = float(
        os.getenv("SEARCH_RATE_LIMIT_MAX_DELAY", "30")
    )
    RATE_LIMIT_JITTER = os.getenv("SEARCH_RATE_LIMIT_JITTER", "full").lower()

    # API-specific settings
    API_RETRY_BASE_DELAY = float(os.getenv("API_RETRY_BASE_DELAY", "1"))
    API_RETRY_MAX_DELAY = float(os.getenv("API_RETRY_MAX_DELAY", "60"))
    API_RETRY_MULTIPLIER = float(os.getenv("API_RETRY_MULTIPLIER", "2"))


def apply_jitter(delay: float, jitter_mode: str = "full") -> float:
    """
    Apply jitter to a delay value to prevent thundering herd.

    Args:
        delay: Base delay in seconds
        jitter_mode: "full", "equal", "decorr", or "none"

    Returns:
        Jittered delay in seconds
    """
    if jitter_mode == "full":
        # Full jitter: uniform random between 0 and delay
        return random.uniform(0, delay)
    if jitter_mode == "equal":
        # Equal jitter: delay/2 + uniform random between 0 and delay/2
        return delay / 2 + random.uniform(0, delay / 2)
    if jitter_mode == "decorr":
        # Decorrelated jitter (simplified):
        # delay * 0.5 + uniform random between 0 and delay * 0.5
        return delay * 0.5 + random.uniform(0, delay * 0.5)
    # No jitter
    return delay


def parse_retry_after(retry_after: str) -> Optional[float]:
    """
    Parse Retry-After header value (seconds or HTTP date).

    Returns:
        Delay in seconds or None if parsing fails
    """
    if not retry_after:
        return None

    value = retry_after.strip()

    # Try parsing as seconds first
    try:
        return float(value)
    except ValueError:
        pass

    # Try parsing as HTTP date
    try:
        from email.utils import parsedate_to_datetime
        when = parsedate_to_datetime(value)
        now = datetime.now(when.tzinfo)
        delta = (when - now).total_seconds()
        return max(0, delta)
    except Exception:
        pass

    return None


def calculate_exponential_backoff(
    attempt: int,
    base_delay: float = None,
    factor: float = None,
    max_delay: float = None,
    jitter_mode: str = None,
) -> float:
    """
    Calculate exponential backoff delay with optional jitter.
    """
    base = base_delay or RetryConfig.RATE_LIMIT_BASE_DELAY
    fac = factor or RetryConfig.RATE_LIMIT_BACKOFF_FACTOR
    cap = max_delay or RetryConfig.RATE_LIMIT_MAX_DELAY
    jitter = jitter_mode or RetryConfig.RATE_LIMIT_JITTER

    delay = base * (fac ** max(0, attempt - 1))
    delay = min(delay, cap)
    return apply_jitter(delay, jitter)


async def handle_rate_limit(
    url: str,
    response_headers: Dict[str, Any],
    attempt: int = 1,
    prefer_server: bool = False,
    base_delay: float = None,
    factor: float = None,
    max_delay: float = None,
    jitter_mode: str = None,
) -> float:
    """
    Handle rate limiting with exponential backoff.
    Respects server Retry-After when provided.

    Returns:
        Actual delay applied in seconds
    """
    computed = calculate_exponential_backoff(
        attempt=attempt,
        base_delay=base_delay,
        factor=factor,
        max_delay=max_delay,
        jitter_mode=jitter_mode,
    )

    # Check for server Retry-After
    retry_after = None
    for key in (
        "retry-after",
        "Retry-After",
        "x-retry-after",
        "X-Retry-After",
    ):
        if key in response_headers:
            retry_after = parse_retry_after(str(response_headers[key]))
            if retry_after is not None:
                break

    # Determine actual delay
    if prefer_server and retry_after is not None:
        delay = retry_after
    elif retry_after is not None:
        delay = min(computed, retry_after)
    else:
        delay = computed

    # Cap at max delay
    cap = max_delay or RetryConfig.RATE_LIMIT_MAX_DELAY
    delay = min(delay, cap)

    logger.info(
        "Rate limited",
        attempt=attempt,
        computed_delay=computed,
        retry_after=retry_after,
        actual_delay=delay,
        url=url,
    )

    await asyncio.sleep(delay)
    return delay


def get_llm_retry_decorator():
    """
    Get standardized retry decorator for LLM operations.
    """
    return retry(
        stop=stop_after_attempt(RetryConfig.LLM_MAX_RETRIES),
        wait=wait_exponential(
            multiplier=1,
            min=RetryConfig.LLM_BACKOFF_MIN_SEC,
            max=RetryConfig.LLM_BACKOFF_MAX_SEC,
        ),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=before_sleep_log(logger, logging.INFO),
        reraise=True,
    )


def get_api_retry_decorator(
    max_attempts: int = None,
    exceptions: tuple = None,
    min_wait: float = None,
    max_wait: float = None,
):
    """
    Get standardized retry decorator for API operations.
    """
    max_attempts = max_attempts or RetryConfig.MAX_RETRIES
    exceptions = exceptions or (Exception,)
    min_wait = min_wait or RetryConfig.API_RETRY_BASE_DELAY
    max_wait = max_wait or RetryConfig.API_RETRY_MAX_DELAY

    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=RetryConfig.API_RETRY_MULTIPLIER,
            min=min_wait,
            max=max_wait,
        ),
        retry=retry_if_exception_type(exceptions),
        before_sleep=before_sleep_log(logger, logging.INFO),
        reraise=True,
    )


def get_search_retry_decorator():
    """
    Get standardized retry decorator for search operations.
    """
    return retry(
        stop=stop_after_attempt(RetryConfig.MAX_RETRIES),
        wait=wait_random_exponential(
            multiplier=RetryConfig.RATE_LIMIT_BACKOFF_FACTOR,
            min=RetryConfig.RATE_LIMIT_BASE_DELAY,
            max=RetryConfig.RATE_LIMIT_MAX_DELAY,
        ),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=before_sleep_log(logger, logging.INFO),
        reraise=True,
    )


class RateLimitedError(Exception):
    """Exception raised when rate limited."""

    def __init__(
        self,
        message: Optional[str] = None,
        retry_after: Optional[float] = None,
    ) -> None:
        # Default to a generic message so callers can omit it for convenience.
        super().__init__(message or "Rate limited")
        self.retry_after = retry_after


async def retry_with_backoff(
    func: Callable,
    *args,
    max_attempts: int = None,
    base_delay: float = None,
    factor: float = None,
    max_delay: float = None,
    jitter_mode: str = None,
    exceptions: tuple = (Exception,),
    **kwargs,
):
    """
    Generic retry wrapper with exponential backoff.
    """
    max_attempts = max_attempts or RetryConfig.MAX_RETRIES

    last_exception = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await func(*args, **kwargs)
        except exceptions as e:  # type: ignore[misc]
            last_exception = e

            if attempt >= max_attempts:
                logger.error(
                    "Max retries exhausted",
                    function=getattr(func, "__name__", "unknown"),
                    exception=str(e),
                )
                raise

            delay = calculate_exponential_backoff(
                attempt,
                base_delay,
                factor,
                max_delay,
                jitter_mode,
            )

            logger.info(
                "Retrying function after backoff",
                function=getattr(func, "__name__", "unknown"),
                attempt=attempt,
                max_attempts=max_attempts,
                delay=delay,
                exception=str(e),
            )

            await asyncio.sleep(delay)

    raise last_exception  # pragma: no cover


async def instrumented_retry(
    func: Callable,
    *args,
    max_attempts: int = None,
    base_delay: float = None,
    factor: float = None,
    max_delay: float = None,
    jitter_mode: str = None,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception, float, bool], None]] = None,
    on_give_up: Optional[Callable[[int, Exception], None]] = None,
    **kwargs,
):
    """
    Retry an async function with backoff and optional hooks.
    """
    max_attempts = max_attempts or RetryConfig.MAX_RETRIES
    base_delay = base_delay or RetryConfig.RATE_LIMIT_BASE_DELAY
    factor = factor or RetryConfig.RATE_LIMIT_BACKOFF_FACTOR
    max_delay = max_delay or RetryConfig.RATE_LIMIT_MAX_DELAY
    jitter_mode = jitter_mode or RetryConfig.RATE_LIMIT_JITTER

    last_exception: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await func(*args, **kwargs)
        except asyncio.CancelledError:
            # Preserve cooperative cancellation semantics
            raise
        except exceptions as e:  # type: ignore[misc]
            last_exception = e
            will_retry = attempt < max_attempts
            if not will_retry:
                try:
                    if on_give_up:
                        on_give_up(attempt, e)
                finally:
                    logger.error(
                        "Max retries exhausted",
                        function=getattr(func, "__name__", "unknown"),
                        attempt=attempt,
                        max_attempts=max_attempts,
                        exception=str(e),
                    )
                raise

            delay = calculate_exponential_backoff(
                attempt=attempt,
                base_delay=base_delay,
                factor=factor,
                max_delay=max_delay,
                jitter_mode=jitter_mode,
            )

            try:
                if on_retry:
                    on_retry(attempt, e, delay, True)
            finally:
                logger.info(
                    "Retrying function after backoff",
                    function=getattr(func, "__name__", "unknown"),
                    attempt=attempt,
                    max_attempts=max_attempts,
                    delay=delay,
                    exception=str(e),
                )

            await asyncio.sleep(delay)

    # Defensive: should not reach here
    if last_exception:
        raise last_exception
    raise RuntimeError("instrumented_retry: exhausted attempts")


def get_retry_state(
    session: Any,
    url: str,
    prefix: str = "_retry_",
) -> Dict[str, Any]:
    """
    Get retry state for a URL from session attributes.
    """
    from urllib.parse import urlparse

    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        host = "generic"

    attr = f"{prefix}{host}"

    return {
        "attempts": getattr(session, f"{attr}_attempts", 0),
        "last_retry": getattr(session, f"{attr}_last_retry", None),
        "total_delay": getattr(session, f"{attr}_total_delay", 0),
    }


def set_retry_state(
    session: Any,
    url: str,
    attempts: int = None,
    last_retry: float = None,
    total_delay: float = None,
    prefix: str = "_retry_",
) -> None:
    """
    Set retry state for a URL in session attributes.
    """
    from urllib.parse import urlparse

    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        host = "generic"

    attr = f"{prefix}{host}"

    try:
        if attempts is not None:
            setattr(session, f"{attr}_attempts", attempts)
        if last_retry is not None:
            setattr(session, f"{attr}_last_retry", last_retry)
        if total_delay is not None:
            setattr(session, f"{attr}_total_delay", total_delay)
    except Exception:
        # Non-fatal if session is immutable
        pass
