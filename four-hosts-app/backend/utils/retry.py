"""
Centralized retry and backoff configuration for the Four-Hosts application.
Consolidates retry policies, exponential backoff, jitter strategies, and rate limit handling.
"""

import asyncio
import os
import random
import time
from typing import Optional, Dict, Any, Callable, TypeVar, Union
from datetime import datetime, timedelta
import structlog

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential,
    retry_if_exception_type,
    RetryCallState,
    before_sleep_log,
)

logger = structlog.get_logger(__name__)

T = TypeVar('T')

# ===== CONFIGURATION FROM ENVIRONMENT =====

class RetryConfig:
    """Centralized retry configuration loaded once from environment."""

    # General retry settings
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

    # LLM-specific settings
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
    LLM_BACKOFF_MIN_SEC = float(os.getenv("LLM_BACKOFF_MIN_SEC", "2"))
    LLM_BACKOFF_MAX_SEC = float(os.getenv("LLM_BACKOFF_MAX_SEC", "8"))

    # Search/fetch rate limit settings
    RATE_LIMIT_BASE_DELAY = float(os.getenv("SEARCH_RATE_LIMIT_BASE_DELAY", "1"))
    RATE_LIMIT_BACKOFF_FACTOR = float(os.getenv("SEARCH_RATE_LIMIT_BACKOFF_FACTOR", "2"))
    RATE_LIMIT_MAX_DELAY = float(os.getenv("SEARCH_RATE_LIMIT_MAX_DELAY", "30"))
    RATE_LIMIT_JITTER = os.getenv("SEARCH_RATE_LIMIT_JITTER", "full").lower()

    # API-specific settings
    API_RETRY_BASE_DELAY = float(os.getenv("API_RETRY_BASE_DELAY", "1"))
    API_RETRY_MAX_DELAY = float(os.getenv("API_RETRY_MAX_DELAY", "60"))
    API_RETRY_MULTIPLIER = float(os.getenv("API_RETRY_MULTIPLIER", "2"))


# ===== JITTER STRATEGIES =====

def apply_jitter(delay: float, jitter_mode: str = "full") -> float:
    """
    Apply jitter to a delay value to prevent thundering herd.

    Args:
        delay: Base delay in seconds
        jitter_mode: Type of jitter - "full", "equal", "decorr", or "none"

    Returns:
        Jittered delay in seconds
    """
    if jitter_mode == "full":
        # Full jitter: uniform random between 0 and delay
        return random.uniform(0, delay)
    elif jitter_mode == "equal":
        # Equal jitter: delay/2 + uniform random between 0 and delay/2
        return delay / 2 + random.uniform(0, delay / 2)
    elif jitter_mode == "decorr":
        # Decorrelated jitter (simplified): delay * 0.5 + random between 0 and delay * 0.5
        return delay * 0.5 + random.uniform(0, delay * 0.5)
    else:
        # No jitter
        return delay


# ===== RETRY-AFTER PARSING =====

def parse_retry_after(retry_after: str) -> Optional[float]:
    """
    Parse Retry-After header value (either seconds or HTTP date).

    Args:
        retry_after: Retry-After header value

    Returns:
        Delay in seconds or None if parsing fails
    """
    if not retry_after:
        return None

    retry_after = retry_after.strip()

    # Try parsing as integer seconds first
    try:
        return float(retry_after)
    except ValueError:
        pass

    # Try parsing as HTTP date
    try:
        from email.utils import parsedate_to_datetime
        retry_date = parsedate_to_datetime(retry_after)
        now = datetime.now(retry_date.tzinfo)
        delta = (retry_date - now).total_seconds()
        return max(0, delta)
    except Exception:
        pass

    return None


# ===== EXPONENTIAL BACKOFF WITH JITTER =====

def calculate_exponential_backoff(
    attempt: int,
    base_delay: float = None,
    factor: float = None,
    max_delay: float = None,
    jitter_mode: str = None
) -> float:
    """
    Calculate exponential backoff delay with optional jitter.

    Args:
        attempt: Attempt number (1-based)
        base_delay: Base delay in seconds
        factor: Multiplication factor for exponential growth
        max_delay: Maximum delay cap
        jitter_mode: Type of jitter to apply

    Returns:
        Calculated delay in seconds
    """
    base_delay = base_delay or RetryConfig.RATE_LIMIT_BASE_DELAY
    factor = factor or RetryConfig.RATE_LIMIT_BACKOFF_FACTOR
    max_delay = max_delay or RetryConfig.RATE_LIMIT_MAX_DELAY
    jitter_mode = jitter_mode or RetryConfig.RATE_LIMIT_JITTER

    # Calculate exponential delay
    delay = base_delay * (factor ** max(0, attempt - 1))

    # Cap at max delay
    delay = min(delay, max_delay)

    # Apply jitter
    return apply_jitter(delay, jitter_mode)


# ===== RATE LIMIT HANDLING =====

async def handle_rate_limit(
    url: str,
    response_headers: Dict[str, Any],
    attempt: int = 1,
    prefer_server: bool = False,
    base_delay: float = None,
    factor: float = None,
    max_delay: float = None,
    jitter_mode: str = None
) -> float:
    """
    Handle rate limiting with exponential backoff and server Retry-After respect.

    Args:
        url: URL that triggered rate limit
        response_headers: HTTP response headers (may contain Retry-After)
        attempt: Current attempt number
        prefer_server: Whether to prefer server's Retry-After over calculated backoff
        base_delay: Base delay for exponential backoff
        factor: Multiplication factor
        max_delay: Maximum delay cap
        jitter_mode: Jitter strategy

    Returns:
        Actual delay applied in seconds
    """
    # Calculate exponential backoff
    computed_delay = calculate_exponential_backoff(
        attempt, base_delay, factor, max_delay, jitter_mode
    )

    # Check for server Retry-After
    retry_after = None
    for key in ("retry-after", "Retry-After", "x-retry-after", "X-Retry-After"):
        if key in response_headers:
            retry_after = parse_retry_after(str(response_headers[key]))
            if retry_after is not None:
                break

    # Determine actual delay
    if prefer_server and retry_after is not None:
        delay = retry_after
    elif retry_after is not None:
        # Use minimum of computed and server-suggested
        delay = min(computed_delay, retry_after)
    else:
        delay = computed_delay

    # Cap at max delay
    max_delay = max_delay or RetryConfig.RATE_LIMIT_MAX_DELAY
    delay = min(delay, max_delay)

    logger.info(
        f"Rate limited on {url}",
        attempt=attempt,
        computed_delay=computed_delay,
        retry_after=retry_after,
        actual_delay=delay
    )

    await asyncio.sleep(delay)
    return delay


# ===== TENACITY DECORATORS =====

def get_llm_retry_decorator():
    """
    Get standardized retry decorator for LLM operations.

    Returns:
        Tenacity retry decorator configured for LLM calls
    """
    return retry(
        stop=stop_after_attempt(RetryConfig.LLM_MAX_RETRIES),
        wait=wait_exponential(
            multiplier=1,
            min=RetryConfig.LLM_BACKOFF_MIN_SEC,
            max=RetryConfig.LLM_BACKOFF_MAX_SEC,
        ),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=before_sleep_log(logger, structlog.INFO),
    )


def get_api_retry_decorator(
    max_attempts: int = None,
    exceptions: tuple = None,
    min_wait: float = None,
    max_wait: float = None
):
    """
    Get standardized retry decorator for API operations.

    Args:
        max_attempts: Maximum number of attempts
        exceptions: Tuple of exception types to retry on
        min_wait: Minimum wait time
        max_wait: Maximum wait time

    Returns:
        Tenacity retry decorator configured for API calls
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
        before_sleep=before_sleep_log(logger, structlog.INFO),
    )


def get_search_retry_decorator():
    """
    Get standardized retry decorator for search operations.

    Returns:
        Tenacity retry decorator configured for search API calls
    """
    return retry(
        stop=stop_after_attempt(RetryConfig.MAX_RETRIES),
        wait=wait_random_exponential(
            multiplier=RetryConfig.RATE_LIMIT_BACKOFF_FACTOR,
            min=RetryConfig.RATE_LIMIT_BASE_DELAY,
            max=RetryConfig.RATE_LIMIT_MAX_DELAY,
        ),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=before_sleep_log(logger, structlog.INFO),
    )


# ===== UTILITY FUNCTIONS =====

class RateLimitedError(Exception):
    """Exception raised when rate limited."""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
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
    **kwargs
):
    """
    Generic retry wrapper with exponential backoff.

    Args:
        func: Async function to retry
        *args: Positional arguments for func
        max_attempts: Maximum number of attempts
        base_delay: Base delay for backoff
        factor: Multiplication factor
        max_delay: Maximum delay
        jitter_mode: Jitter strategy
        exceptions: Tuple of exceptions to retry on
        **kwargs: Keyword arguments for func

    Returns:
        Result of successful function call

    Raises:
        Last exception if all retries exhausted
    """
    max_attempts = max_attempts or RetryConfig.MAX_RETRIES

    last_exception = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            last_exception = e

            if attempt >= max_attempts:
                logger.error(
                    f"Max retries ({max_attempts}) exhausted",
                    function=func.__name__,
                    exception=str(e)
                )
                raise

            delay = calculate_exponential_backoff(
                attempt, base_delay, factor, max_delay, jitter_mode
            )

            logger.info(
                f"Retrying {func.__name__} after {delay:.2f}s",
                attempt=attempt,
                max_attempts=max_attempts,
                exception=str(e)
            )

            await asyncio.sleep(delay)

    raise last_exception


def get_retry_state(session: Any, url: str, prefix: str = "_retry_") -> Dict[str, Any]:
    """
    Get retry state for a URL from session attributes.

    Args:
        session: Session object (e.g., aiohttp.ClientSession)
        url: URL to get state for
        prefix: Attribute prefix for storing state

    Returns:
        Dictionary with retry state (attempts, last_retry, etc.)
    """
    from urllib.parse import urlparse

    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        host = "generic"

    attr_name = f"{prefix}{host}"

    return {
        "attempts": getattr(session, f"{attr_name}_attempts", 0),
        "last_retry": getattr(session, f"{attr_name}_last_retry", None),
        "total_delay": getattr(session, f"{attr_name}_total_delay", 0),
    }


def set_retry_state(
    session: Any,
    url: str,
    attempts: int = None,
    last_retry: float = None,
    total_delay: float = None,
    prefix: str = "_retry_"
):
    """
    Set retry state for a URL in session attributes.

    Args:
        session: Session object
        url: URL to set state for
        attempts: Number of attempts
        last_retry: Timestamp of last retry
        total_delay: Total delay accumulated
        prefix: Attribute prefix
    """
    from urllib.parse import urlparse

    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        host = "generic"

    attr_name = f"{prefix}{host}"

    try:
        if attempts is not None:
            setattr(session, f"{attr_name}_attempts", attempts)
        if last_retry is not None:
            setattr(session, f"{attr_name}_last_retry", last_retry)
        if total_delay is not None:
            setattr(session, f"{attr_name}_total_delay", total_delay)
    except Exception:
        # Non-fatal if session is immutable
        pass