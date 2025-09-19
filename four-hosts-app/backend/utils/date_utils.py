"""
Centralized Date Utilities Module
Consolidates date parsing, formatting, and handling across the application
"""

import re
from datetime import datetime, date, timedelta, timezone
from typing import Optional, Union, Any


def safe_parse_date(raw: Optional[Union[str, datetime, date]]) -> Optional[datetime]:
    """
    Parse various date formats into timezone-aware datetime.

    Supports:
    - ISO format strings
    - datetime/date objects
    - Year-only strings (YYYY)
    - Year-month strings (YYYY-MM)

    Args:
        raw: Input date in various formats

    Returns:
        Timezone-aware datetime or None if parsing fails
    """
    if raw is None:
        return None

    # If already a datetime, ensure timezone aware
    if isinstance(raw, datetime):
        if raw.tzinfo is None:
            return raw.replace(tzinfo=timezone.utc)
        return raw

    # Convert date to datetime
    if isinstance(raw, date):
        return datetime(raw.year, raw.month, raw.day, tzinfo=timezone.utc)

    # Handle string inputs
    if not isinstance(raw, str):
        return None

    raw = raw.strip()
    if not raw:
        return None

    # Handle Z timezone indicator
    raw = raw.replace("Z", "+00:00")

    try:
        # Try ISO format first
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, AttributeError):
        pass

    # Fallback: year-only format (YYYY)
    m = re.match(r"^\s*(\d{4})\s*$", raw)
    if m:
        try:
            return datetime(int(m.group(1)), 1, 1, tzinfo=timezone.utc)
        except Exception:
            return None

    # Fallback: year-month format (YYYY-MM)
    m = re.match(r"^\s*(\d{4})-(\d{1,2})\s*$", raw)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        try:
            return datetime(y, mo, 1, tzinfo=timezone.utc)
        except Exception:
            return None

    return None


def ensure_datetime(value: Any) -> Optional[datetime]:
    """
    Convert various date-like inputs to datetime.
    Alias for safe_parse_date for backwards compatibility.
    """
    return safe_parse_date(value)


def iso_or_none(dt: Optional[Union[datetime, date]]) -> Optional[str]:
    """
    Convert datetime/date to ISO format string, handling None safely.

    Args:
        dt: datetime or date object

    Returns:
        ISO format string or None
    """
    if dt is None:
        return None

    try:
        if isinstance(dt, datetime):
            return dt.isoformat()
        elif isinstance(dt, date):
            return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc).isoformat()
        else:
            # Try to parse and convert
            parsed = safe_parse_date(dt)
            return parsed.isoformat() if parsed else None
    except Exception:
        return None


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """
    Format datetime as timestamp for filenames (YYYYMMDD_HHMMSS).

    Args:
        dt: datetime to format (defaults to current UTC time)

    Returns:
        Formatted timestamp string
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.strftime("%Y%m%d_%H%M%S")


def format_human_readable(dt: Optional[datetime] = None, include_tz: bool = True) -> str:
    """
    Format datetime in human-readable format.

    Args:
        dt: datetime to format (defaults to current UTC time)
        include_tz: Whether to include timezone in output

    Returns:
        Human-readable date string
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    if include_tz:
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    else:
        return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_date_only(dt: Optional[datetime] = None) -> str:
    """
    Format datetime as date only (YYYY-MM-DD).

    Args:
        dt: datetime to format (defaults to current UTC time)

    Returns:
        Date string in YYYY-MM-DD format
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.strftime("%Y-%m-%d")


def get_current_utc() -> datetime:
    """
    Get current UTC datetime with timezone awareness.

    Returns:
        Current UTC datetime
    """
    return datetime.now(timezone.utc)


def get_current_iso() -> str:
    """
    Get current UTC time as ISO format string.

    Returns:
        Current time in ISO format
    """
    return datetime.now(timezone.utc).isoformat()


def calculate_age_days(dt: Optional[datetime]) -> Optional[float]:
    """
    Calculate age in days from given datetime to now.

    Args:
        dt: datetime to calculate age from

    Returns:
        Age in days or None if dt is None
    """
    if dt is None:
        return None

    parsed = ensure_datetime(dt)
    if parsed is None:
        return None

    now = get_current_utc()
    age = now - parsed
    return age.total_seconds() / 86400  # Convert to days


def calculate_age_timedelta(dt: Optional[datetime]) -> Optional[timedelta]:
    """
    Calculate age as timedelta from given datetime to now.

    Args:
        dt: datetime to calculate age from

    Returns:
        Age as timedelta or None if dt is None
    """
    if dt is None:
        return None

    parsed = ensure_datetime(dt)
    if parsed is None:
        return None

    return get_current_utc() - parsed


def is_within_window(
    dt: Optional[datetime],
    window_hours: Optional[float] = None,
    window_days: Optional[float] = None
) -> bool:
    """
    Check if datetime is within specified time window from now.

    Args:
        dt: datetime to check
        window_hours: Window size in hours
        window_days: Window size in days (used if window_hours not specified)

    Returns:
        True if within window, False otherwise
    """
    if dt is None:
        return False

    parsed = ensure_datetime(dt)
    if parsed is None:
        return False

    age = get_current_utc() - parsed

    if window_hours is not None:
        return age.total_seconds() / 3600 <= window_hours
    elif window_days is not None:
        return age.total_seconds() / 86400 <= window_days
    else:
        return False


def add_utc_timezone(dt: datetime) -> datetime:
    """
    Add UTC timezone to naive datetime.

    Args:
        dt: datetime object

    Returns:
        Timezone-aware datetime in UTC
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# Backwards compatibility exports
parse_date = safe_parse_date
to_iso = iso_or_none