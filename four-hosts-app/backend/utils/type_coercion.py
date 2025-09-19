"""Shared helpers for safe numeric type coercion across backend services."""

from typing import Any, Optional


def as_int(value: Any, default: int = 0) -> int:
    """
    Convert *value* to ``int`` returning *default* if conversion fails.

    Parameters
    ----------
    value : Any
        Input value that may be ``int``, ``float``, ``str`` or any other type.
    default : int, optional
        Value returned when conversion is unsuccessful. Defaults to ``0``.

    Returns
    -------
    int
        Normalised integer value.
    """
    try:
        return int(value or 0)
    except Exception:
        return default


def as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """
    Convert *value* to ``float`` returning *default`` if conversion fails.

    Parameters
    ----------
    value : Any
        Input that might be convertible to ``float``.
    default : float | None, optional
        Value returned when conversion is unsuccessful. Defaults to ``None``.

    Returns
    -------
    float | None
        Parsed float or *default*.
    """
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default