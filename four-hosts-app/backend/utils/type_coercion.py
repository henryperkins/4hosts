"""Shared helpers for safe type coercion across backend services."""

from typing import Any, Optional, Iterable, List, Tuple, Set, Union


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


def as_iterable(value: Any, default: Optional[Iterable[Any]] = None) -> Iterable[Any]:
    """
    Convert *value* to an iterable, wrapping single values in a list.

    Parameters
    ----------
    value : Any
        Input that might be a single value or already an iterable.
    default : Iterable[Any] | None, optional
        Value returned when value is None. Defaults to empty list.

    Returns
    -------
    Iterable[Any]
        Value as an iterable (list, tuple, set) or wrapped in a list.
    """
    if value is None:
        return default if default is not None else []
    if isinstance(value, (list, tuple, set)):
        return value
    if isinstance(value, str):
        # Special case: don't iterate over string characters
        return [value]
    # Try to iterate over it (generators, dict_keys, etc.)
    try:
        # Check if it's iterable but not string/bytes
        iter(value)
        return value
    except TypeError:
        # Not iterable, wrap in list
        return [value]


def as_list(value: Any, default: Optional[List[Any]] = None) -> List[Any]:
    """
    Convert *value* to a list, handling various input types.

    Parameters
    ----------
    value : Any
        Input that might be a single value or an iterable.
    default : List[Any] | None, optional
        Value returned when value is None. Defaults to empty list.

    Returns
    -------
    List[Any]
        Value converted to a list.
    """
    if value is None:
        return default if default is not None else []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    if isinstance(value, str):
        # Special case: don't iterate over string characters
        return [value]
    # Try to convert iterable to list
    try:
        iter(value)
        return list(value)
    except TypeError:
        # Not iterable, wrap in list
        return [value]


def coerce_iterable(value: Any) -> Iterable[Any]:
    """
    Alias for as_iterable with empty list default.
    Matches the telemetry_pipeline._coerce_iterable signature.

    Parameters
    ----------
    value : Any
        Input that might be a single value or already an iterable.

    Returns
    -------
    Iterable[Any]
        Value as an iterable or wrapped in a list.
    """
    return as_iterable(value, default=[])