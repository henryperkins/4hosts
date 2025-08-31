"""
Centralized helpers for paradigm naming.

Unifies and normalizes the different naming schemes used across the codebase:
- Enum values (revolutionary, devotion, analytical, strategic)
- Internal code names (dolores, teddy, bernard, maeve)

Use these helpers instead of ad-hoc mappings in modules.
"""

from __future__ import annotations

from typing import Optional, Union, Dict

from services.classification_engine import HostParadigm

# Map any accepted string representation to HostParadigm
_NAME_TO_ENUM: Dict[str, HostParadigm] = {
    # Enum values
    "revolutionary": HostParadigm.DOLORES,
    "devotion": HostParadigm.TEDDY,
    "analytical": HostParadigm.BERNARD,
    "strategic": HostParadigm.MAEVE,
    # Internal code names
    "dolores": HostParadigm.DOLORES,
    "teddy": HostParadigm.TEDDY,
    "bernard": HostParadigm.BERNARD,
    "maeve": HostParadigm.MAEVE,
}

# Map enum to internal code name
_ENUM_TO_INTERNAL: Dict[HostParadigm, str] = {
    HostParadigm.DOLORES: "dolores",
    HostParadigm.TEDDY: "teddy",
    HostParadigm.BERNARD: "bernard",
    HostParadigm.MAEVE: "maeve",
}


def normalize_to_enum(value: Union[str, HostParadigm, None]) -> Optional[HostParadigm]:
    """Normalize various representations to HostParadigm.

    Accepts enum instance, enum value string (e.g. "analytical"), or internal name (e.g. "bernard").
    Returns None if value is None.
    Raises ValueError for unknown strings.
    """
    if value is None:
        return None
    if isinstance(value, HostParadigm):
        return value
    key = str(value).strip().lower()
    if key not in _NAME_TO_ENUM:
        raise ValueError(f"Unknown paradigm: {value}")
    return _NAME_TO_ENUM[key]


def normalize_to_internal_code(value: Union[str, HostParadigm]) -> str:
    """Normalize to internal code name (dolores|teddy|bernard|maeve)."""
    enum_val = normalize_to_enum(value)
    if enum_val is None:
        # Should not happen for required parameter; default to bernard defensively
        return "bernard"
    return _ENUM_TO_INTERNAL[enum_val]


def enum_value(value: Union[str, HostParadigm]) -> str:
    """Return the canonical enum value string for a paradigm."""
    enum_val = normalize_to_enum(value)
    if enum_val is None:
        return "analytical"
    return enum_val.value


__all__ = [
    "normalize_to_enum",
    "normalize_to_internal_code",
    "enum_value",
]

