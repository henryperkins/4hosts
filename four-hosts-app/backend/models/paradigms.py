"""
Centralized helpers for paradigm naming.

Unifies and normalizes the different naming schemes used across the codebase:
- Enum values (revolutionary, devotion, analytical, strategic)
- Internal code names (dolores, teddy, bernard, maeve)

Use these helpers instead of ad-hoc mappings in modules.
"""

from __future__ import annotations

from typing import Optional, Union, Dict, List

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

# ---------------------------------------------------------------------------
# Canonical paradigm canon: keywords, patterns, and domain biases
# This centralizes identity so classification and generation stay in sync.
# ---------------------------------------------------------------------------

# Canonical keyword sets used across the system
PARADIGM_KEYWORDS: Dict[HostParadigm, List[str]] = {
    HostParadigm.DOLORES: [
        "justice", "injustice", "unfair", "expose", "reveal", "fight",
        "oppression", "oppressed", "system", "corrupt", "corruption",
        "revolution", "rebel", "resistance", "monopoly", "exploitation",
        "rights", "violation", "abuse", "scandal", "truth",
    ],
    HostParadigm.TEDDY: [
        "help", "support", "protect", "care", "assist", "aid",
        "vulnerable", "community", "together", "safe", "safety",
        "wellbeing", "welfare", "nurture", "comfort", "heal",
        "serve", "service", "volunteer", "guide", "defend",
        "compassion", "empathy", "resources",
    ],
    HostParadigm.BERNARD: [
        "analyze", "analysis", "research", "study", "examine",
        "investigate", "data", "evidence", "facts", "statistics",
        "compare", "evaluate", "measure", "test", "experiment",
        "understand", "explain", "theory", "hypothesis", "prove",
        "empirical", "methodology", "correlation", "significance",
    ],
    HostParadigm.MAEVE: [
        "strategy", "strategic", "compete", "competition", "win",
        "influence", "control", "optimize", "maximize", "leverage",
        "advantage", "opportunity", "tactic", "plan", "design",
        "implement", "execute", "achieve", "succeed", "dominate",
        "growth", "roi", "market", "scale", "efficiency",
    ],
}

# Canonical pattern sets for lightweight intent detection
PARADIGM_PATTERNS: Dict[HostParadigm, List[str]] = {
    HostParadigm.DOLORES: [
        r"how to (fight|expose|reveal|stop)",
        r"why is .* (unfair|unjust|wrong)",
        r"expose the .* (truth|corruption|scandal)",
        r"(victims?|suffering) of",
        r"stand up (to|against)",
        r"bring down the",
    ],
    HostParadigm.TEDDY: [
        r"how to (help|support|protect|care for)",
        r"best way to (assist|aid|serve)",
        r"support for .* (community|people|group)",
        r"(caring|helping) (for|with)",
        r"protect .* from",
        r"resources for",
    ],
    HostParadigm.BERNARD: [
        r"(what|how) does .* work",
        r"research (on|about|into)",
        r"evidence (for|against|of)",
        r"studies? (show|prove|indicate)",
        r"statistical .* (analysis|data)",
        r"scientific .* (explanation|theory)",
    ],
    HostParadigm.MAEVE: [
        r"(best|optimal) strategy (for|to)",
        r"how to (compete|win|succeed|influence)",
        r"competitive advantage",
        r"(increase|improve|optimize) .* (performance|results)",
        r"strategic .* (plan|approach|framework)",
        r"tactics? (for|to)",
    ],
}

# Canonical domain → paradigm prior used by classifiers
DOMAIN_PARADIGM_BIAS: Dict[str, HostParadigm] = {
    "technology": HostParadigm.BERNARD,
    "business": HostParadigm.MAEVE,
    "healthcare": HostParadigm.TEDDY,
    "education": HostParadigm.BERNARD,
    "social_justice": HostParadigm.DOLORES,
    "science": HostParadigm.BERNARD,
    "nonprofit": HostParadigm.TEDDY,
}

__all__ += [
    "PARADIGM_KEYWORDS",
    "PARADIGM_PATTERNS",
    "DOMAIN_PARADIGM_BIAS",
]
