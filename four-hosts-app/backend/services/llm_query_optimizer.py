"""
LLM Query Optimizer (semantic expansions)
----------------------------------------
Adds optional, LLM-powered semantic expansions to complement the
rule-based QueryOptimizer in services.search_apis.

Usage:
    from services.llm_query_optimizer import propose_semantic_variations
    vars = await propose_semantic_variations(query, paradigm)

Guarded by env ENABLE_QUERY_LLM in callers.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List
import logging

from services.llm_client import llm_client

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    """Unified flag: prefer UNIFIED_QUERY_ENABLE_LLM; fallback to legacy."""
    try:
        unified = os.getenv("UNIFIED_QUERY_ENABLE_LLM")
        if unified is not None:
            return unified.lower() in {"1", "true", "yes"}
        legacy = os.getenv("ENABLE_QUERY_LLM", "0")
        return legacy.lower() in {"1", "true", "yes"}
    except Exception:
        return False


async def propose_semantic_variations(
    query: str,
    paradigm: str = "bernard",
    *,
    max_variants: int = 4,
    key_terms: List[str] | None = None,
) -> List[str]:
    """Return up to max_variants semantically diverse, search-ready variations.

    On failure or when disabled, returns an empty list (caller merges with
    heuristic variants).
    """
    if not _enabled():
        return []

    schema: Dict[str, Any] = {
        "name": "query_variations",
        "schema": {
            "type": "object",
            "properties": {
                "variations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": max_variants,
                }
            },
            "required": ["variations"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    hints = ""
    if key_terms:
        hints = f"\nKey terms to preserve: {', '.join(key_terms[:8])}"

    prompt = (
        "Rewrite the user query into semantically different, "
        "search-effective variations. "
        "Preserve the original intent, quote proper names/phrases, "
        "and avoid hallucinating facts.\n\n"
        f"Query: {query}{hints}\n\n"
        f"Return {max_variants} concise variations (6-14 words)."
    )

    try:
        data = await llm_client.generate_structured_output(
            prompt=prompt,
            schema=schema,
            paradigm=paradigm,
        )
        raw_vars = (data.get("variations") or [])
        vars: List[str] = [
            v.strip()
            for v in raw_vars
            if isinstance(v, str) and v.strip()
        ]
        # Deduplicate while preserving order
        seen: set[str] = set()
        out = [
            v for v in vars
            if not (v in seen or seen.add(v))
        ][:max_variants]
        return out
    except Exception as e:
        logger.debug(f"LLM query variations failed: {e}")
        return []

