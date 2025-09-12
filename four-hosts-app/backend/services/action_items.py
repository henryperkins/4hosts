"""
Dynamic Action Items Service
---------------------------
Generates context-aware action items using the active paradigm and
available evidence/search results. Enabled by env ENABLE_DYNAMIC_ACTIONS.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List
import logging

from services.llm_client import llm_client

logger = logging.getLogger(__name__)


def enabled() -> bool:
    try:
        return os.getenv("ENABLE_DYNAMIC_ACTIONS", "0").lower() in {"1", "true", "yes"}
    except Exception:
        return False


def _schema(max_items: int = 6) -> Dict[str, Any]:
    return {
        "name": "action_items",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "maxItems": max_items,
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "description": {"type": "string"},
                            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                            "timeline": {"type": "string"},
                            "source_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 4,
                            }
                        },
                        "required": ["action", "priority"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        },
    }


async def generate_action_items(
    *,
    query: str,
    paradigm: str,
    search_results: List[Dict[str, Any]] | None = None,
    evidence_quotes: List[Dict[str, Any]] | None = None,
    max_items: int = 6,
) -> List[Dict[str, Any]]:
    """Generate action items aligned to the paradigm and grounded in evidence.

    Returns [] on failure or when disabled; caller should fall back to
    heuristic items.
    """
    if not enabled():
        return []

    # Build compact grounding context from results/quotes
    sr_lines: List[str] = []
    for r in (search_results or [])[:8]:
        try:
            dom = (r.get("domain") or r.get("source") or "").lower()
            snip = (r.get("snippet") or "").strip()
            sr_lines.append(f"- [{dom}] {snip[:180]}")
        except Exception:
            continue
    sr_block = "\n".join(sr_lines) or "(no search results provided)"

    qt_lines: List[str] = []
    for q in (evidence_quotes or [])[:8]:
        try:
            dom = (q.get("domain") or "").lower()
            txt = (q.get("quote") or "").strip()
            qt_lines.append(f"- [{dom}] {txt[:220]}")
        except Exception:
            continue
    qt_block = "\n".join(qt_lines) or "(no evidence quotes provided)"

    prompt = f"""
You are generating concrete, next-step actions aligned with the "{paradigm}" paradigm.
User query: {query}

Grounding evidence (search results):
{sr_block}

Grounding evidence (quotes):
{qt_block}

Guidelines:
- Be specific and feasible; avoid vague advice.
- Include short rationale in description when helpful.
- Prefer actions that leverage cited domains/sources.
- Balance ambition (high) with safety/ethics.
"""

    try:
        data = await llm_client.generate_structured_output(
            prompt=prompt,
            schema=_schema(max_items),
            paradigm=paradigm,
        )
        items = data.get("items") or []
        out: List[Dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            action = (it.get("action") or "").strip()
            if not action:
                continue
            # Normalize
            out.append({
                "action": action,
                "description": (it.get("description") or "").strip(),
                "priority": (it.get("priority") or "medium").lower(),
                "timeline": (it.get("timeline") or "").strip(),
                "source_ids": list(it.get("source_ids") or []),
            })
        return out[:max_items]
    except Exception as e:
        logger.debug(f"Dynamic action items failed: {e}")
        return []

