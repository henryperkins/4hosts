"""
LLM-based critic for coverage and claim consistency checks.
Disabled by default; controlled by orchestrator.agentic_config["enable_llm_critic"].
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import json
import logging

from .llm_client import llm_client

logger = logging.getLogger(__name__)


CRITIC_SCHEMA = {
    "name": "CoverageCritique",
    "schema": {
        "type": "object",
        "properties": {
            "coverage_score": {"type": "number"},
            "missing_facets": {"type": "array", "items": {"type": "string"}},
            "flagged_sources": {"type": "array", "items": {"type": "string"}},
            "warnings": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["coverage_score", "missing_facets"],
        "additionalProperties": True
    }
}


def _build_prompt(query: str, paradigm: str, themes: List[str], focus: List[str], sources: List[Dict[str, Any]]) -> str:
    lines = [
        "You are a research critic.",
        "Task:",
        "- Estimate coverage_score (0..1) of the current sources vs the target facets.",
        "- Identify missing_facets (list of short phrases).",
        "- Flag any sources with inconsistent or unsupported claims (flagged_sources by URL).",
        "- Provide short warnings if there are common pitfalls.",
        "Context:",
        f"Original query: {query}",
        f"Paradigm: {paradigm}",
        f"Target themes: {', '.join(themes[:10])}",
        f"Focus areas: {', '.join(focus[:8])}",
        "Sources (title | url | snippet):",
    ]
    for s in sources[:8]:
        lines.append(f"- {s.get('title','')[:120]} | {s.get('url','')} | {s.get('snippet','')[:200]}")
    lines.append("Return JSON with fields: coverage_score, missing_facets, flagged_sources, warnings.")
    return "\n".join(lines)


async def llm_coverage_and_claims(
    query: str,
    paradigm: str,
    themes: List[str],
    focus: List[str],
    sources: List[Dict[str, Any]],
) -> Dict[str, Any]:
    try:
        prompt = _build_prompt(query, paradigm, themes or [], focus or [], sources or [])
        raw = await llm_client.generate_completion(
            prompt,
            paradigm=paradigm or "bernard",
            json_schema={"name": CRITIC_SCHEMA["name"], "schema": CRITIC_SCHEMA["schema"]},
            max_tokens=800,
            temperature=0.2,
        )
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except Exception:
                data = {"coverage_score": 0.5, "missing_facets": [], "flagged_sources": [], "warnings": ["critic_parse_failed"]}
        else:
            data = {"coverage_score": 0.5, "missing_facets": [], "flagged_sources": [], "warnings": ["critic_non_string_output"]}
        # Clamp coverage
        try:
            c = float(data.get("coverage_score", 0.5))
            data["coverage_score"] = max(0.0, min(1.0, c))
        except Exception:
            data["coverage_score"] = 0.5
        # Normalize arrays
        for k in ["missing_facets", "flagged_sources", "warnings"]:
            v = data.get(k, [])
            if not isinstance(v, list):
                data[k] = [str(v)]
        return data
    except Exception as e:
        logger.warning("LLM critic failed: %s", e)
        return {"coverage_score": 0.5, "missing_facets": [], "flagged_sources": [], "warnings": ["critic_failed"]}

