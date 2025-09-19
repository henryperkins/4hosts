"""
LLM-based critic for coverage and claim consistency checks.
Disabled by default; controlled by orchestrator.agentic_config["enable_llm_critic"].
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import json
import logging
import os
import asyncio
from utils.url_utils import extract_base_domain

from .llm_client import llm_client
from .credibility import get_source_credibility
from pydantic import BaseModel, ValidationError

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


class CoverageCritiqueModel(BaseModel):
    """Pydantic model to validate and coerce critic output."""
    coverage_score: float = 0.5
    missing_facets: List[str] = []
    flagged_sources: List[str] = []
    warnings: List[str] = []


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to extract a JSON object from a possibly noisy LLM string."""
    try:
        # Fast path: exact JSON string
        return json.loads(text)
    except Exception:
        pass
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
    except Exception:
        return None
    return None


def _domain_from_url(url: str) -> Optional[str]:
    domain = extract_base_domain(url)
    return domain if domain else None


def _build_prompt(query: str, paradigm: str, themes: List[str], focus: List[str], sources: List[Dict[str, Any]], cred_hints: Optional[List[str]] = None) -> str:
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
    if cred_hints:
        lines.append("Credibility hints (domain | score | factual | bias | controversy):")
        lines.extend(cred_hints[:10])
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
        # Optional credibility verification (domain-level), reusing the app's credibility system.
        # Controlled by env CRITIC_VERIFY_CREDIBILITY ("1" to enable). This may use Brave citations
        # if ENABLE_BRAVE_GROUNDING=1 and BRAVE_API_KEY is set (handled in credibility service).
        cred_hints: List[str] = []
        flagged_low_cred: List[str] = []
        if os.getenv("CRITIC_VERIFY_CREDIBILITY", "0").lower() in ("1", "true", "yes"):
            # Prepare up to 8 sources for quick checks
            subset = sources[:8] if sources else []
            tasks = []
            meta: List[Tuple[str, str]] = []  # (url, domain)
            for s in subset:
                url = s.get("url") or ""
                domain = _domain_from_url(url)
                if not domain:
                    continue
                meta.append((url, domain))
                content = " ".join(filter(None, [s.get("title", ""), s.get("snippet", "")]))
                tasks.append(get_source_credibility(domain=domain, paradigm=paradigm or "bernard", content=content, search_terms=(query or "").split()[:8]))
            if tasks:
                try:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for (url, domain), res in zip(meta, results):
                        if isinstance(res, Exception) or res is None:
                            continue
                        score = getattr(res, "overall_score", 0.5)
                        factual = getattr(res, "fact_check_rating", None) or "unknown"
                        bias = getattr(res, "bias_rating", None) or "center"
                        contr = getattr(res, "controversy_score", 0.0)
                        contr_lbl = "high" if contr > 0.7 else ("low" if contr < 0.3 else "moderate")
                        cred_hints.append(f"- {domain} | {score:.2f} | {factual} | {bias} | {contr_lbl}")
                        if score < 0.3 or (factual == "low") or contr > 0.8:
                            flagged_low_cred.append(url)
                except Exception as e:
                    logger.debug("credibility precheck failed: %s", e)

        prompt = _build_prompt(query, paradigm, themes or [], focus or [], sources or [], cred_hints if cred_hints else None)
        raw = await llm_client.generate_completion(
            prompt,
            paradigm=paradigm or "bernard",
            json_schema={"name": CRITIC_SCHEMA["name"], "schema": CRITIC_SCHEMA["schema"]},
        )
        # Robust parsing and validation
        default_payload = {"coverage_score": 0.5, "missing_facets": [], "flagged_sources": [], "warnings": ["critic_parse_failed"]}
        if isinstance(raw, str):
            try:
                model = CoverageCritiqueModel.model_validate_json(raw)
                data = model.model_dump()
            except Exception:
                try:
                    obj = _extract_json_object(raw) or {}
                    model = CoverageCritiqueModel.model_validate(obj)
                    data = model.model_dump()
                except ValidationError as ve:
                    logger.debug("critic validation failed: %s", ve)
                    data = default_payload
                except Exception:
                    data = default_payload
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
        # Merge any low-credibility flags from verification step
        if flagged_low_cred:
            fs = data.get("flagged_sources", [])
            data["flagged_sources"] = list({*fs, *flagged_low_cred})
            warns = data.get("warnings", [])
            data["warnings"] = list({*warns, "low_credibility_sources_detected"})
        return data
    except Exception as e:
        logger.warning("LLM critic failed: %s", e)
        return {"coverage_score": 0.5, "missing_facets": [], "flagged_sources": [], "warnings": ["critic_failed"]}
