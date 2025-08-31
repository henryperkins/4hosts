"""
Agentic Research Process utilities: lightweight planning, coverage critique,
and query proposal to drive iterative research.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any
import re


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def evaluate_coverage_from_sources(
    original_query: str,
    context_engineered: Any,
    sources: List[Dict[str, Any]],
) -> Tuple[float, List[str]]:
    """
    Heuristic coverage evaluation:
    - Build a target set from W (key_themes) and I (focus_areas)
    - Check presence across top sources (title+snippet)
    - Return (coverage_score 0..1, missing_terms)
    """
    try:
        themes = set([t.lower() for t in getattr(context_engineered.write_output, "key_themes", []) or []])
    except Exception:
        themes = set()
    try:
        focus = set([t.lower() for t in getattr(context_engineered.isolate_output, "focus_areas", []) or []])
    except Exception:
        focus = set()

    targets = [t for t in list(themes.union(focus)) if len(t) > 2]
    if not targets:
        return 1.0, []

    covered = set()
    joined = [
        _normalize(f"{s.get('title','')} {s.get('snippet','')}") for s in (sources or [])
    ]
    for term in targets:
        term_norm = _normalize(term)
        for text in joined:
            if term_norm and term_norm in text:
                covered.add(term)
                break

    coverage = len(covered) / float(len(targets)) if targets else 1.0
    missing = [t for t in targets if t not in covered]
    return coverage, missing


def propose_queries_from_missing(
    original_query: str,
    paradigm: str,
    missing_terms: List[str],
    max_new: int = 4,
) -> List[str]:
    """
    Propose new queries mixing missing themes with paradigm-flavored modifiers.
    """
    # Guard against None input
    base = (original_query or "").strip()
    modifiers = {
        "dolores": ["investigation", "expose", "scandal"],
        "bernard": ["study", "evidence", "site:.edu OR site:.gov"],
        "maeve": ["strategy", "framework", "tactics"],
        "teddy": ["resources", "support", "guide"],
    }.get(paradigm, [])

    proposals: List[str] = []
    for term in missing_terms[: max_new * 2]:
        term = (term or "").strip()
        if not term:
            continue
        # Combine with top modifiers
        for m in modifiers[:2]:
            q = f"{term} {base} {m}".strip()
            if q not in proposals:
                proposals.append(q)
                if len(proposals) >= max_new:
                    return proposals
        # Plain term + base fallback
        if len(proposals) < max_new:
            q2 = f"{term} {base}".strip()
            if q2 not in proposals:
                proposals.append(q2)
                if len(proposals) >= max_new:
                    return proposals

    return proposals[:max_new]


def summarize_domain_gaps(sources: List[Dict[str, Any]]) -> Dict[str, int]:
    """Return simple counts by domain class: academic, government, nonprofit, media, industry."""
    from urllib.parse import urlparse
    counts = {"academic": 0, "government": 0, "nonprofit": 0, "media": 0, "industry": 0}
    for s in sources or []:
        u = s.get("url") or ""
        try:
            host = urlparse(u).netloc.lower()
        except Exception:
            host = ""
        if host.endswith(".edu"):
            counts["academic"] += 1
        elif host.endswith(".gov") or ".gov/" in u:
            counts["government"] += 1
        elif host.endswith(".org"):
            counts["nonprofit"] += 1
        elif any(k in host for k in ["nytimes", "guardian", "washingtonpost", "bloomberg", "reuters", "apnews", "bbc"]):
            counts["media"] += 1
        else:
            counts["industry"] += 1
    return counts


def propose_queries_enriched(
    base_query: str,
    paradigm: str,
    missing_terms: List[str],
    gap_counts: Dict[str, int],
    max_new: int = 6,
) -> List[str]:
    """
    Enrich proposals using source-type gaps and paradigm strategies.
    """
    proposals = []
    # Base modifier pools by paradigm
    extra_by_paradigm = {
        "bernard": ["site:.edu", "arxiv", "pubmed"],
        "maeve": ["case study", "benchmark", "mckinsey OR bcg OR gartner"],
        "dolores": ["propublica", "icij", "investigation", "whistleblower"],
        "teddy": ["resources", "nonprofit", "guide"],
    }.get(paradigm, [])

    # If academic gap, bias towards scholarly modifiers
    if gap_counts.get("academic", 0) < 2:
        extra_by_paradigm = ["site:.edu", "arxiv", *extra_by_paradigm]
    # If industry gap and paradigm strategic, add consultancy lenses
    if gap_counts.get("industry", 0) < 2 and paradigm in {"maeve"}:
        extra_by_paradigm = ["mckinsey", "bcg", "gartner", *extra_by_paradigm]
    # If media/nonprofit gap and paradigm dolores/teddy, add investigative/community lenses
    if (gap_counts.get("media", 0) + gap_counts.get("nonprofit", 0)) < 2 and paradigm in {"dolores", "teddy"}:
        extra_by_paradigm = ["investigation", "report", "community", *extra_by_paradigm]

    # Create proposals combining missing terms + enriched modifiers
    for term in missing_terms[: max_new * 2]:
        for m in extra_by_paradigm[:3]:
            q = f"{term} {base_query} {m}".strip()
            if q not in proposals:
                proposals.append(q)
                if len(proposals) >= max_new:
                    return proposals
    # Fallback to simple combo
    return proposals or propose_queries_from_missing(base_query, paradigm, missing_terms, max_new=max_new)
