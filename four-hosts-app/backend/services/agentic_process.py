"""
Agentic Research Process utilities: lightweight planning, coverage critique,
and query proposal to drive iterative research.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any
import re
from utils.url_utils import extract_domain


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if t]


def _token_overlap(a: str, b: str) -> float:
    """Jaccard token overlap between two strings (0..1)."""
    sa, sb = set(_tokenize(a)), set(_tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def evaluate_coverage_from_sources(
    original_query: str,
    context_engineered: Any,
    sources: List[Dict[str, Any]],
) -> Tuple[float, List[str]]:
    """
    Heuristic coverage evaluation.

    Strategy:
    1. Collect *targets* from Write layer `key_themes` and Isolate layer
       `focus_areas` (lower-cased, deduped).
    2. For each target term, mark *covered* if **any** source satisfies:
       a. Exact substring match (case-insensitive)
       b. Token-overlap ≥ 0.6 (handles reordered / partial phrases)
       c. All non-stop-words of target appear individually in the text.
    3. Coverage = |covered| / |targets|.
    Returns `(coverage, missing_terms)`.
    """
    try:
        themes = set([t.lower() for t in getattr(context_engineered.write_output, "key_themes", []) or []])
    except Exception:
        themes = set()
    try:
        focus = set([t.lower() for t in getattr(context_engineered.isolate_output, "focus_areas", []) or []])
    except Exception:
        focus = set()

    targets = [t.strip() for t in list(themes.union(focus)) if len(t.strip()) > 2]
    if not targets:
        return 1.0, []

    covered = set()
    joined = [
        _normalize(f"{s.get('title','')} {s.get('snippet','')}") for s in (sources or [])
    ]
    for term in targets:
        term_norm = _normalize(term)
        words = [w for w in term_norm.split() if w]
        for text in joined:
            if not term_norm:
                continue
            # a) exact substring
            if term_norm in text:
                covered.add(term)
                break
            # b) fuzzy token overlap ≥0.6
            if _token_overlap(term_norm, text) >= 0.6:
                covered.add(term)
                break
            # c) all content words present individually (for 3+ word phrases)
            if len(words) >= 3 and all(w in text for w in words):
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
    from utils.domain_categorizer import categorize

    counts = {"academic": 0, "government": 0, "nonprofit": 0, "media": 0, "industry": 0}
    for s in sources or []:
        u = s.get("url") or ""
        host = extract_domain(u)
        category = categorize(host)

        # Map domain categorizer categories to agentic process categories
        if category == "academic":
            counts["academic"] += 1
        elif category == "government":
            counts["government"] += 1
        elif category in ["blog", "reference"]:  # .org domains and reference sites
            counts["nonprofit"] += 1
        elif category == "news":
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
