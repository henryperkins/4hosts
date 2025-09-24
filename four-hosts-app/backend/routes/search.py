"""
SSOTA Search routes: paradigm-aware single-shot search
"""

from typing import Any, Dict, List, Optional
from dataclasses import asdict
import logging
import re

import structlog

from fastapi import APIRouter, HTTPException, Depends

from models.base import Paradigm
from core.dependencies import get_current_user
from services.paradigm_search import get_search_strategy, SearchContext
from services.search_apis import SearchConfig

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


PARADIGM_KEYWORDS: Dict[str, List[str]] = {
    Paradigm.DOLORES.value: [
        "expose",
        "reveal",
        "corruption",
        "injustice",
        "accountability",
    ],
    Paradigm.TEDDY.value: [
        "help",
        "support",
        "care",
        "resources",
        "community",
    ],
    Paradigm.BERNARD.value: [
        "research",
        "evidence",
        "statistical",
        "data",
        "analysis",
    ],
    Paradigm.MAEVE.value: [
        "strategy",
        "market",
        "kpi",
        "roi",
        "competitive",
    ],
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def compute_paradigm_alignment(paradigm: str, *text_parts: str) -> float:
    """Return heuristic alignment score between text and paradigm keywords."""

    keywords = PARADIGM_KEYWORDS.get((paradigm or "").lower())
    if not keywords:
        return 0.0

    combined_text = " ".join(tp for tp in text_parts if isinstance(tp, str))
    if not combined_text:
        return 0.0

    text_l = combined_text.lower()
    hits = sum(1 for kw in keywords if kw in text_l)
    if not hits:
        return 0.0
    return round(min(1.0, hits / len(keywords)), 2)


def _split_sentences(text: str, max_items: int) -> List[str]:
    sentences = []
    if not text:
        return sentences
    for segment in _SENTENCE_SPLIT_RE.split(text.strip()):
        seg = segment.strip()
        if len(seg.split()) >= 6:
            sentences.append(seg)
        if len(sentences) >= max_items:
            break
    return sentences


def extract_key_insights_from_result(data: Dict[str, Any], max_items: int = 3) -> List[str]:
    """Pull concise insight snippets from search result payloads."""

    insights: List[str] = []
    seen = set()

    def _add_candidates(candidates: Any) -> None:
        if not candidates or len(insights) >= max_items:
            return
        if isinstance(candidates, str):
            candidates_iter = [candidates]
        else:
            candidates_iter = candidates if isinstance(candidates, (list, tuple, set)) else []
        for cand in candidates_iter:
            if not isinstance(cand, str):
                continue
            cleaned = cand.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            insights.append(cleaned)
            if len(insights) >= max_items:
                break

    # Priority: existing insights or highlights from providers
    _add_candidates(data.get("key_insights"))

    raw_data = data.get("raw_data") if isinstance(data, dict) else None
    if isinstance(raw_data, dict):
        _add_candidates(raw_data.get("highlights") or [])
        exa_meta = raw_data.get("exa")
        if isinstance(exa_meta, dict):
            _add_candidates(exa_meta.get("highlights") or [])
        _add_candidates(raw_data.get("sections") or [])

    _add_candidates(data.get("sections") or [])

    if len(insights) < max_items:
        text_blocks = [
            data.get("content"),
            data.get("snippet"),
        ]
        for block in text_blocks:
            if not block:
                continue
            _add_candidates(_split_sentences(block, max_items - len(insights)))
            if len(insights) >= max_items:
                break

    return insights[:max_items]


@router.post("/paradigm-aware")
async def paradigm_aware_search(
    payload: Dict[str, Any],
    current_user=Depends(get_current_user),
):
    """
    Execute a single paradigm-aware search.

    Request body (minimal):
    {
      "query": str,
      "paradigm": "dolores"|"teddy"|"bernard"|"maeve",
      "options": {"max_results": int, "date_range": str, "language": str, "region": str}
    }
    """
    try:
        query = (payload or {}).get("query")
        paradigm = (payload or {}).get("paradigm")
        options = (payload or {}).get("options") or {}

        if not isinstance(query, str) or not query.strip():
            raise HTTPException(status_code=400, detail="query is required")
        if paradigm not in {p.value for p in Paradigm}:
            raise HTTPException(status_code=400, detail="invalid paradigm")

        # Build search context and strategy
        strategy = get_search_strategy(paradigm)
        context = SearchContext(
            original_query=query.strip(),
            paradigm=paradigm,
            secondary_paradigm=None,
            region=str(options.get("region") or "us"),
            language=str(options.get("language") or "en"),
        )

        # Generate paradigm-specific queries
        queries = await strategy.generate_search_queries(context)
        if not queries:
            raise HTTPException(status_code=500, detail="no queries generated")

        # Execute top queries (limit 2) via the app's search manager
        from fastapi import Request
        # Use a dummy request dependency via starlette context; here we access app via router
        # Grab the app from any route state through dependency injection pattern
        # Instead, import the global search manager from app state through a hack:
        # We'll rely on a late import inside request lifecycle
        async def _run_search(q: str, max_results: int) -> List[Any]:
            from fastapi import Request  # local import to avoid circulars
            # We cannot access request here; instead, use services factory each time (fallback)
            from services.cache import cache_manager
            from services.search_apis import create_search_manager

            search_manager = None
            try:
                # Try to reuse a manager-like singleton if app stored one (during runtime)
                # This function cannot access app context cleanly; fall back to factory
                search_manager = create_search_manager(cache_manager=cache_manager)
            except Exception:
                search_manager = create_search_manager()

            config = SearchConfig(max_results=min(int(options.get("max_results") or 25), 50),
                                  language=context.language, region=context.region)
            return await search_manager.search_with_fallback(q, config)

        aggregated: List[Any] = []
        for q in queries[:2]:
            qtext = q.get("query") if isinstance(q, dict) else str(q)
            if not qtext:
                continue
            try:
                results = await _run_search(qtext, int(options.get("max_results") or 25))
                aggregated.extend(results)
            except Exception as e:
                logger.warning("search error: %s", e)

        # Filter/rank via strategy-specific logic where possible
        try:
            filtered = await strategy.filter_and_rank_results(aggregated, context)  # type: ignore[arg-type]
        except Exception:
            filtered = aggregated

        def _to_item(r) -> Dict[str, Any]:
            try:
                d = r.to_dict() if hasattr(r, "to_dict") else (asdict(r) if hasattr(r, "__dataclass_fields__") else dict(r))
            except Exception:
                d = getattr(r, "__dict__", {})
            alignment_score = compute_paradigm_alignment(
                paradigm,
                d.get("title"),
                d.get("snippet"),
                d.get("content"),
                " ".join(d.get("sections", []) if isinstance(d.get("sections"), list) else []),
            )
            insights = extract_key_insights_from_result(d)
            return {
                "title": d.get("title", ""),
                "source": d.get("domain") or d.get("source_api") or d.get("source") or "",
                "url": d.get("url", ""),
                "relevance_score": float(d.get("credibility_score", 0.0) or 0.0),
                "paradigm_alignment": alignment_score,
                "key_insights": insights,
            }

        items = [_to_item(r) for r in (filtered or [])]

        return {
            "query": query,
            "paradigm": paradigm,
            "search_modifications": [
                "Applied paradigm-specific modifiers",
                f"Generated {len(queries)} queries; executed {min(2, len(queries))}",
            ],
            "results": items,
            "total_results": len(aggregated),
            "filtered_results": len(items),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("paradigm_aware_search failed: %s", e)
        raise HTTPException(status_code=500, detail="search failed")
