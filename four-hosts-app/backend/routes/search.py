"""
SSOTA Search routes: paradigm-aware single-shot search
"""

from typing import Any, Dict, List, Optional
from dataclasses import asdict
import logging

from fastapi import APIRouter, HTTPException, Depends

from models.base import Paradigm
from core.dependencies import get_current_user
from services.paradigm_search import get_search_strategy, SearchContext
from services.search_apis import SearchConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


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
            return {
                "title": d.get("title", ""),
                "source": d.get("domain") or d.get("source_api") or d.get("source") or "",
                "url": d.get("url", ""),
                "relevance_score": float(d.get("credibility_score", 0.0) or 0.0),
                "paradigm_alignment": 1.0,  # placeholder since we don't compute alignment here
                "key_insights": [],
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

