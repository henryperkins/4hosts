"""
Result Normalizer Utility
Consolidates result normalization logic from research_orchestrator
"""

from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import date, datetime
import structlog

logger = structlog.get_logger(__name__)


def normalize_result(
    adapter: Any,
    url: str,
    credibility: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    is_dict_result: bool = False,
    result: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Normalize a search result into a standard dictionary format.

    Args:
        adapter: Result adapter object with title, snippet, content, domain, source_api
        url: The URL of the result
        credibility: Credibility score (optional)
        metadata: Additional metadata dict (optional)
        is_dict_result: Whether the original result was a dict (affects result_type handling)
        result: Original result object (optional, for fallback data retrieval)

    Returns:
        Normalized dictionary with standard fields
    """
    if metadata is None:
        metadata = {}

    # Normalize published date to ISO format
    published_date = metadata.get("published_date")
    if published_date is None and result is not None:
        published_date = getattr(result, "published_date", None)
    if published_date:
        try:
            if isinstance(published_date, (datetime, date)):
                published_date = published_date.isoformat()
        except Exception:
            pass

    # Determine result type
    result_type = metadata.get("result_type")
    if not result_type and not is_dict_result:
        result_type = getattr(adapter, "result_type", "web")

    # Build normalized result
    normalized = {
        "title": adapter.title,
        "url": url,
        "snippet": adapter.snippet,
        "content": adapter.content,
        "domain": adapter.domain,
        "credibility_score": float(credibility or 0.0),
        "published_date": published_date,
        "result_type": result_type or "web",
        "source_api": adapter.source_api,
        "metadata": metadata,
    }

    # Include source_category if present (for backwards compatibility)
    if "source_category" in metadata:
        normalized["source_category"] = metadata["source_category"]

    return normalized


def repair_and_filter_results(
    search_results: Dict[str, List[Any]],
    *,
    enforce_url_presence: bool = True,
    metrics: Optional[Dict[str, Any]] = None,
    diag_samples: Optional[Dict[str, List]] = None,
    progress_callback: Optional[Any] = None,
    research_id: Optional[str] = None
) -> Tuple[List[Any], List[Tuple[Any, str]]]:
    """
    Repair and filter search results, ensuring they have valid URLs and content.

    Args:
        search_results: Dictionary of search results by query key
        enforce_url_presence: Whether to drop results without URLs
        metrics: Optional metrics dictionary to track dropped results
        diag_samples: Optional diagnostics samples dictionary
        progress_callback: Optional progress callback
        research_id: Optional research ID for progress tracking

    Returns:
        Tuple of (combined valid results, results needing content backfill)
    """
    combined: List[Any] = []
    to_backfill: List[Tuple[Any, str]] = []

    for qkey, batch in (search_results or {}).items():
        if not batch:
            continue
        for r in batch:
            if r is None:
                continue

            # Enforce URL presence when enabled
            url_val = getattr(r, "url", None)
            if not url_val or not isinstance(url_val, str) or not url_val.strip():
                try:
                    if enforce_url_presence:
                        if metrics is not None:
                            metrics["dropped_no_url"] = int(metrics.get("dropped_no_url", 0)) + 1
                        if diag_samples is not None:
                            diag_samples.setdefault("no_url", []).append(getattr(r, "title", "") or "")
                        continue
                except Exception:
                    pass

            # Minimal repair mirroring legacy behavior
            content = getattr(r, "content", "") or ""
            if not str(content).strip():
                try:
                    sn = getattr(r, "snippet", "") or ""
                    tt = getattr(r, "title", "") or ""
                    if sn.strip():
                        r.content = f"Summary from search results: {sn.strip()}"
                    elif tt.strip():
                        r.content = tt.strip()
                except Exception:
                    pass

            # Stage a second-chance fetch for empty content
            if not str(getattr(r, "content", "") or "").strip() and getattr(r, "url", None):
                to_backfill.append((r, getattr(r, "url")))

            # Require content non-empty after repair/fetch
            if not str(getattr(r, "content", "") or "").strip():
                try:
                    logger.debug("[repair_and_filter_results] Dropping empty-content result: %s", getattr(r, "url", ""))
                except Exception:
                    pass
                continue

            combined.append(r)

    return combined, to_backfill