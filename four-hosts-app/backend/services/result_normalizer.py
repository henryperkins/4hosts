"""
Result Normalizer Utility
Consolidates result normalization logic from research_orchestrator
"""

from typing import Dict, Any, Optional, Union
from datetime import date, datetime


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