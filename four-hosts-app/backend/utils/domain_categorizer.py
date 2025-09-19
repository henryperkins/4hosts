"""
Domain/source categorization utility.

Single canonical function categorize(domain, content_type|meta) used by:
- services/search_apis.py (was _derive_source_category)
- services/credibility.py (was _infer_category)

Categories aim to be stable across UI and scoring:
- "academic", "government", "news", "video", "pdf",
- "social", "blog", "reference", "tech", "other"
"""

from __future__ import annotations

from typing import Optional, Dict


NEWS_DOMAINS = {
    "nytimes.com",
    "washingtonpost.com",
    "theguardian.com",
    "wsj.com",
    "ft.com",
    "bloomberg.com",
    "bbc.co.uk",
    "bbc.com",
    "cnn.com",
    "reuters.com",
    "apnews.com",
}

VIDEO_DOMAINS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "vimeo.com",
}

ACADEMIC_HINTS = (
    ".edu",
    "arxiv",
    "pubmed",
    "semanticscholar",
    "crossref",
    "jstor",
    "sciencedirect",
    "springer",
    "wiley",
    "tandfonline",
)

REFERENCE_DOMAINS = {
    "wikipedia.org",
    "britannica.com",
}

SOCIAL_DOMAINS = {
    "twitter.com",
    "x.com",
    "facebook.com",
    "reddit.com",
    "youtube.com",
    "medium.com",
    "substack.com",
}

TECH_HINT_TLDS = (".io", ".dev")


def _norm(domain: str) -> str:
    d = (domain or "").strip().lower()
    if d.startswith("www."):
        d = d[4:]
    return d


def categorize(
    domain: str,
    content_type: Optional[str] = None,
    meta: Optional[Dict[str, str]] = None,
) -> str:
    """
    Best-effort categorization for UI grouping, recency decay, and scoring.

    Args:
        domain:
            host portion of the URL (lowercase recommended,
            normalized here)
        content_type:
            HTTP Content-Type header (optional)
        meta:
            optional metadata map (e.g., {"content_type": "...", ...})

    Returns:
        One of:
        "academic" | "government" | "news" | "video" | "pdf" |
        "social" | "blog" | "reference" | "tech" | "other"
    """
    dom = _norm(domain)
    ctype = (content_type or (meta or {}).get("content_type") or "").lower()

    # Format-led categories
    if "application/pdf" in ctype:
        return "pdf"
    if dom in VIDEO_DOMAINS or "video" in ctype:
        return "video"

    # Government
    if dom.endswith(".gov") or dom.endswith(".mil"):
        return "government"

    # Academic
    if dom.endswith(".edu") or any(h in dom for h in ACADEMIC_HINTS):
        return "academic"

    # News
    if dom in NEWS_DOMAINS:
        return "news"
    # Heuristic: common newsroom tokens
    news_tokens = ("news", "times", "post", "journal", "daily")
    if any(tok in dom for tok in news_tokens):
        return "news"

    # Reference
    if dom in REFERENCE_DOMAINS:
        return "reference"

    # Social
    if dom in SOCIAL_DOMAINS:
        return "social"

    # Tech/blog heuristics
    if dom.endswith(TECH_HINT_TLDS) or dom.endswith(".tech"):
        return "tech"
    if "blog" in dom:
        return "blog"

    # Non-profit blogs often end with .org but aren't necessarily "academic"
    if dom.endswith(".org"):
        return "blog"

    return "other"


__all__ = ["categorize"]