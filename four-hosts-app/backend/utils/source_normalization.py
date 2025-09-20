"""
Source normalization utilities for consistent result processing.

Consolidates scattered source normalization logic from research orchestrator
and other services into reusable helpers.
"""

from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import quote_plus

from utils.url_utils import canonicalize_url, extract_domain
from utils.domain_categorizer import categorize
from utils.text_sanitize import sanitize_text
from utils.date_utils import get_current_utc, get_current_iso


def normalize_source_fields(result_or_adapter: Any) -> Dict[str, Any]:
    """
    Normalize source fields from a result or adapter object.

    Args:
        result_or_adapter: SearchResult object or ResultAdapter

    Returns:
        Dictionary with normalized source fields
    """
    try:
        # Handle both direct results and adapters
        if hasattr(result_or_adapter, 'title'):
            adapter = result_or_adapter
        else:
            # Import here to avoid circular dependency
            from services.result_adapter import ResultAdapter
            adapter = ResultAdapter(result_or_adapter)

        # Extract and normalize URL
        url = (adapter.url or "").strip()
        canonical_url = canonicalize_url(url)

        # Extract domain
        domain = extract_domain(canonical_url or url)

        # Normalize title and snippet
        title = sanitize_text(adapter.title or "", max_len=200)
        snippet = sanitize_text(adapter.snippet or "", max_len=500)

        # Get or categorize source category
        metadata = adapter.metadata or {}
        raw_data = getattr(result_or_adapter, "raw_data", {}) or {}

        source_category = None
        if isinstance(raw_data, dict):
            source_category = raw_data.get("source_category")
        if not source_category and isinstance(metadata, dict):
            source_category = metadata.get("source_category")

        if not source_category and domain:
            source_category = categorize(domain)

        source_category = (str(source_category).strip().lower() or "general") if source_category else "general"

        # Extract credibility score
        credibility = adapter.credibility_score
        if credibility is None and isinstance(metadata, dict):
            try:
                credibility = float(metadata.get("credibility_score", 0.0) or 0.0)
            except Exception:
                credibility = 0.0
        credibility = max(0.0, min(1.0, float(credibility or 0.0)))

        # Build normalized source dict
        normalized = {
            "id": canonical_url or url or f"no_url_{hash(str(result_or_adapter))}",
            "url": canonical_url or url,
            "title": title,
            "snippet": snippet,
            "domain": domain,
            "source_category": source_category,
            "credibility_score": credibility,
        }

        # Add optional metadata fields if present (metadata first, then raw_data fallback)
        if isinstance(metadata, dict):
            for key in ("credibility_explanation", "extracted_meta", "published_date"):
                if key in metadata:
                    normalized[key] = metadata[key]

        if isinstance(raw_data, dict):
            try:
                if "extracted_meta" not in normalized and raw_data.get("extracted_meta") is not None:
                    normalized["extracted_meta"] = raw_data.get("extracted_meta")
                if "credibility_explanation" not in normalized and raw_data.get("credibility_explanation"):
                    normalized["credibility_explanation"] = raw_data.get("credibility_explanation")
                # Respect explicit source_category from raw_data if present
                if "source_category" in raw_data and "source_category" not in normalized:
                    normalized["source_category"] = (str(raw_data["source_category"]).strip().lower() or normalized["source_category"])
            except Exception:
                pass

        return normalized

    except Exception:
        # Fallback: return minimal normalized dict
        return {
            "id": f"error_{hash(str(result_or_adapter))}",
            "url": "",
            "title": "",
            "snippet": "",
            "domain": "",
            "source_category": "general",
            "credibility_score": 0.0,
        }


def compute_category_distribution(results: List[Dict[str, Any]]) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    """
    Compute category distribution from results and add categorization.

    Args:
        results: List of result dictionaries

    Returns:
        Tuple of (category_distribution_dict, results_with_categories)
    """
    category_distribution: Dict[str, int] = {}
    results_with_categories = []

    for result in results:
        # Ensure result has source_category
        if "source_category" not in result:
            domain = result.get("domain", "")
            result["source_category"] = categorize(domain) if domain else "general"

        # Update distribution
        category = result.get("source_category", "general")
        category_distribution[category] = category_distribution.get(category, 0) + 1

        results_with_categories.append(result)

    return category_distribution, results_with_categories


def merge_supplemental_sources(
    base_results: List[Dict[str, Any]],
    supplemental_sources: List[Dict[str, Any]],
    original_query: str
) -> List[Dict[str, Any]]:
    """
    Merge supplemental sources (e.g., from Exa) with base results,
    avoiding duplicates by canonical URL.

    Args:
        base_results: Base search results
        supplemental_sources: Supplemental sources to merge
        original_query: Original query for pseudo-source generation

    Returns:
        Merged results list
    """
    # Build set of canonical URLs from base results
    seen_urls = set()
    for result in base_results:
        url = result.get("url", "")
        canonical = canonicalize_url(url)
        if canonical:
            seen_urls.add(canonical)

    # Add supplemental sources that aren't duplicates
    merged = list(base_results)  # Copy base results
    for supp in supplemental_sources:
        url = supp.get("url", "")
        canonical = canonicalize_url(url)
        if canonical and canonical not in seen_urls:
            # Normalize supplemental source
            normalized_supp = normalize_source_fields(supp)
            merged.append(normalized_supp)
            seen_urls.add(canonical)

    return merged


def build_supplemental_source(
    summary: str,
    supplemental_sources: List[Dict[str, Any]],
    original_query: str
) -> Dict[str, Any]:
    """
    Build a synthetic supplemental source from research summary.

    Args:
        summary: Research summary text
        supplemental_sources: List of supplemental sources
        original_query: Original query

    Returns:
        Synthetic source dictionary
    """
    pseudo_url = f"exa://research/{quote_plus(original_query[:80])}"

    content_lines = [f"Summary: {summary.strip()}"]
    if supplemental_sources:
        content_lines.append("Supplemental Sources:")
        for src in supplemental_sources[:5]:
            title = src.get("title") or src.get("url", "")
            content_lines.append(f"* {title} ({src.get('url', '')})")

    return {
        "id": pseudo_url,
        "url": pseudo_url,
        "title": "Research Synthesis",
        "snippet": summary[:280],
        "domain": "research.synthesis",
        "source_category": "research",
        "credibility_score": 0.85,
        "content": "\n".join(content_lines),
        "result_type": "research",
        "search_api": "research_synthesis",
        "source_api": "research_synthesis",
        "source": "research_synthesis",
    }


def dedupe_by_url(results: List[Any]) -> List[Any]:
    """
    Deduplicate results by canonical URL (supports dicts and objects).

    Args:
        results: List of result items (dicts or objects with url attribute)

    Returns:
        List of unique items preserving order based on canonical URL
    """
    seen_urls: set[str] = set()
    deduped: List[Any] = []

    for item in results:
        try:
            # Extract URL from dict or via ResultAdapter for objects
            if isinstance(item, dict):
                url = (item.get("url", "") or "").strip()
            else:
                try:
                    from services.result_adapter import ResultAdapter  # lazy import to avoid cycles
                    url = (ResultAdapter(item).url or "").strip()
                except Exception:
                    url = (getattr(item, "url", "") or "").strip()

            canonical = canonicalize_url(url) if url else None
            if canonical and canonical not in seen_urls:
                seen_urls.add(canonical)
                deduped.append(item)
        except Exception:
            # Best-effort: keep item if we cannot determine URL uniqueness
            if item not in deduped:
                deduped.append(item)

    return deduped


def extract_dedup_metrics(dedup_stats: Dict[str, Any]) -> Tuple[float, Dict[str, int]]:
    """
    Extract deduplication metrics for consistent reporting.

    Args:
        dedup_stats: Raw deduplication statistics

    Returns:
        Tuple of (deduplication_rate, counts_dict)
    """
    try:
        original_count = int(dedup_stats.get("original_count", 0))
        final_count = int(dedup_stats.get("final_count", 0))
        duplicates_removed = int(dedup_stats.get("duplicates_removed", 0))

        if original_count > 0:
            dedup_rate = (duplicates_removed / original_count) * 100
        else:
            dedup_rate = 0.0

        counts_dict = {
            "original_count": original_count,
            "final_count": final_count,
            "duplicates_removed": duplicates_removed,
        }

        return dedup_rate, counts_dict

    except Exception:
        return 0.0, {
            "original_count": 0,
            "final_count": 0,
            "duplicates_removed": 0,
        }


def normalize_and_dedupe_evidence(quotes: List[Any], matches: List[Any]) -> Tuple[List[Any], List[Any]]:
    """
    Normalize and deduplicate evidence quotes and matches.

    Args:
        quotes: List of evidence quotes
        matches: List of evidence matches

    Returns:
        Tuple of (normalized_quotes, normalized_matches)
    """
    # Import here to avoid circular dependency
    from utils.evidence_utils import deduplicate_evidence_quotes, deduplicate_evidence_matches

    # Normalize quotes - ensure they have required fields
    normalized_quotes = []
    for quote in quotes:
        if hasattr(quote, 'strip') and str(quote).strip():
            # Basic normalization for string quotes
            if isinstance(quote, str):
                normalized_quotes.append({
                    "quote": quote.strip(),
                    "id": f"quote_{hash(quote)}",
                })
            else:
                normalized_quotes.append(quote)

    # Normalize matches - ensure they have required fields
    normalized_matches = []
    for match in matches:
        if hasattr(match, 'get') and match.get('domain'):
            # Ensure fragments are strings and stripped
            fragments = match.get('fragments', [])
            if isinstance(fragments, list):
                normalized_fragments = [str(f).strip() for f in fragments if str(f).strip()]
                if normalized_fragments:
                    normalized_match = dict(match)
                    normalized_match['fragments'] = normalized_fragments
                    normalized_matches.append(normalized_match)

    # Deduplicate
    deduped_quotes = deduplicate_evidence_quotes(normalized_quotes)
    deduped_matches = deduplicate_evidence_matches(normalized_matches)

    return deduped_quotes, deduped_matches
