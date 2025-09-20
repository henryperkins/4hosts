"""
Utilities for evidence handling and deduplication
"""

from typing import Dict, List, Any, Optional, Tuple


def deduplicate_evidence_quotes(quotes: List[Any]) -> List[Any]:
    """
    Deduplicate evidence quotes by ID or URL.

    Args:
        quotes: List of quote objects

    Returns:
        List of unique quotes
    """
    dedup_quotes: Dict[str, Any] = {}
    for q in quotes:
        key = getattr(q, "id", None) or getattr(q, "url", None) or f"idx-{len(dedup_quotes)}"
        key = str(key)
        if key not in dedup_quotes:
            dedup_quotes[key] = q
    return list(dedup_quotes.values())


def deduplicate_evidence_matches(matches: List[Any]) -> List[Any]:
    """
    Deduplicate evidence matches by domain and fragments.

    Args:
        matches: List of match objects

    Returns:
        List of unique matches
    """
    seen_match: set[tuple[str, tuple[str, ...]]] = set()
    unique_matches: List[Any] = []

    for m in matches:
        dom = getattr(m, "domain", "") or ""
        fragments = tuple(getattr(m, "fragments", []) or [])
        key = (dom, fragments)
        if key in seen_match:
            continue
        seen_match.add(key)
        unique_matches.append(m)

    return unique_matches


def ensure_match(item: Any) -> Optional[Any]:
    """
    Ensure an item is a valid EvidenceMatch.

    Args:
        item: The item to validate

    Returns:
        Valid EvidenceMatch or None
    """
    # Import here to avoid circular dependency
    try:
        from models.evidence import EvidenceMatch

        if item is None:
            return None
        if isinstance(item, EvidenceMatch):
            return item
        if isinstance(item, dict):
            try:
                # Try using model_validate if available (Pydantic v2)
                if hasattr(EvidenceMatch, 'model_validate'):
                    return EvidenceMatch.model_validate(item)
                else:
                    return EvidenceMatch(**item)
            except Exception:
                # Fallback: try to construct manually if has required fields
                if item.get("domain"):
                    fragments = item.get("fragments") or []
                    if isinstance(fragments, list):
                        return EvidenceMatch(domain=item.get("domain"), fragments=fragments)
        return None
    except ImportError:
        # If EvidenceMatch is not available, perform basic validation
        if item is None:
            return None
        if hasattr(item, 'domain') and hasattr(item, 'fragments'):
            return item
        return None


def normalize_evidence_quotes(quotes: List[Any]) -> List[Dict[str, Any]]:
    """
    Normalize evidence quotes to ensure consistent format.

    Args:
        quotes: List of quote objects or strings

    Returns:
        List of normalized quote dictionaries
    """
    normalized = []

    for quote in quotes:
        if isinstance(quote, str):
            # Convert string quotes to dict format
            if quote.strip():
                normalized.append({
                    "quote": quote.strip(),
                    "id": f"quote_{hash(quote)}",
                    "url": "",
                    "title": "",
                    "domain": "",
                    "start": None,
                    "end": None,
                    "published_date": None,
                    "credibility_score": None,
                    "suspicious": False,
                    "doc_summary": "",
                    "source_type": "unknown",
                })
        elif isinstance(quote, dict):
            # Ensure required fields exist
            normalized_quote = dict(quote)
            if "quote" not in normalized_quote:
                continue
            if not normalized_quote["quote"].strip():
                continue

            # Normalize string fields
            normalized_quote["quote"] = str(normalized_quote["quote"]).strip()
            for field in ["url", "title", "domain", "doc_summary"]:
                if field in normalized_quote:
                    normalized_quote[field] = str(normalized_quote[field]).strip()

            # Ensure ID exists
            if "id" not in normalized_quote:
                normalized_quote["id"] = f"quote_{hash(normalized_quote['quote'])}"

            normalized.append(normalized_quote)
        else:
            # Try to convert object to dict
            try:
                quote_dict = dict(quote) if hasattr(quote, '__dict__') else {"quote": str(quote)}
                if quote_dict.get("quote", "").strip():
                    normalized.append(quote_dict)
            except Exception:
                continue

    return normalized


def normalize_evidence_matches(matches: List[Any]) -> List[Dict[str, Any]]:
    """
    Normalize evidence matches to ensure consistent format.

    Args:
        matches: List of match objects or dictionaries

    Returns:
        List of normalized match dictionaries
    """
    normalized = []

    for match in matches:
        if isinstance(match, dict):
            # Ensure required fields exist
            if not match.get("domain"):
                continue

            normalized_match = dict(match)
            normalized_match["domain"] = str(normalized_match["domain"]).strip()

            # Normalize fragments
            fragments = normalized_match.get("fragments", [])
            if isinstance(fragments, list):
                normalized_fragments = []
                for frag in fragments:
                    if isinstance(frag, str) and frag.strip():
                        normalized_fragments.append(frag.strip())
                normalized_match["fragments"] = normalized_fragments

            if normalized_match["fragments"]:  # Only include if we have fragments
                normalized.append(normalized_match)
        else:
            # Try to convert object to dict
            try:
                match_dict = dict(match) if hasattr(match, '__dict__') else {}
                if match_dict.get("domain"):
                    normalized.append(match_dict)
            except Exception:
                continue

    return normalized


def filter_empty_evidence(quotes: List[Any], matches: List[Any]) -> Tuple[List[Any], List[Any]]:
    """
    Filter out empty or invalid evidence items.

    Args:
        quotes: List of evidence quotes
        matches: List of evidence matches

    Returns:
        Tuple of (filtered_quotes, filtered_matches)
    """
    filtered_quotes = []
    for quote in quotes:
        if isinstance(quote, str):
            if quote.strip():
                filtered_quotes.append(quote)
        elif isinstance(quote, dict):
            if quote.get("quote", "").strip():
                filtered_quotes.append(quote)
        elif hasattr(quote, 'quote') and str(getattr(quote, 'quote', '')).strip():
            filtered_quotes.append(quote)

    filtered_matches = []
    for match in matches:
        if isinstance(match, dict):
            if match.get("domain") and match.get("fragments"):
                filtered_matches.append(match)
        elif hasattr(match, 'domain') and hasattr(match, 'fragments'):
            if getattr(match, 'domain') and getattr(match, 'fragments'):
                filtered_matches.append(match)

    return filtered_quotes, filtered_matches
