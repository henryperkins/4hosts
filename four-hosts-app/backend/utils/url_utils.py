"""
URL utilities for normalizing, validating, and extracting information from URLs.
Consolidates URL-related functionality previously scattered across multiple modules.
"""

import re
from typing import Optional, Tuple, Dict, Any
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode, unquote, urljoin

# Constants
MAX_URL_LENGTH = 2048
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")

# Tracking parameters to remove for clean URLs
TRACKING_PARAMS = {
    'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
    'fbclid', 'gclid', 'dclid', 'msclkid', 'twclid',
    'ref', 'ref_src', 'ref_url', 'referrer',
    '_ga', '_gid', '_gac', '_gl', '_gclid',
    'mc_cid', 'mc_eid', 'mkt_tok',
    'yclid', 'ysclid', 'zanpid', 'kbid', 'pinterest_id', 'pp'
}


def normalize_url(url: str) -> str:
    """
    Normalize a URL for consistent comparison and storage.
    Handles DOIs, lowercases domain, preserves path/query/fragment.

    Args:
        url: URL string to normalize

    Returns:
        Normalized URL string
    """
    if not url:
        return ""

    url = url.strip()

    # Handle DOI strings
    if url.startswith("10."):
        return f"https://doi.org/{url}"

    # Handle existing DOI URLs
    if "doi.org/" in url:
        doi = unquote(url.split("doi.org/", 1)[1])
        return f"https://doi.org/{doi}"

    # Parse and normalize regular URLs
    if "://" not in url:
        url = f"https://{url}"

    p = urlparse(url)
    # Lowercase the domain but preserve case for path/query
    return urlunparse((
        p.scheme,
        p.netloc.lower(),
        p.path,
        p.params,
        p.query,
        p.fragment
    ))


def is_valid_url(url: str) -> bool:
    """
    Check if a URL is valid and well-formed.

    Args:
        url: URL string to validate

    Returns:
        True if URL is valid, False otherwise
    """
    if not url or len(url) > MAX_URL_LENGTH:
        return False

    try:
        p = urlparse(url)
        # Must have both scheme and netloc
        return bool(p.scheme and p.netloc)
    except Exception:
        return False


def extract_domain(url: str) -> str:
    """
    Extract the domain (netloc) from a URL.

    Args:
        url: URL string

    Returns:
        Lowercase domain or empty string if extraction fails
    """
    if not url:
        return ""

    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def extract_base_domain(url: str) -> str:
    """
    Extract base domain without www prefix.

    Args:
        url: URL string

    Returns:
        Base domain without www or empty string
    """
    domain = extract_domain(url)
    if domain.startswith("www."):
        return domain[4:]
    return domain


def extract_doi(text: str) -> Optional[str]:
    """
    Extract DOI from text or URL.

    Args:
        text: Text that may contain a DOI

    Returns:
        DOI string if found, None otherwise
    """
    if not text:
        return None

    match = DOI_PATTERN.search(text)
    return match.group(0) if match else None


def clean_url(url: str, remove_tracking: bool = True, remove_fragment: bool = False) -> str:
    """
    Clean a URL by removing tracking parameters and optionally fragments.

    Args:
        url: URL to clean
        remove_tracking: Whether to remove tracking parameters
        remove_fragment: Whether to remove URL fragments (#...)

    Returns:
        Cleaned URL
    """
    if not url:
        return ""

    try:
        p = urlparse(url)

        # Process query parameters
        if remove_tracking and p.query:
            params = parse_qs(p.query, keep_blank_values=True)
            # Remove tracking parameters
            cleaned_params = {
                k: v for k, v in params.items()
                if k not in TRACKING_PARAMS
            }
            new_query = urlencode(cleaned_params, doseq=True)
        else:
            new_query = p.query

        # Reconstruct URL
        return urlunparse((
            p.scheme,
            p.netloc,
            p.path,
            p.params,
            new_query,
            "" if remove_fragment else p.fragment
        ))
    except Exception:
        return url


def sanitize_url(url: str) -> Optional[str]:
    """
    Validate and sanitize a URL for safe usage.
    Incorporates security checks from security.py.

    Args:
        url: URL to sanitize

    Returns:
        Sanitized URL or None if invalid/unsafe
    """
    if not url or len(url) > MAX_URL_LENGTH:
        return None

    url = url.strip()

    # Basic URL validation pattern (from security.py)
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )

    if not url_pattern.match(url):
        return None

    # Additional validation
    if not is_valid_url(url):
        return None

    # Clean and normalize
    cleaned = clean_url(url, remove_tracking=True)
    return normalize_url(cleaned)


def join_url(base: str, path: str) -> str:
    """
    Safely join a base URL with a path.

    Args:
        base: Base URL
        path: Path to append

    Returns:
        Joined URL
    """
    if not base:
        return path or ""
    if not path:
        return base

    return urljoin(base, path)


def parse_url_parts(url: str) -> Dict[str, Any]:
    """
    Parse URL into component parts.

    Args:
        url: URL to parse

    Returns:
        Dictionary with URL components
    """
    if not url:
        return {}

    try:
        p = urlparse(url)
        return {
            'scheme': p.scheme,
            'domain': p.netloc.lower(),
            'path': p.path,
            'params': p.params,
            'query': p.query,
            'fragment': p.fragment,
            'query_dict': parse_qs(p.query) if p.query else {},
            'is_doi': 'doi.org' in url or url.startswith('10.')
        }
    except Exception:
        return {}


def get_url_with_params(base_url: str, params: Dict[str, Any]) -> str:
    """
    Build a URL with query parameters.

    Args:
        base_url: Base URL
        params: Dictionary of query parameters

    Returns:
        URL with parameters
    """
    if not base_url:
        return ""

    if not params:
        return base_url

    # Parse existing URL
    p = urlparse(base_url)

    # Merge with existing params if any
    existing_params = parse_qs(p.query) if p.query else {}
    existing_params.update(params)

    # Build new query string
    new_query = urlencode(existing_params, doseq=True)

    # Reconstruct URL
    return urlunparse((
        p.scheme,
        p.netloc,
        p.path,
        p.params,
        new_query,
        p.fragment
    ))


# Convenience aliases for backward compatibility
validate_and_sanitize_url = sanitize_url

def is_same_domain(url1: str, url2: str) -> bool:
    """
    Check if two URLs have the same domain.

    Args:
        url1: First URL
        url2: Second URL

    Returns:
        True if same domain, False otherwise
    """
    return extract_domain(url1) == extract_domain(url2)