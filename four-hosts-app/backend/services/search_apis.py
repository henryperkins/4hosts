"""
Search-API integrations for the Four-Hosts Research application
(Brave, Google Custom Search, ArXiv, PubMed, Semantic Scholar, CrossRef).

Large blocks of formerly duplicated code (retry / 429 logic, date parsing,
tokenisation, snippet fall-back, identical stop-word lists, etc.) have been
centralised into small utilities directly in this module to keep the file
self-contained.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import structlog
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Callable,
    Awaitable,
    Union,
    TYPE_CHECKING,
)
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import aiohttp

# Import URL utilities
from ..utils.url_utils import (
    normalize_url,
    is_valid_url,
    extract_domain,
    extract_doi,
)
# Import retry utilities
from utils.retry import (
    handle_rate_limit,
    RateLimitedError as RetryRateLimitedError,
    parse_retry_after,
    get_search_retry_decorator,
    get_api_retry_decorator,
)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None  # Optional; PDF parsing will be skipped if unavailable
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Circuit breaker for resilient external API calls
from utils.circuit_breaker import (
    with_circuit_breaker,
    CircuitOpenError,
    circuit_manager,
)
from services.rate_limiter import ClientRateLimiter
from services.text_utils import tokenize
from utils.text_sanitize import sanitize_text
from utils.domain_categorizer import categorize
from utils.date_utils import safe_parse_date
if TYPE_CHECKING:
    from search.query_planner import QueryCandidate


# --------------------------------------------------------------------------- #
#                        ENV / LOGGING / INITIALISATION                       #
# --------------------------------------------------------------------------- #

load_dotenv()

logger = structlog.get_logger(__name__)

from utils.otel import otel_span as _otel_span

MAX_LOG_BODY = int(os.getenv("SEARCH_LOG_BODY_MAX", "2048"))


def _safe_truncate(val: Any, max_len: int = MAX_LOG_BODY) -> str:
    try:
        txt = val if isinstance(val, str) else json.dumps(val, default=str)
    except Exception:
        txt = str(val)
    return txt[: max_len]


def _structured_log(level: str, event: str, meta: Dict[str, Any]):
    record = {"event": event, **meta}
    try:
        line = json.dumps(record, default=str)
    except Exception:
        safe = {k: _safe_truncate(v) for k, v in record.items()}
        line = json.dumps(safe)
    getattr(logger, level, logger.info)(line)


# --------------------------------------------------------------------------- #
#                        SHARED UTILITY FUNCTIONS                             #
# --------------------------------------------------------------------------- #
# safe_parse_date moved to utils.date_utils

def ngram_tokenize(text: str, n: int = 3) -> List[str]:
    toks = tokenize(text)
    return [" ".join(toks[i: i + n]) for i in range(max(0, len(toks) - n + 1))]

# extract_doi moved to utils.url_utils


def extract_arxiv_id(url: str | None) -> Optional[str]:
    if not url:
        return None
    m = re.search(
        r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})",
        url,
        re.I,
    )
    return m.group(1) if m else None


def ensure_snippet_content(result: "SearchResult"):
    """Guarantee SearchResult.content has at least the search snippet."""
    if not result.content and result.snippet:
        result.content = f"Summary from search results: {result.snippet}"
        result.raw_data.setdefault("content_source", "search_api_snippet")
        result.raw_data.setdefault("content_type", "snippet_only")


def normalize_result_text_fields(result: "SearchResult") -> None:
    """Normalize user-visible fields for safe display/logging.

    Called after result construction/fetch to ensure provider markup does not
    leak into progress logs or the UI (e.g. <jats:p>, Word <span data-...>).
    """
    try:
        max_title = int(os.getenv("SEARCH_TITLE_MAX_LEN", "300"))
        result.title = sanitize_text(result.title, max_len=max_title)
    except Exception:
        pass
    try:
        max_snippet = int(os.getenv("SEARCH_SNIPPET_MAX_LEN", "800"))
        result.snippet = sanitize_text(result.snippet, max_len=max_snippet)
    except Exception:
        pass


def _safe_truncate_query(q: str, limit: int) -> str:
    """Truncate without cutting mid-token or leaving unmatched quotes.

    - Prefer last whitespace within limit
    - If a quote is opened but not closed, drop the dangling part
    """
    if len(q) <= limit:
        return q
    cut = q[:limit]
    # Backtrack to last whitespace to avoid mid-token cut
    ws = max(cut.rfind(" "), cut.rfind("\t"), cut.rfind("\n"))
    if ws > limit * 0.6:  # only backtrack if reasonably close
        cut = cut[: ws].rstrip()
    # Balance quotes – if odd number of quotes, drop trailing unmatched fragment
    if cut.count('"') % 2 == 1:
        last_q = cut.rfind('"')
        if last_q >= 0:
            cut = cut[: last_q].rstrip()
    return cut


# --------------------------------------------------------------------------- #
#                        NETWORK / FETCH HELPERS                              #
# --------------------------------------------------------------------------- #

# Use RateLimitedError from utils.retry, but keep alias for backward compatibility
RateLimitedError = RetryRateLimitedError


async def response_body_snippet(
    response: aiohttp.ClientResponse, limit: int = MAX_LOG_BODY
) -> str:
    try:
        return (await response.text())[:limit]
    except Exception:
        try:
            return (await response.read())[:limit].decode(errors="replace")
        except Exception:
            return "<unreadable>"


# _retry_after_to_seconds replaced by utils.retry.parse_retry_after
def _retry_after_to_seconds(val: str) -> float:
    """Parse Retry-After which may be seconds or HTTP-date."""
    result = parse_retry_after(val)
    return result if result is not None else 0.0


async def _rate_limit_backoff(
    url: str,
    response_headers: Dict[str, Any],
    *,
    session: Any,
    attr_prefix: str = "_rate_attempts_",
    prefer_server: bool = False,
) -> None:
    """Sleep based on Retry-After and raise RateLimitedError for retry logic.

    Centralised helper to keep identical 429 handling paths in sync across
    different fetch functions.
    """
    # Track attempts per host on the session (for test visibility and tuning)
    host = extract_domain(url) or "generic"
    attr_name = f"{attr_prefix}{host}"
    attempts = int(getattr(session, attr_name, 0) or 0) + 1
    try:
        setattr(session, attr_name, attempts)
    except Exception:
        # Non-fatal if session is immutable
        pass

    # Use centralized rate limit handler - it handles all the logic and sleeps
    delay = await handle_rate_limit(
        url=url,
        response_headers=response_headers,
        attempt=attempts,
        prefer_server=prefer_server
    )

    _structured_log("warning", "rate_limited_fetch", {
        "url": url,
        "attempt": attempts,
        "delay": round(delay, 3)
    })

    # Note: handle_rate_limit already slept, but we need to raise for retry handlers
    raise RateLimitedError(f"Rate limited on {url}")


def _first_non_empty(*vals: Optional[str]) -> Optional[str]:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _extract_metadata_from_html(html: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Extract structured metadata from HTML head (OG tags, JSON-LD, citation_*, canonical).

    Returns a dict with keys like: title, description, canonical_url, site_name, published_date,
    authors (List[str]), language, http (headers subset), and raw maps for debugging (og, meta, ld_json).
    """
    soup = BeautifulSoup(html, "html.parser")
    meta: Dict[str, Any] = {"og": {}, "meta": {}, "ld": {}}
    # Title / lang
    try:
        meta["title"] = (soup.title.get_text(strip=True) if soup.title else None)
    except Exception:
        pass
    try:
        html_tag = soup.find("html")
        if html_tag and html_tag.has_attr("lang"):
            meta["language"] = (html_tag.get("lang") or "").strip() or None
    except Exception:
        pass
    # Canonical
    try:
        link_canon = soup.find(
            "link",
            rel=lambda v: bool(v and "canonical" in str(v).lower())
        )
        if link_canon and link_canon.get("href"):
            meta["canonical_url"] = link_canon.get("href")
    except Exception:
        pass
    # OG / Twitter / generic meta
    try:
        for m in soup.find_all("meta"):
            k = (m.get("property") or m.get("name") or "").strip().lower()
            v = m.get("content")
            if not k or v is None:
                continue
            if k.startswith("og:") or k.startswith("article:"):
                meta["og"][k] = v
            else:
                meta["meta"][k] = v
    except Exception:
        pass
    # JSON-LD Article
    ld_main = {}
    try:
        import json as _json
        for s in soup.find_all("script", type=lambda t: t and "ld+json" in t):
            try:
                data = _json.loads(s.get_text() or "{}")
            except Exception:
                continue
            items = data if isinstance(data, list) else [data]
            for it in items:
                t = (it.get("@type") if isinstance(it, dict) else None)
                if isinstance(t, list):
                    t = t[0] if t else None
                if str(t).lower() in {"article", "newsarticle", "blogposting", "scholarlyarticle"}:
                    ld_main = it
                    break
            if ld_main:
                break
    except Exception:
        pass
    if ld_main:
        meta["ld"] = ld_main

    # Authors extraction
    authors: list[str] = []
    # citation_author may appear multiple times
    try:
        for m in soup.find_all("meta", attrs={"name": "citation_author"}):
            val = (m.get("content") or "").strip()
            if val:
                authors.append(val)
    except Exception:
        pass
    # Generic author fields
    for key in ("author", "article:author", "parsely-author"):
        v = meta["meta"].get(key) or meta["og"].get(key)
        if isinstance(v, str) and v.strip():
            authors.append(v.strip())
    # JSON-LD authors
    try:
        a = ld_main.get("author") if isinstance(ld_main, dict) else None
        if isinstance(a, list):
            for ai in a:
                if isinstance(ai, dict) and ai.get("name"):
                    authors.append(str(ai["name"]))
                elif isinstance(ai, str):
                    authors.append(ai)
        elif isinstance(a, dict) and a.get("name"):
            authors.append(str(a["name"]))
        elif isinstance(a, str):
            authors.append(a)
    except Exception:
        pass
    if authors:
        meta["authors"] = list(dict.fromkeys([a for a in authors if a]))

    # Title/description selection
    meta["title"] = _first_non_empty(meta.get("og", {}).get("og:title"), meta.get("ld", {}).get("headline") if isinstance(meta.get("ld"), dict) else None, meta.get("meta", {}).get("twitter:title"), meta.get("title"))
    meta["description"] = _first_non_empty(meta.get("og", {}).get("og:description"), meta.get("meta", {}).get("description"), meta.get("meta", {}).get("twitter:description"))
    meta["site_name"] = _first_non_empty(meta.get("og", {}).get("og:site_name"), meta.get("meta", {}).get("application-name"))
    # Published date
    pub = _first_non_empty(
        (meta.get("ld", {}) or {}).get("datePublished") if isinstance(meta.get("ld"), dict) else None,
        meta.get("og", {}).get("article:published_time"),
        meta.get("meta", {}).get("date"),
        meta.get("meta", {}).get("citation_publication_date"),
        meta.get("meta", {}).get("dc.date"),
        meta.get("meta", {}).get("dcterms.date"),
        meta.get("og", {}).get("og:updated_time"),
    )
    if pub:
        meta["published_date"] = pub

    # HTTP headers subset
    h = {k.lower(): v for k, v in (headers or {}).items()}
    http_info: Dict[str, Any] = {}
    if h:
        http_info["content_type"] = h.get("content-type")
        http_info["last_modified"] = h.get("last-modified")
        http_info["content_length"] = h.get("content-length")
    if http_info:
        meta["http"] = http_info
    return meta


# _derive_source_category removed in favour of utils.domain_categorizer.categorize

def _clean_html_noise(soup: BeautifulSoup) -> None:
    """Strip noisy elements and obvious non-content blocks by id/class heuristics."""
    for s in soup(["script", "style", "noscript", "template", "iframe", "svg", "canvas"]):
        s.decompose()
    # Remove common boilerplate containers
    noise_keys = (
        "nav", "header", "footer", "aside", "form", "button", "input", "select", "label",
        "figure", "figcaption", "video", "audio"
    )
    for tag in noise_keys:
        for el in soup.find_all(tag):
            try:
                el.decompose()
            except Exception:
                continue
    # Class/id based heuristics
    negative = (
        "nav", "menu", "breadcrumb", "footer", "header", "sidebar", "widget", "subscribe",
        "newsletter", "social", "share", "modal", "overlay", "banner", "cookie", "consent",
        "promo", "advert", "ad-", "ads", "related", "recommend", "comment", "pagination",
    )
    for el in soup.find_all(True):
        try:
            ident = (" ".join([el.get("id") or "", " ".join(el.get("class") or [])])).lower()
            if any(k in ident for k in negative):
                el.decompose()
        except Exception:
            continue


def _node_text_len(el) -> int:
    try:
        return len(el.get_text(" ", strip=True))
    except Exception:
        return 0


def _link_text_len(el) -> int:
    try:
        return sum(len(a.get_text(" ", strip=True) or "") for a in el.find_all("a"))
    except Exception:
        return 0


def _punct_count(el) -> int:
    try:
        import re as _re
        txt = el.get_text(" ", strip=True)
        return len(_re.findall(r"[\.!?,;:]", txt))
    except Exception:
        return 0


def _score_block(el) -> float:
    # Core readability-esque scoring: text mass, low link density, punctuation density, headings bonus
    tlen = _node_text_len(el)
    if tlen == 0:
        return 0.0
    lden = (_link_text_len(el) / max(1.0, float(tlen)))
    pden = (_punct_count(el) / max(1.0, float(tlen)))
    bonus = 0.0
    try:
        if el.find(["h1", "h2", "h3"]):
            bonus += 0.15
    except Exception:
        pass
    return (tlen * (1.0 - min(0.9, lden)) * (1.0 + min(0.5, pden))) * (1.0 + bonus)


def _assemble_text_from_block(el, max_chars: int | None = None) -> str:
    parts: list[str] = []
    for node in el.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "pre"], recursive=True):
        try:
            tag = node.name.lower()
            txt = node.get_text(" ", strip=True)
            if not txt:
                continue
            if tag.startswith("h") and len(tag) == 2:
                parts.append(txt)
            elif tag == "li":
                parts.append(f"- {txt}")
            else:
                parts.append(txt)
            if max_chars and sum(len(p) + 1 for p in parts) >= max_chars:
                break
        except Exception:
            continue
    out = "\n".join(parts).strip()
    return out[:max_chars] if max_chars else out


def _extract_main_text(html: str, base_url: str | None = None, max_chars: int | None = None) -> str:
    """Heuristic main-content extractor with graceful fallbacks.

    Strategy:
    - Strip obvious boilerplate (nav/header/footer/aside/forms, cookie banners, promos)
    - Score candidate blocks (<article>, <main>, <section>, <div>) for text mass vs link density
    - Choose top block and include high-scoring siblings
    - Preserve headings, lists, paragraphs with newlines
    """
    soup = BeautifulSoup(html, "html.parser")
    _clean_html_noise(soup)

    # Prefer <article> / <main> quickly if present and non-trivial
    candidates = []
    for sel in ("article", "main", "section", "div"):
        candidates.extend(list(soup.find_all(sel)))
    scored = []
    for el in candidates:
        sc = _score_block(el)
        if sc > 0:
            scored.append((sc, el))
    if not scored:
        # fallback to structured text of whole page
        return _structured_text_fallback(soup, max_chars)
    scored.sort(key=lambda x: x[0], reverse=True)
    top_score, top_el = scored[0]

    # Pull in meaningful siblings from the same parent for continuity
    assembled = [top_el]
    try:
        parent = top_el.parent
        if parent:
            for sib in parent.find_all(recursive=False):
                if sib is top_el:
                    continue
                sc = _score_block(sib)
                if sc >= 0.35 * top_score:
                    assembled.append(sib)
    except Exception:
        pass

    # Assemble text
    pieces: list[str] = []
    # Optional: prepend <title>
    try:
        t = (soup.title.get_text(strip=True) if soup.title else "").strip()
        if t:
            pieces.append(t)
    except Exception:
        pass
    for el in assembled:
        txt = _assemble_text_from_block(el, max_chars)
        if txt:
            pieces.append(txt)
        if max_chars and sum(len(p) + 1 for p in pieces) >= max_chars:
            break
    out = "\n".join(pieces).strip()
    if not out:
        return _structured_text_fallback(soup, max_chars)
    return out[:max_chars] if max_chars else out


def _structured_text_fallback(soup: BeautifulSoup, max_chars: int | None) -> str:
    pieces: list[str] = []
    try:
        title = soup.title.get_text(strip=True) if soup.title else ""
        if title:
            pieces.append(title)
    except Exception:
        pass
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "pre"]):
        try:
            tag = el.name.lower()
            text = el.get_text(" ", strip=True)
            if not text:
                continue
            if tag.startswith("h") and len(tag) == 2:
                pieces.append(text)
            elif tag == "li":
                pieces.append(f"- {text}")
            else:
                pieces.append(text)
            if max_chars and sum(len(p) + 1 for p in pieces) >= max_chars:
                break
        except Exception:
            continue
    out = "\n".join(pieces).strip() or soup.get_text(" ", strip=True)
    return out[:max_chars] if max_chars else out


async def fetch_and_parse_url(session: aiohttp.ClientSession, url: str, with_meta: bool = False) -> Union[str, Tuple[str, Dict[str, Any]]]:
    """GET `url`, honour 429 back-off, return plaintext (PDF or HTML).

    Args:
        session: aiohttp client session
        url: URL to fetch and parse
        with_meta: If True, return tuple of (text, metadata_dict). If False, return just text.

    Returns:
        str if with_meta=False, or Tuple[str, Dict[str, Any]] if with_meta=True
    """
    headers = {
        "User-Agent": "FourHostsResearch/1.0 (+https://github.com/four-hosts/research-bot)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "DNT": "1",
    }

    # Timeouts – allow tuning and avoid penalising academic gateways
    timeout_sec = float(os.getenv("SEARCH_FETCH_TIMEOUT_SEC", "25"))
    if "doi.org" in url or "ssrn.com" in url:
        acad_min = float(os.getenv("SEARCH_ACADEMIC_MIN_TIMEOUT", "20"))
        timeout_sec = max(timeout_sec, acad_min)
    timeout = aiohttp.ClientTimeout(total=timeout_sec)

    async with session.get(url, headers=headers, timeout=timeout, allow_redirects=True) as r:
        if r.status == 429:
            await _rate_limit_backoff(url, dict(r.headers), session=session)
            return ("", {}) if with_meta else ""
        if r.status != 200:
            return ("", {}) if with_meta else ""

        ctype = (r.headers.get("Content-Type") or "").lower()
        is_pdf = "application/pdf" in ctype or url.lower().endswith(".pdf")
        if is_pdf and fitz is not None:
            data = await r.read()
            text = ""
            try:
                with fitz.open(stream=data, filetype="pdf") as pdf:
                    max_pages = int(os.getenv("SEARCH_PDF_MAX_PAGES", "15"))
                    max_chars = int(os.getenv("SEARCH_PDF_MAX_CHARS", "200000"))
                    structured = os.getenv("SEARCH_PDF_STRUCTURED", "1") in {"1", "true", "yes"}
                    stop_at_refs = os.getenv("SEARCH_PDF_STOP_AT_REFERENCES", "1") in {"1", "true", "yes"}
                    parts: list[str] = []
                    font_sizes: list[float] = []
                    for i, p in enumerate(pdf):
                        if i >= max_pages:
                            break
                        if structured:
                            try:
                                # Prefer dict to detect span sizes
                                d = p.get_text("dict")
                                for block in d.get("blocks", []):
                                    for line in block.get("lines", []):
                                        for span in line.get("spans", []):
                                            txt = (span.get("text") or "").strip()
                                            if not txt:
                                                continue
                                            sz = float(span.get("size") or 0.0)
                                            font_sizes.append(sz)
                                            parts.append(txt)
                                # Column order is handled by underlying extraction; fall back on blocks
                                if not parts:
                                    blocks = p.get_text("blocks")
                                    blocks = sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))
                                    for b in blocks:
                                        txt = (b[4] or "").strip()
                                        if txt:
                                            parts.append(txt)
                            except Exception:
                                parts.append(p.get_text())
                        else:
                            parts.append(p.get_text())
                        if sum(len(x) + 1 for x in parts) >= max_chars:
                            break
                    # Heuristic: stop at References section for academic PDFs
                    joined = "\n".join(parts)
                    if stop_at_refs:
                        import re as _re
                        m = _re.search(r"\n\s*(References|Bibliography|Acknowledg(e)?ments)\s*\n", joined, _re.I)
                        if m:
                            joined = joined[:m.start()].strip()
                    text = joined[:max_chars]
            except Exception:
                # Fall back to HTML parse if PDF parse fails
                text = ""

            if with_meta:
                meta = {"http": {"content_type": ctype, "last_modified": r.headers.get("Last-Modified")}}
                return text, meta
            else:
                return text

        # HTML/text parse – prefer readability/main-content extractor with fallback
        html = await r.text()
        max_chars = int(os.getenv("SEARCH_HTML_MAX_CHARS", "250000"))
        mode = (os.getenv("SEARCH_HTML_MODE", "main") or "main").lower()

        # Content extraction
        text = ""
        try:
            # Optional: use readability-lxml if available and requested
            if mode in {"readability", "auto"}:
                try:
                    from readability import Document  # type: ignore
                    doc = Document(html)
                    content_html = doc.summary() or ""
                    if content_html:
                        soup = BeautifulSoup(content_html, "html.parser")
                        text = _assemble_text_from_block(soup, max_chars)
                    else:
                        raise RuntimeError("no readability content")
                except Exception:
                    text = _extract_main_text(html, url, max_chars=max_chars)
            else:
                text = _extract_main_text(html, url, max_chars=max_chars)
        except Exception:
            try:
                soup = BeautifulSoup(html, "html.parser")
                text = _structured_text_fallback(soup, max_chars)
            except Exception:
                text = sanitize_text(html, max_len=max_chars)

        if with_meta:
            # Metadata extraction
            meta = _extract_metadata_from_html(html, headers=dict(r.headers)) if os.getenv("SEARCH_EXTRACT_METADATA", "1") in {"1", "true", "yes"} else {}
            return text, meta
        else:
            return text

# --------------------------------------------------------------------------- #
#                            FETCHER WRAPPER                                  #
# --------------------------------------------------------------------------- #


class RespectfulFetcher:
    """robots.txt aware fetcher with 1 req/sec per domain + circuit breaker."""

    def __init__(self):
        self.robot_cache: Dict[str, RobotFileParser] = {}
        self.robot_checked: Dict[str, float] = {}
        self.last_fetch: Dict[str, float] = {}
        self.ua = "FourHostsResearch/1.0 (+https://github.com/four-hosts/research-bot)"
        default_block = {"semanticscholar.org", "api.wiley.com", "onlinelibrary.wiley.com"}
        extra = {d.strip().lower() for d in os.getenv("SEARCH_FETCH_DOMAIN_BLOCKLIST", "").split(",") if d.strip()}
        self.blocked_domains = default_block | extra
        self.robots_ttl = int(os.getenv("SEARCH_ROBOTS_TTL", "86400"))

    async def _can_fetch(self, url: str) -> bool:
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        now = time.time()

        if base not in self.robot_cache or now - self.robot_checked.get(base, 0) > self.robots_ttl:
            rp = RobotFileParser()
            rp.set_url(f"{base}/robots.txt")
            try:
                await asyncio.to_thread(rp.read)
            except Exception:
                # If robots can't be read, default to allowing fetch (conservative alternative is to block)
                self.robot_cache[base], self.robot_checked[base] = rp, now
                return True
            self.robot_cache[base], self.robot_checked[base] = rp, now
        try:
            return self.robot_cache[base].can_fetch(self.ua, url)
        except Exception:
            return True

    async def _pace_domain(self, domain: str) -> None:
        """Ensure ~1 req/sec pacing per domain."""
        elapsed = time.time() - self.last_fetch.get(domain, 0)
        if elapsed < 1.0:
            await asyncio.sleep(1 - elapsed)
        self.last_fetch[domain] = time.time()

    async def fetch(self, session: aiohttp.ClientSession, url: str, on_rate_limit: Optional[Callable[[str], Awaitable[None]]] = None) -> str | None:
        url = normalize_url(url)
        if not is_valid_url(url):
            return None
        domain = extract_domain(url)

        if domain in self.blocked_domains or any(domain.endswith("." + d) for d in self.blocked_domains):
            return None
        breaker = circuit_manager.get_or_create(
            f"fetch:{domain}", failure_threshold=5, recovery_timeout=60
        )
        try:
            # Raise CircuitOpenError if breaker is open
            await breaker.call(lambda: None)
        except CircuitOpenError:
            if on_rate_limit:
                try:
                    await on_rate_limit(domain)
                except Exception:
                    pass
            return None
        if not await self._can_fetch(url):
            return None

        await self._pace_domain(domain)

        try:
            text = await breaker.call(fetch_and_parse_url, session, url, False)
            return text
        except Exception as e:
            if isinstance(e, (RateLimitedError, CircuitOpenError)) and on_rate_limit:
                try:
                    await on_rate_limit(domain)
                except Exception:
                    pass
            return None

    async def fetch_with_meta(self, session: aiohttp.ClientSession, url: str, on_rate_limit: Optional[Callable[[str], Awaitable[None]]] = None) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        url = normalize_url(url)
        if not is_valid_url(url):
            return None, None
        domain = extract_domain(url)

        if domain in self.blocked_domains or any(domain.endswith("." + d) for d in self.blocked_domains):
            return None, None
        breaker = circuit_manager.get_or_create(
            f"fetch:{domain}", failure_threshold=5, recovery_timeout=60
        )
        try:
            await breaker.call(lambda: None)
        except CircuitOpenError:
            if on_rate_limit:
                try:
                    await on_rate_limit(domain)
                except Exception:
                    pass
            return None, None
        if not await self._can_fetch(url):
            return None, None

        await self._pace_domain(domain)

        try:
            text, meta = await breaker.call(fetch_and_parse_url, session, url, True)
            return text, meta
        except Exception as e:
            if isinstance(e, (RateLimitedError, CircuitOpenError)) and on_rate_limit:
                try:
                    await on_rate_limit(domain)
                except Exception:
                    pass
            return None, None


# --------------------------------------------------------------------------- #
#                          DATA MODELS                                        #
# --------------------------------------------------------------------------- #

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str
    published_date: Optional[datetime] = None
    domain: str = ""
    credibility_score: float = 0.0
    bias_rating: Optional[str] = None
    result_type: str = "web"
    content: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 0.0
    is_primary_source: bool = False
    author: Optional[str] = None
    publication_type: Optional[str] = None
    citation_count: Optional[int] = None
    content_length: Optional[int] = None
    last_modified: Optional[datetime] = None
    content_hash: Optional[str] = None
    sections: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.domain and self.url:
            try:
                self.domain = extract_domain(self.url)
            except Exception:
                self.domain = ""
        # Prefer hashing substantive content; fall back to title+snippet signature
        try:
            if isinstance(self.content, str) and self.content.strip():
                base = (self.url or "") + "\n" + self.content.strip()
                self.content_hash = hashlib.md5(base.encode("utf-8", errors="ignore")).hexdigest()
            elif self.title or self.snippet:
                sig = f"{(self.title or '').lower().strip()}\n{(self.snippet or '').lower().strip()}"
                self.content_hash = hashlib.md5(sig.encode("utf-8", errors="ignore")).hexdigest()
        except Exception:
            # Leave unset if hashing fails
            pass


@dataclass
class SearchConfig:
    max_results: int = 50
    language: str = "en"
    region: str = "us"
    safe_search: str = "moderate"
    date_range: Optional[str] = None
    source_types: List[str] = field(default_factory=list)
    exclusion_keywords: List[str] = field(default_factory=list)
    authority_whitelist: List[str] = field(default_factory=list)
    authority_blacklist: List[str] = field(default_factory=list)
    prefer_primary_sources: bool = True
    min_relevance_score: float = 0.15  # Lowered from 0.25 to be more inclusive

    # Brave-specific
    offset: int = 0
    result_filter: Optional[List[str]] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    units: Optional[str] = None
    goggles: Optional[str] = None

    # ------------------------------------------------------------------
    # Dynamic overrides
    # ------------------------------------------------------------------

    def __post_init__(self):
        """Allow environment variables to override relevance threshold.

        Administrators can fine-tune recall/precision trade-offs at runtime
        without redeploying code by setting `SEARCH_MIN_RELEVANCE` in the
        environment (e.g. `export SEARCH_MIN_RELEVANCE=0.12`).  The value is
        clamped to the range 0–1 to avoid accidental misconfiguration.
        """
        try:
            env_val = os.getenv("SEARCH_MIN_RELEVANCE")
            if env_val is not None:
                v = float(env_val)
                # Clamp between 0.0 and 1.0
                v = max(0.0, min(1.0, v))
                self.min_relevance_score = v
                logger.debug(
                    "SearchConfig: min_relevance_score overridden via env",
                    new_value=v,
                )
        except Exception as exc:  # pragma: no cover – defensive
            logger.warning(
                "Invalid SEARCH_MIN_RELEVANCE value '%s': %s", env_val, exc
            )
    extra_snippets: bool = False
    summary: bool = False
    # Optional per-call timeout override (seconds). When set, manager/provider
    # timeouts will honor this value instead of environment defaults.
    timeout: Optional[float] = None


# --------------------------------------------------------------------------- #
#                           Query Optimiser                                   #
# --------------------------------------------------------------------------- #

# Module-level singleton to avoid repeated instantiation
_query_optimizer_instance = None


def _get_query_optimizer():
    """Get or create shared QueryOptimizer instance."""
    global _query_optimizer_instance
    if _query_optimizer_instance is None:
        try:
            from services.query_planning.optimizer import QueryOptimizer  # type: ignore
            _query_optimizer_instance = QueryOptimizer()
        except Exception:
            # Minimal fallback: expose optimize_query and get_key_terms
            class _QO:
                def optimize_query(self, q: str) -> str:
                    return q or ""

                def get_key_terms(self, q: str) -> list[str]:
                    return [t for t in re.split(r"\W+", q or "") if t]

            _query_optimizer_instance = _QO()  # type: ignore
    return _query_optimizer_instance


# --------------------------------------------------------------------------- #
#                       CONTENT RELEVANCE FILTER                              #
# --------------------------------------------------------------------------- #

class ContentRelevanceFilter:
    """Simple relevance scoring; uses shared utilities."""

    def __init__(self):
        # Use shared singleton instance for efficiency
        self.qopt = _get_query_optimizer()
        self.consensus_threshold = 0.7

    def _term_frequency(self, text: str, terms: List[str]) -> float:
        if not terms or not text:
            return 0.1  # Give minimal score instead of 0
        # Pre-compute lowercase once for efficiency
        text_lower = text.lower()
        terms_lower = [t.lower() for t in terms]
        # More forgiving: count partial matches and give bonus for any match
        matches = sum(1 for t in terms_lower if t in text_lower)
        if matches > 0:
            # At least one term matched, give a base score + frequency bonus
            return min(0.3 + (matches / len(terms)) * 0.7, 1.0)
        return 0.1  # No matches still get minimal score

    def _title_relevance(self, title: str, terms: List[str]) -> float:
        if not terms or not title:
            return 0.1  # Give minimal score instead of 0
        # Pre-compute lowercase once for efficiency
        title_lower = title.lower()
        terms_lower = [t.lower() for t in terms]
        # More forgiving for title matches
        matches = sum(1 for t in terms_lower if t in title_lower)
        if matches > 0:
            # Title matches are valuable, give higher base score
            return min(0.4 + (matches / len(terms)) * 0.6, 1.0)
        return 0.1

    def _freshness(self, dt: Optional[datetime]) -> float:
        if not dt:
            return 0.6  # Unknown date gets better default score
        age = (datetime.now(timezone.utc) - dt).days
        if age <= 7:
            return 1.0
        if age <= 30:
            return 0.85
        if age <= 90:
            return 0.7
        if age <= 365:
            return 0.55
        if age <= 730:  # 2 years
            return 0.4
        return 0.3  # Even old content gets some score

    def score(self, res: SearchResult, query: str, key_terms: List[str], cfg: SearchConfig) -> float:
        text = (res.title or "").lower() + " " + (res.snippet or "").lower()

        # Rebalanced scoring system (weights sum to ~1.0)
        # Core relevance: 60% (term freq 25%, title 20%, freshness 15%)
        score = 0.25 * self._term_frequency(text, key_terms)
        score += 0.20 * self._title_relevance((res.title or "").lower(), key_terms)
        score += 0.15 * self._freshness(res.published_date)

        # Quality signals: 25% (content completeness + domain authority)
        quality_bonus = 0.0
        if res.snippet and len(res.snippet) > 50:
            quality_bonus += 0.10
        if res.title and len(res.title) > 10:
            quality_bonus += 0.05
        if res.domain:
            domain_lower = res.domain.lower()
            if any(auth in domain_lower for auth in ['.edu', '.gov', 'wikipedia', 'nature.com', 'science.org']):
                quality_bonus += 0.10
        score += min(0.25, quality_bonus)

        # Base completeness bonus: 15% (for having basic metadata)
        score += 0.15

        return min(1.0, score)

    def filter(self, results: List[SearchResult], query: str, cfg: SearchConfig) -> List[SearchResult]:
        k = self.qopt.get_key_terms(query)
        for r in results:
            r.relevance_score = self.score(r, query, k, cfg)

        # Sort by relevance score
        sorted_results = sorted(results, key=lambda r: r.relevance_score, reverse=True)

        # Filter by threshold, but ensure we keep at least some results
        filtered = [r for r in sorted_results if r.relevance_score >= cfg.min_relevance_score]

        # Fallback: if we filtered out everything, keep top 10 results regardless of score
        if not filtered and sorted_results:
            logger.warning(f"All {len(sorted_results)} results below threshold {cfg.min_relevance_score}, keeping top 10")
            filtered = sorted_results[:10]
            # Mark these as below threshold for transparency
            for r in filtered:
                r.raw_data["below_relevance_threshold"] = True

        # Emit debug diagnostics about the filtering step (always executed)
        try:
            top_scores = [r.relevance_score for r in sorted_results[:5]]
        except Exception:
            top_scores = []

        _structured_log(
            "debug",
            "relevance_filter_complete",
            {
                "query": query[:120],
                "results_in": len(results),
                "results_passed": len(filtered),
                "threshold": cfg.min_relevance_score,
                "top_scores": top_scores,
            },
        )

        return filtered


# --------------------------------------------------------------------------- #
#                       PROGRESS REPORTING HELPER                             #
# --------------------------------------------------------------------------- #

async def _report_provider_progress(pt, rid, msg, done, total):
    """Helper function to report provider-level search progress"""
    if pt and rid:
        await pt.update_progress(rid, phase="search",
                                 message=msg,
                                 items_done=done, items_total=total)

# --------------------------------------------------------------------------- #
#                       BASE SEARCH API                                       #
# --------------------------------------------------------------------------- #


class BaseSearchAPI:
    def __init__(self, api_key: str = "", rate: int = 60):
        self.api_key = api_key
        self.rate = ClientRateLimiter(calls_per_minute=rate)
        self.session: Optional[aiohttp.ClientSession] = None
        # Use shared singleton instance for efficiency
        self.qopt = _get_query_optimizer()
        self.rfilter = ContentRelevanceFilter()

    async def __aenter__(self):
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self

    async def __aexit__(self, *_exc):
        if self.session and not self.session.closed:
            await self.session.close()

    def _sess(self) -> aiohttp.ClientSession:
        if not self.session or self.session.closed:
            # Use configurable timeout with default 30s
            timeout_sec = float(os.getenv("SEARCH_API_TIMEOUT_SEC", "30"))
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_sec))
        return self.session

    # Test seam: allow tests to inject a dummy session
    def _get_session(self) -> aiohttp.ClientSession:  # pragma: no cover - trivial alias
        return self._sess()

    async def search(self, query: str, cfg: SearchConfig) -> List[SearchResult]:
        raise NotImplementedError

    async def search_with_variations(
        self,
        seed_query: str,
        cfg: SearchConfig,
        *,
        planned: Sequence["QueryCandidate"],
    ) -> List[SearchResult]:
        """Execute the planned query candidates against this provider."""
        if not planned:
            raise ValueError("planned query candidates required for search_with_variations")

        limit = int(os.getenv("SEARCH_QUERY_VARIATIONS_LIMIT", "12"))
        concurrency = int(os.getenv("SEARCH_VARIANT_CONCURRENCY", "4"))
        ordered = list(planned)[:limit]

        seen: Set[str] = set()
        out: List[SearchResult] = []

        sem = asyncio.Semaphore(max(1, concurrency))

        async def _run_candidate(candidate: "QueryCandidate") -> List[SearchResult]:
            async with sem:
                try:
                    return await self.search(candidate.query, cfg)
                except RateLimitedError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "%s:%s failed: %s",
                        self.__class__.__name__,
                        candidate.label,
                        exc,
                    )
                    return []

        tasks = [asyncio.create_task(_run_candidate(candidate)) for candidate in ordered]
        results_by_candidate = await asyncio.gather(*tasks, return_exceptions=False)

        for candidate, results in zip(ordered, results_by_candidate):
            stage_label = f"{candidate.stage}:{candidate.label}".strip(":")
            for result in results:
                if result.url and result.url not in seen:
                    seen.add(result.url)
                    result.raw_data.setdefault("query_stage", candidate.stage)
                    result.raw_data.setdefault("query_label", candidate.label)
                    result.raw_data.setdefault("query_weight", candidate.weight)
                    if candidate.source_filter:
                        result.raw_data.setdefault("query_source_filter", candidate.source_filter)
                    if candidate.tags:
                        result.raw_data.setdefault("query_tags", candidate.tags)
                    result.raw_data["query_variant"] = stage_label
                    out.append(result)
        return out


# --------------------------------------------------------------------------- #
#                  INDIVIDUAL PROVIDER IMPLEMENTATIONS                        #
# --------------------------------------------------------------------------- #

class BraveSearchAPI(BaseSearchAPI):
    def __init__(self, api_key: str):
        super().__init__(api_key, rate=100)
        self.base = "https://api.search.brave.com/res/v1/web/search"

    @with_circuit_breaker("brave_search", failure_threshold=3, recovery_timeout=30)
    @get_search_retry_decorator()
    async def search(self, query: str, cfg: SearchConfig) -> List[SearchResult]:
        from utils.otel import otel_span  # local import to avoid optional dep
        import time

        start = time.perf_counter()
        attrs = {"provider": "brave", "success": False}

        with otel_span("rag.provider.search", attrs):
            await self.rate.wait()
            q_limit = int(os.getenv("SEARCH_PROVIDER_Q_LIMIT", "400"))
            params = {
                "q": _safe_truncate_query(self.qopt.optimize_query(query), q_limit),
                "count": min(cfg.max_results, 20),
                "search_lang": cfg.language,
            }
            headers = {"X-Subscription-Token": self.api_key, "Accept": "application/json"}
            async with self._sess().get(self.base, params=params, headers=headers) as r:
                if r.status != 200:
                    attrs["http_status"] = r.status
                    return []
                data = await r.json()
            attrs["http_status"] = 200
            attrs["success"] = True
            attrs["latency_ms"] = int((time.perf_counter() - start) * 1000)
        results: List[SearchResult] = []
        for item in data.get("web", {}).get("results", []):
            # Best-effort published date
            published = (
                (item.get("meta_url") or {}).get("cite", {}).get("datePublished")
                or item.get("page_age")
                or item.get("age")
            )
            dt = safe_parse_date(published) if isinstance(published, str) and re.search(r"\d{4}", str(published)) else None

            res = SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
                source="brave",
                published_date=dt,
                domain=extract_domain(item.get("url", "")),
                raw_data=item,
            )
            ensure_snippet_content(res)
            results.append(res)
        return self.rfilter.filter(results, query, cfg)


class GoogleCustomSearchAPI(BaseSearchAPI):
    """
    Google Custom Search JSON API.
    Requires GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX env variables.
    """

    BASE = "https://www.googleapis.com/customsearch/v1"

    def __init__(self, api_key: str, cx: str):
        super().__init__(api_key, rate=60)
        self.cx = cx

    @with_circuit_breaker("google_search", failure_threshold=3, recovery_timeout=30)
    @get_api_retry_decorator()
    async def search(self, query: str, cfg: SearchConfig) -> List[SearchResult]:
        from utils.otel import otel_span
        import time

        start = time.perf_counter()
        attrs = {"provider": "google_cse", "success": False}

        with otel_span("rag.provider.search", attrs):
            await self.rate.wait()
            # Map SearchConfig fields to official Custom Search parameters
            safe_pref = str(getattr(cfg, "safe_search", "moderate") or "").lower()
            safe_level = "off" if safe_pref in {"off", "disabled"} else "active"

            num = max(1, min(int(cfg.max_results or 10), 10))
            params: Dict[str, Any] = {
                "key": self.api_key,
                "cx": self.cx,
                "q": self.qopt.optimize_query(query),
                "num": num,
                "safe": safe_level,
                "hl": cfg.language,
            }

            lang = (cfg.language or "").lower()
            if lang:
                primary_lang = lang.split("-")[0]
                if len(primary_lang) == 2 and primary_lang.isalpha():
                    params["lr"] = f"lang_{primary_lang}"

            if cfg.region and isinstance(cfg.region, str) and len(cfg.region) == 2:
                params["gl"] = cfg.region.lower()

            if cfg.date_range:
                params["dateRestrict"] = cfg.date_range

            if cfg.exclusion_keywords:
                cleaned = sorted({kw.strip() for kw in cfg.exclusion_keywords if kw and kw.strip()})
                if cleaned:
                    params["excludeTerms"] = " ".join(cleaned)

            offset = max(0, int(getattr(cfg, "offset", 0) or 0))
            if offset:
                # API allows start indices up to 91 (start + num - 1 <= 100)
                start_idx = offset + 1
                max_start = max(1, 100 - num + 1)
                params["start"] = min(start_idx, max_start)
            # Use a reasonable timeout for Google Search
            timeout = aiohttp.ClientTimeout(total=15)
            try:
                async with self._sess().get(self.BASE, params=params, timeout=timeout) as r:
                    if r.status == 429:
                        raise RateLimitedError()
                    if 500 <= r.status <= 599:
                        raise aiohttp.ClientError(f"Google CSE {r.status}")
                    if r.status != 200:
                        attrs["http_status"] = r.status
                        return []
                    data = await r.json()
                    if isinstance(data, dict) and data.get("error"):
                        err = data.get("error", {})
                        reasons = ",".join(
                            [e.get("reason", "?") for e in err.get("errors", []) if isinstance(e, dict)]
                        )
                        if "rateLimitExceeded" in reasons or "dailyLimitExceeded" in reasons:
                            raise RateLimitedError()
                        return []
            except asyncio.TimeoutError:
                attrs["timeout"] = True
                logger.warning(f"Google Search timeout for query: {query[:50]}")
                return []

            attrs["http_status"] = 200
            attrs["success"] = True
            info = data.get("searchInformation", {}) if isinstance(data, dict) else {}
            if isinstance(info, dict):
                total = info.get("totalResults")
                if total:
                    attrs["total_results"] = total
                try:
                    if info.get("searchTime") is not None:
                        attrs["search_time_ms"] = int(float(info["searchTime"]) * 1000)
                except Exception:
                    pass
            attrs["latency_ms"] = int((time.perf_counter() - start) * 1000)
        items = data.get("items", []) or []
        results: List[SearchResult] = []
        for it in items:
            title = it.get("title", "") or ""
            url = it.get("link") or it.get("formattedUrl") or it.get("htmlFormattedUrl") or ""
            display_link = it.get("displayLink") or ""
            snippet = it.get("snippet", "") or it.get("htmlSnippet", "") or ""
            # Attempt to extract a date if present in pagemap or snippet
            pagemap = it.get("pagemap") or {}
            meta_dt = None
            # Various places dates show up
            for k in ("metatags", "scholarlyarticle", "newsarticle"):
                try:
                    entries = pagemap.get(k) or []
                    if entries:
                        e0 = entries[0]
                        for dk in ("article:published_time", "datepublished", "pubdate", "dc.date"):
                            if e0.get(dk):
                                meta_dt = safe_parse_date(e0.get(dk))
                                if meta_dt:
                                    break
                        if meta_dt:
                            break
                except Exception:
                    pass
            file_format = it.get("fileFormat")
            mime_type = it.get("mime")
            if not url and display_link and "//" not in display_link:
                domain_hint = f"https://{display_link}"
            else:
                domain_hint = url or display_link
            raw_item = dict(it)
            raw_item.setdefault("pagemap", pagemap)
            raw_item["google_meta"] = {
                "cache_id": it.get("cacheId"),
                "display_link": display_link,
                "formatted_url": it.get("formattedUrl"),
                "html_formatted_url": it.get("htmlFormattedUrl"),
                "file_format": file_format,
                "mime": mime_type,
                "labels": it.get("labels"),
            }

            res = SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                source="google_cse",
                published_date=meta_dt,
                result_type=("file" if file_format else "web"),
                domain=extract_domain(domain_hint) if domain_hint else extract_domain(url),
                raw_data=raw_item,
            )
            res.raw_data.setdefault("displayLink", display_link)
            ensure_snippet_content(res)
            results.append(res)
        return self.rfilter.filter(results, query, cfg)


# --------------------------------------------------------------------------- #
#                               ExaSearchAPI                                   #
# --------------------------------------------------------------------------- #

class ExaSearchAPI(BaseSearchAPI):
    """Exa.ai web search provider.

    Implements BaseSearchAPI.search() using Exa REST API and the shared
    resiliency/rate limiting stack in this module.
    """

    def __init__(self, api_key: str, base_url: str | None = None):
        super().__init__(api_key, rate=60)
        self.base = (base_url or os.getenv("EXA_BASE_URL") or "https://api.exa.ai").rstrip("/")
        self._search_path = "/search"

    @with_circuit_breaker("exa_search", failure_threshold=3, recovery_timeout=30)
    @get_api_retry_decorator()
    async def search(self, query: str, cfg: SearchConfig) -> List[SearchResult]:
        await self.rate.wait()
        q_limit = int(os.getenv("SEARCH_PROVIDER_Q_LIMIT", "400"))
        payload: Dict[str, Any] = {
            "query": _safe_truncate_query(self.qopt.optimize_query(query), q_limit),
            "num_results": min(cfg.max_results, 10),
        }

        if cfg.authority_whitelist:
            payload["include_domains"] = list(cfg.authority_whitelist)
        if cfg.authority_blacklist:
            payload["exclude_domains"] = list(cfg.authority_blacklist)

        # Request snippet-friendly highlights by default; allow opt-out via env.
        if os.getenv("EXA_INCLUDE_HIGHLIGHTS", "1").lower() in {"1", "true", "yes"}:
            payload["highlights"] = True

        # Optional: include full text in results
        if os.getenv("EXA_INCLUDE_TEXT", "0").lower() in {"1", "true", "yes"}:
            payload["text"] = True

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=float(os.getenv("EXA_TIMEOUT_SEC", "15")))

        async with self._sess().post(f"{self.base}{self._search_path}", headers=headers, json=payload, timeout=timeout) as r:
            if r.status == 429:
                # Let manager apply cooldown via RateLimitedError bubbling
                raise RateLimitedError()
            if r.status >= 500:
                # Trigger tenacity retry on server errors
                raise aiohttp.ClientError(f"Exa {r.status}")
            if r.status != 200:
                return []
            try:
                data = await r.json()
            except Exception:
                # Unexpected body
                return []

        items = (data or {}).get("results", []) or []
        results: List[SearchResult] = []
        for it in items:
            # Highlights list -> choose first for snippet, preserve all in raw_data
            raw_highlights = it.get("highlights")
            if isinstance(raw_highlights, list):
                highlights = raw_highlights
            elif isinstance(raw_highlights, dict):
                highlights = list(raw_highlights.get("values", []))
            else:
                highlights = []
            highlight_scores = it.get("highlightScores")
            if highlight_scores is None:
                highlight_scores = it.get("highlight_scores")

            published_raw = it.get("publishedDate") or it.get("published_date")
            parsed_date = (
                safe_parse_date(published_raw)
                if isinstance(published_raw, str)
                else None
            )
            res = SearchResult(
                title=it.get("title") or "",
                url=it.get("url") or "",
                snippet=(highlights[0] if highlights else ""),
                source="exa",
                published_date=parsed_date,
                author=(it.get("author") or None),
                raw_data={
                    "exa": {
                        "id": it.get("id"),
                        "score": it.get("score"),
                        "highlights": highlights,
                        "highlight_scores": highlight_scores,
                        "image": it.get("image"),
                        "favicon": it.get("favicon"),
                    }
                },
            )
            # Optional: include page text if returned
            if os.getenv("EXA_INCLUDE_TEXT", "0").lower() in {"1", "true", "yes"}:
                txt = it.get("text")
                if isinstance(txt, str) and txt.strip():
                    res.content = txt[: int(os.getenv("SEARCH_HTML_MAX_CHARS", "250000"))]
            ensure_snippet_content(res)
            results.append(res)
        return self.rfilter.filter(results, query, cfg)


# --------------------------------------------------------------------------- #
#                              ArxivAPI                                       #
# --------------------------------------------------------------------------- #

class ArxivAPI(BaseSearchAPI):
    """
    ArXiv does not require an API-key.  It returns Atom XML which we parse
    into SearchResult objects.
    """

    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(self):
        super().__init__(rate=30)           # be polite: 30 calls/min

    @with_circuit_breaker("arxiv_search", failure_threshold=5, recovery_timeout=60)
    async def search(self, query: str, cfg: SearchConfig) -> List[SearchResult]:
        await self.rate.wait()
        params = {
            "search_query": f"all:{self.qopt.optimize_query(query)}",
            "start": 0,
            "max_results": min(cfg.max_results, 100),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        async with self._sess().get(self.BASE_URL, params=params) as r:
            if r.status != 200:
                logger.warning(f"ArXiv API error {r.status}")
                return []
            xml_txt = await r.text()

        import xml.etree.ElementTree as ET
        results: List[SearchResult] = []
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml_txt)
        for entry in root.findall("a:entry", ns):
            title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
            url = entry.findtext("a:id", default="", namespaces=ns) or ""
            summ = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
            date_raw = entry.findtext("a:published", default="", namespaces=ns)
            dt = safe_parse_date(date_raw)

            authors = [
                a.findtext("a:name", default="", namespaces=ns)
                for a in entry.findall("a:author", ns)
            ]
            author_str = ", ".join(a for a in authors if a)[:200] or None

            res = SearchResult(
                title=title,
                url=url,
                snippet=summ[:300] + ("…" if len(summ) > 300 else ""),
                source="arxiv",
                published_date=dt,
                result_type="academic",
                domain="arxiv.org",
                author=author_str,
                publication_type="research_paper",
                raw_data={},
            )
            # Capture arXiv id if present
            aid = extract_arxiv_id(url)
            if aid:
                res.raw_data["arxiv_id"] = aid
            ensure_snippet_content(res)
            results.append(res)

        return self.rfilter.filter(results, query, cfg)


# --------------------------------------------------------------------------- #
#                              PubMedAPI                                      #
# --------------------------------------------------------------------------- #

class PubMedAPI(BaseSearchAPI):
    """
    Uses the NCBI E-Utilities (esearch + efetch) to retrieve PubMed records.
    """

    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, api_key: str | None = None):
        super().__init__(api_key or "", rate=10)

    async def _esearch(self, query: str, cfg: SearchConfig) -> List[str]:
        params = {
            "db": "pubmed",
            "term": self.qopt.optimize_query(query),
            "retmode": "json",
            "retmax": min(cfg.max_results, 200),
            "tool": "FourHostsResearch",
            "email": os.getenv("PUBMED_EMAIL", "research@fourhosts.ai"),
        }
        if self.api_key:
            params["api_key"] = self.api_key
        # Shorter timeout for index query to avoid blocking
        _to = aiohttp.ClientTimeout(total=float(os.getenv("PUBMED_ESEARCH_TIMEOUT_SEC", "12") or 12))
        async with self._sess().get(f"{self.BASE}/esearch.fcgi", params=params, timeout=_to) as r:
            if r.status != 200:
                return []
            data = await r.json()
            return data.get("esearchresult", {}).get("idlist", [])

    async def _efetch(self, ids: List[str]) -> str:
        if not ids:
            return ""
        params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml",
            "tool": "FourHostsResearch",
            "email": os.getenv("PUBMED_EMAIL", "research@fourhosts.ai"),
        }
        if self.api_key:
            params["api_key"] = self.api_key
        _to = aiohttp.ClientTimeout(total=float(os.getenv("PUBMED_EFETCH_TIMEOUT_SEC", "20") or 20))
        async with self._sess().get(f"{self.BASE}/efetch.fcgi", params=params, timeout=_to) as r:
            return await r.text() if r.status == 200 else ""

    @with_circuit_breaker("pubmed_search", failure_threshold=5, recovery_timeout=60)
    async def search(self, query: str, cfg: SearchConfig) -> List[SearchResult]:
        await self.rate.wait()
        pmids = await self._esearch(query, cfg)
        if not pmids:
            return []
        xml_txt = await self._efetch(pmids[:20])  # fetch first 20 for perf.
        if not xml_txt:
            return []

        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_txt)
        results: List[SearchResult] = []
        for art in root.findall(".//PubmedArticle"):
            pmid = art.findtext(".//PMID")
            title = (art.findtext(".//ArticleTitle") or "").strip()
            abstract = (art.findtext(".//AbstractText") or "").strip()
            year_txt = art.findtext(".//PubDate/Year")
            dt = safe_parse_date(year_txt + "-01-01") if year_txt else None
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

            authors = art.findall(".//Author")
            author_names = []
            for au in authors[:3]:
                last = au.findtext("LastName") or ""
                fore = au.findtext("ForeName") or ""
                full = f"{fore} {last}".strip()
                if full:
                    author_names.append(full)
            if authors and len(authors) > 3:
                author_str = ", ".join(author_names) + " et al."
            else:
                author_str = ", ".join(author_names)

            res = SearchResult(
                title=title,
                url=url,
                snippet=abstract[:300] + ("…" if len(abstract) > 300 else ""),
                source="pubmed",
                published_date=dt,
                result_type="academic",
                domain="pubmed.ncbi.nlm.nih.gov",
                author=author_str or None,
                publication_type="medical_research",
                raw_data={"pmid": pmid} if pmid else {},
            )
            ensure_snippet_content(res)
            results.append(res)

        return self.rfilter.filter(results, query, cfg)


# --------------------------------------------------------------------------- #
#                          SemanticScholarAPI                                 #
# --------------------------------------------------------------------------- #

class SemanticScholarAPI(BaseSearchAPI):
    """
    Free Graph API ( /graph/v1/paper/search ).  Supports optional key-rotation
    via SEMANTIC_SCHOLAR_API_KEYS env var.
    """

    BASE = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self):
        keys = [k.strip() for k in os.getenv("SEMANTIC_SCHOLAR_API_KEYS", "").split(",") if k.strip()]
        self._keys = keys or [""]
        self._key_idx = 0
        super().__init__(self._keys[0], rate=6)

    def _rotate_key(self):
        if len(self._keys) > 1:
            self._key_idx = (self._key_idx + 1) % len(self._keys)
            self.api_key = self._keys[self._key_idx]

    @with_circuit_breaker("semantic_scholar", failure_threshold=5, recovery_timeout=60)
    @get_api_retry_decorator()
    async def search(self, query: str, cfg: SearchConfig) -> List[SearchResult]:
        await self.rate.wait()
        params = {
            "query": query,
            "limit": min(cfg.max_results, 100),
            "fields": "title,abstract,authors,year,url,citationCount,influentialCitationCount",
        }
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        # Use shorter timeout for Semantic Scholar (10s)
        timeout = aiohttp.ClientTimeout(total=10)
        try:
            # Some test doubles do not accept the 'timeout' kwarg; fall back gracefully
            sess = self._get_session()
            try:
                getter = sess.get(self.BASE, params=params, headers=headers, timeout=timeout)
            except TypeError:
                getter = sess.get(self.BASE, params=params, headers=headers)
            async with getter as r:
                if r.status == 429:
                    # Sleep based on server hint and record attempts for tests
                    await _rate_limit_backoff(
                        self.BASE,
                        dict(r.headers),
                        session=self._get_session(),
                        attr_prefix="_ss_rate_attempts_",
                        prefer_server=True,
                    )
                    self._rotate_key()
                    # Allow tenacity to retry after our own backoff
                    raise RateLimitedError()
                if r.status != 200:
                    return []
                data = await r.json()
        except asyncio.TimeoutError:
            logger.warning(f"Semantic Scholar timeout for query: {query[:50]}")
            return []

        papers = data.get("data") or data.get("papers") or []
        results: List[SearchResult] = []
        for p in papers:
            title = p.get("title", "") or ""
            url = p.get("url") or (f"https://www.semanticscholar.org/paper/{p.get('paperId')}" if p.get("paperId") else "")
            snippet = (p.get("abstract") or "")[:300]
            year = p.get("year")
            dt = safe_parse_date(f"{year}-01-01") if year else None

            res = SearchResult(
                title=title,
                url=url,
                snippet=snippet + ("…" if len(snippet) == 300 else ""),
                source="semantic_scholar",
                published_date=dt,
                result_type="academic",
                domain="semanticscholar.org",
                citation_count=p.get("citationCount"),
                raw_data=p,
            )
            ensure_snippet_content(res)
            results.append(res)

        return self.rfilter.filter(results, query, cfg)


# --------------------------------------------------------------------------- #
#                              CrossRefAPI                                    #
# --------------------------------------------------------------------------- #

class CrossRefAPI(BaseSearchAPI):
    """
    CrossRef “works” endpoint – great for open-access metadata & DOIs.
    """

    BASE = "https://api.crossref.org/works"

    def __init__(self):
        super().__init__(rate=50)
        self.mailto = os.getenv("CROSSREF_EMAIL", "research@fourhosts.ai")

    @with_circuit_breaker("crossref_search", failure_threshold=5, recovery_timeout=60)
    async def search(self, query: str, cfg: SearchConfig) -> List[SearchResult]:
        await self.rate.wait()
        params = {
            "query": query,
            "rows": min(cfg.max_results, 100),
            "mailto": self.mailto,
        }
        async with self._sess().get(self.BASE, params=params) as r:
            if r.status != 200:
                return []
            data = await r.json()

        items = data.get("message", {}).get("items", [])
        results: List[SearchResult] = []
        for it in items:
            url = it.get("URL", "") or ""
            # pick a PDF if available
            for link in it.get("link", []) or []:
                if ((link.get("content-type") or "").lower() == "application/pdf") and link.get("URL"):
                    url = link.get("URL")
                    break

            title = " ".join(it.get("title", [])).strip() or "(untitled)"
            abstract = (it.get("abstract") or "").strip()
            snippet = abstract[:300] + ("…" if len(abstract) > 300 else "")

            # Year resolution
            year = None
            for k in ("published-print", "published-online", "issued", "created"):
                dp = it.get(k, {}).get("date-parts", [[]])
                if dp and dp[0]:
                    year = dp[0][0]
                    if year:
                        break
            dt = safe_parse_date(f"{year}-01-01") if year else None

            res = SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                source="crossref",
                published_date=dt,
                result_type="academic",
                raw_data=it,
            )
            doi = it.get("DOI") or extract_doi(url)
            if doi:
                res.raw_data["doi"] = doi
            ensure_snippet_content(res)
            results.append(res)

        return self.rfilter.filter(results, query, cfg)


# --------------------------------------------------------------------------- #
#                 SEARCH API MANAGER (dedup refs simplified)                  #
# --------------------------------------------------------------------------- #

class SearchAPIManager:
    def __init__(self):
        self.apis: Dict[str, BaseSearchAPI] = {}
        self.primary_api: Optional[str] = None  # Primary search provider (Brave)
        self.fallback_apis: List[str] = []  # Fallback providers in priority order (Google, etc.)
        self.fetcher = RespectfulFetcher()
        self.rfilter = ContentRelevanceFilter()
        self._fetch_sem = asyncio.Semaphore(int(os.getenv("SEARCH_FETCH_CONCURRENCY", "8")))
        self._fallback_session: Optional[aiohttp.ClientSession] = None
        # Quota exhaustion handling: provider -> unblock timestamp (epoch seconds)
        self.quota_blocked: Dict[str, float] = {}

    def add_api(self, name: str, api: BaseSearchAPI, is_primary: bool = False, is_fallback: bool = False):
        """Add a search API with optional priority designation.

        Args:
            name: API provider name
            api: BaseSearchAPI instance
            is_primary: If True, set as primary provider (Brave)
            is_fallback: If True, add to fallback list (Google)
        """
        self.apis[name] = api
        if is_primary:
            self.primary_api = name
            logger.info(f"Set {name} as primary search provider")
        elif is_fallback:
            self.fallback_apis.append(name)
            logger.info(f"Added {name} as fallback search provider")

    async def __aenter__(self):
        for api in self.apis.values():
            await api.__aenter__()
        # Prepare a fallback session only if no APIs
        if not self.apis:
            self._fallback_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self

    async def __aexit__(self, *_exc):
        for api in self.apis.values():
            await api.__aexit__(None, None, None)
        if self._fallback_session and not self._fallback_session.closed:
            await self._fallback_session.close()
            self._fallback_session = None

    def _any_session(self) -> aiohttp.ClientSession:
        try:
            api = next(iter(self.apis.values()))
            return api._sess()
        except StopIteration:
            if not self._fallback_session or self._fallback_session.closed:
                self._fallback_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
            return self._fallback_session

    async def search_with_priority(
        self,
        planned: Sequence["QueryCandidate"],
        cfg: SearchConfig,
        progress_callback: Optional[Any] = None,
        research_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search with priority: Try primary provider first, then fallbacks."""
        if not planned:
            return []
        # If neither a primary provider (e.g., Brave) nor any fallback providers
        # (e.g., Google CSE) are configured, switch to the parallel mode which
        # emits per-provider progress events. This avoids a "0%" progress UI when
        # only academic/open providers are available.
        if not self.primary_api and not self.fallback_apis:
            try:
                logger.info(
                    "No primary/fallback providers configured; using parallel mode",
                    stage="search_strategy",
                    available_apis=list(self.apis.keys()),
                )
            except Exception:
                pass
            return await self.search_all_parallel(planned, cfg, progress_callback, research_id)
        all_results = []
        now_ts = time.time()
        min_results = int(os.getenv("SEARCH_MIN_RESULTS_THRESHOLD", "5"))

        # Emit strategy selection details for observability
        try:
            logger.info(
                "Search strategy selection",
                stage="search_strategy",
                research_id=research_id,
                primary_api=self.primary_api,
                available_apis=list(self.apis.keys()),
                fallback_order=list(self.fallback_apis),
                min_results_threshold=min_results,
                plan_size=len(planned),
            )
        except Exception:
            pass

        # Step 1: Try primary API (Brave) first
        if self.primary_api and self.primary_api in self.apis:
            if self.quota_blocked.get(self.primary_api, 0) <= now_ts:
                logger.info(
                    "Searching with primary provider: %s for plan of %d candidates",
                    self.primary_api,
                    len(planned),
                )
                try:
                    _t0 = time.time()
                    results = await self._search_single_provider(
                        self.primary_api,
                        self.apis[self.primary_api],
                        planned,
                        cfg,
                        progress_callback,
                        research_id,
                    )
                    if results:
                        all_results.extend(results)
                        try:
                            resp_ms = int((time.time() - _t0) * 1000)
                            unique_domains = len({extract_domain(r.url) for r in results if getattr(r, "url", "")})
                            avg_cred = (
                                sum(getattr(r, "credibility_score", 0.0) for r in results) / len(results)
                                if results else 0.0
                            )
                            logger.info(
                                "API search complete",
                                stage="api_search_result",
                                research_id=research_id,
                                api_name=self.primary_api,
                                results_count=len(results),
                                unique_domains=unique_domains,
                                avg_credibility=round(avg_cred, 3),
                                response_time_ms=resp_ms,
                                used_cache=False,
                            )
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning(f"Primary provider {self.primary_api} failed: {e}")
                    try:
                        logger.warning(
                            "Search fallback triggered",
                            stage="search_fallback",
                            research_id=research_id,
                            primary_api=self.primary_api,
                            primary_results=len(all_results),
                            fallback_api=(self.fallback_apis[0] if self.fallback_apis else None),
                            reason="api_error",
                        )
                    except Exception:
                        pass
            else:
                logger.warning(f"Primary provider {self.primary_api} is rate-limited")
                try:
                    logger.warning(
                        "Search fallback triggered",
                        stage="search_fallback",
                        research_id=research_id,
                        primary_api=self.primary_api,
                        primary_results=len(all_results),
                        fallback_api=(self.fallback_apis[0] if self.fallback_apis else None),
                        reason="rate_limited",
                    )
                except Exception:
                    pass

        # Step 2: If primary didn't return enough results, try fallbacks (Google)
        if len(all_results) < min_results:
            for fallback_name in self.fallback_apis:
                if fallback_name in self.apis and self.quota_blocked.get(fallback_name, 0) <= now_ts:
                    try:
                        logger.info(
                            "Insufficient results, trying fallback",
                            stage="search_fallback",
                            research_id=research_id,
                            primary_api=self.primary_api,
                            primary_results=len(all_results),
                            fallback_api=fallback_name,
                            reason="insufficient_results",
                        )
                    except Exception:
                        pass
                    try:
                        _t1 = time.time()
                        fallback_results = await self._search_single_provider(
                            fallback_name,
                            self.apis[fallback_name],
                            planned,
                            cfg,
                            progress_callback,
                            research_id,
                        )
                        if fallback_results:
                            all_results.extend(fallback_results)
                            try:
                                resp_ms = int((time.time() - _t1) * 1000)
                                unique_domains = len({extract_domain(r.url) for r in fallback_results if getattr(r, "url", "")})
                                avg_cred = (
                                    sum(getattr(r, "credibility_score", 0.0) for r in fallback_results) / len(fallback_results)
                                    if fallback_results else 0.0
                                )
                                logger.info(
                                    "API search complete",
                                    stage="api_search_result",
                                    research_id=research_id,
                                    api_name=fallback_name,
                                    results_count=len(fallback_results),
                                    unique_domains=unique_domains,
                                    avg_credibility=round(avg_cred, 3),
                                    response_time_ms=resp_ms,
                                    used_cache=False,
                                )
                            except Exception:
                                pass
                    except Exception as e:
                        logger.warning(f"Fallback provider {fallback_name} failed: {e}")

        # Step 3: Run academic providers in parallel (always)
        academic_results = await self._search_academic_parallel(
            planned,
            cfg,
            progress_callback,
            research_id,
        )
        all_results.extend(academic_results)

        # Process and deduplicate results
        return await self._process_results(all_results, cfg)

    async def _search_single_provider(
        self,
        name: str,
        api: BaseSearchAPI,
        planned: Sequence["QueryCandidate"],
        cfg: SearchConfig,
        progress_callback: Optional[Any],
        research_id: Optional[str],
    ) -> List[SearchResult]:
        """Search using a single provider with timeout and error handling."""
        if not planned:
            return []
        seed_query = planned[0].query
        if progress_callback and research_id:
            try:
                await progress_callback.report_search_started(
                    research_id, seed_query, name, 1, 1
                )
            except Exception:
                pass

        # Prefer config.timeout if provided; otherwise fall back to env variable.
        try:
            timeout = float(cfg.timeout) if getattr(cfg, "timeout", None) else float(os.getenv("SEARCH_PER_PROVIDER_TIMEOUT_SEC", "15"))
        except Exception:
            timeout = float(os.getenv("SEARCH_PER_PROVIDER_TIMEOUT_SEC", "15"))
        try:
            _t0 = time.time()
            with _otel_span(
                "rag.provider.search",
                {
                    "provider": name,
                    "seed_query": seed_query[:120],
                    "planned_count": len(planned),
                },
            ) as _sp:
                results = await asyncio.wait_for(
                    api.search_with_variations(seed_query, cfg, planned=planned),
                    timeout=timeout,
                )
                try:
                    if _sp:
                        _sp.set_attribute("results.count", len(results) if results else 0)
                        _sp.set_attribute("latency_ms", int((time.time() - _t0) * 1000))
                        _sp.set_attribute("success", True)
                except Exception:
                    pass

            if progress_callback and research_id:
                try:
                    await progress_callback.report_search_completed(
                        research_id,
                        seed_query,
                        len(results) if results else 0,
                    )
                except Exception:
                    pass

            return results if results else []
        except asyncio.TimeoutError:
            logger.warning(f"Provider timeout ({timeout:.1f}s): {name}")
            raise
        except RateLimitedError:
            # Apply rate limit cooldown
            cooldown = float(os.getenv("SEARCH_QUOTA_COOLDOWN_SEC", "3600"))
            self.quota_blocked[name] = time.time() + cooldown
            logger.warning(f"{name} rate limited - cooling down for {int(cooldown)}s")
            raise
        except CircuitOpenError as e:
            logger.warning(f"{name} circuit breaker open: {e}")
            raise

    async def _search_academic_parallel(
        self,
        planned: Sequence["QueryCandidate"],
        cfg: SearchConfig,
        progress_callback: Optional[Any],
        research_id: Optional[str],
    ) -> List[SearchResult]:
        """Search academic providers in parallel using the planned candidates."""
        if not planned:
            return []
        academic_providers = ['arxiv', 'crossref', 'semanticscholar', 'pubmed']
        tasks = {}
        now_ts = time.time()
        seed_query = planned[0].query

        # Identify enabled academic providers
        enabled_academic: List[str] = []
        for name in academic_providers:
            if name in self.apis and self.quota_blocked.get(name, 0) <= now_ts:
                api = self.apis[name]
                enabled_academic.append(name)
                tasks[name] = asyncio.create_task(
                    self._search_provider_silent(name, api, planned, cfg,
                                                  progress_callback=progress_callback,
                                                  research_id=research_id,
                                                  seed_query=seed_query)
                )

        # Emit provider-level start events so UI reflects progress during
        # academic-only search runs.
        if progress_callback and research_id and enabled_academic:
            try:
                total = len(enabled_academic)
                for idx, pname in enumerate(enabled_academic, start=1):
                    await progress_callback.report_search_started(
                        research_id,
                        seed_query,
                        pname,
                        idx,
                        total,
                    )
            except Exception:
                pass

        if not tasks:
            return []

        results = []
        # Prefer per-call timeout if provided
        try:
            timeout = float(cfg.timeout) if getattr(cfg, "timeout", None) else float(os.getenv("SEARCH_ACADEMIC_TIMEOUT_SEC", "20"))
        except Exception:
            timeout = float(os.getenv("SEARCH_ACADEMIC_TIMEOUT_SEC", "20"))
        done, pending = await asyncio.wait(tasks.values(), timeout=timeout, return_when=asyncio.ALL_COMPLETED)

        # Map tasks back to provider names for progress completion events
        name_by_task = {task: name for name, task in tasks.items()}

        for task in done:
            task_results = []
            try:
                task_results = task.result()
                if task_results:
                    results.extend(task_results)
            except Exception:
                pass
            # Note: completion events are reported by _search_provider_silent

        # Cancel pending tasks and report completion with zero results
        for task in pending:
            task.cancel()
            if progress_callback and research_id:
                try:
                    pname = name_by_task.get(task)
                    if pname:
                        await progress_callback.report_search_completed(research_id, seed_query, 0)
                except Exception:
                    pass

        # Drain cancelled tasks to prevent warnings and resource leaks
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        return results

    async def _search_provider_silent(
        self,
        name: str,
        api: BaseSearchAPI,
        planned: Sequence["QueryCandidate"],
        cfg: SearchConfig,
        *,
        progress_callback: Optional[Any] = None,
        research_id: Optional[str] = None,
        seed_query: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search without progress reporting (for parallel academic searches)."""
        if not planned:
            return []
        seed_query = seed_query or planned[0].query
        try:
            timeout = float(os.getenv("SEARCH_ACADEMIC_TIMEOUT_SEC", "20"))
            _t0 = time.time()
            _t0 = time.time()
            with _otel_span(
                "rag.provider.search",
                {
                    "provider": name,
                    "seed_query": seed_query[:120],
                    "planned_count": len(planned),
                },
            ) as _sp:
                results = await asyncio.wait_for(
                    api.search_with_variations(seed_query, cfg, planned=planned),
                    timeout=timeout,
                )
                try:
                    if _sp:
                        _sp.set_attribute("results.count", len(results) if results else 0)
                        _sp.set_attribute("latency_ms", int((time.time() - _t0) * 1000))
                        _sp.set_attribute("success", True)
                except Exception:
                    pass
            try:
                resp_ms = int((time.time() - _t0) * 1000)
                unique_domains = len({extract_domain(r.url) for r in (results or []) if getattr(r, "url", "")})
                avg_cred = (
                    sum(getattr(r, "credibility_score", 0.0) for r in (results or [])) / len(results)
                    if results else 0.0
                )
                logger.info(
                    "API search complete",
                    stage="api_search_result",
                    api_name=name,
                    results_count=len(results) if results else 0,
                    unique_domains=unique_domains,
                    avg_credibility=round(avg_cred, 3),
                    response_time_ms=resp_ms,
                    used_cache=False,
                )
            except Exception:
                pass
            # Emit completion for this provider (even in silent mode) when a
            # progress callback is available – helps avoid a stuck 0% UI.
            if progress_callback and research_id:
                try:
                    await progress_callback.report_search_completed(
                        research_id,
                        seed_query,
                        len(results) if results else 0,
                    )
                except Exception:
                    pass
            return results
        except Exception as e:
            logger.debug(f"Academic provider {name} failed: {e}")
            return []

    async def _process_results(self, results: List[SearchResult], cfg: SearchConfig) -> List[SearchResult]:
        """Deduplicate results and limit to max_results."""
        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for res in results:
            if res.url and res.url not in seen_urls:
                seen_urls.add(res.url)
                unique_results.append(res)

        # Sort by relevance/credibility if available
        unique_results.sort(key=lambda r: getattr(r, 'credibility_score', 0.5), reverse=True)

        # Limit to max results
        if cfg.max_results:
            unique_results = unique_results[:cfg.max_results]

        try:
            logger.info(
                "Result deduplication complete",
                stage="deduplication",
                input_count=len(results),
                output_count=len(unique_results),
                duplicates_removed=(len(results) - len(unique_results)),
                dedup_methods=["url_norm"],
                unique_domains=len({getattr(r, 'domain', '') for r in unique_results if getattr(r, 'domain', '')}),
            )
        except Exception:
            pass

        return unique_results

    async def search_all(
        self,
        planned: Sequence["QueryCandidate"],
        cfg: SearchConfig,
        progress_callback: Optional[Any] = None,
        research_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """Main search method executing the provided plan."""
        if not planned:
            return []
        # Use the new priority-based search method
        use_priority = os.getenv("SEARCH_USE_PRIORITY_MODE", "1") not in {"0", "false", "no"}
        if use_priority:
            return await self.search_with_priority(planned, cfg, progress_callback, research_id)

        # Legacy parallel search mode (if priority mode is disabled)
        return await self.search_all_parallel(planned, cfg, progress_callback, research_id)

    async def search_all_parallel(
        self,
        planned: Sequence["QueryCandidate"],
        cfg: SearchConfig,
        progress_callback: Optional[Any] = None,
        research_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """Parallel search method supporting planned candidates."""
        if not planned:
            return []
        seed_query = planned[0].query
        # Provider-specific start events so UI shows provider progress
        if progress_callback and research_id:
            try:
                total = max(1, len(self.apis))
                for idx, name in enumerate(self.apis.keys()):
                    await progress_callback.report_search_started(
                        research_id,
                        seed_query,
                        name,
                        idx + 1,
                        total,
                    )
            except Exception:
                pass

        # Launch one task per provider with an individual timeout budget
        # so a single slow provider cannot stall the overall search.
        tasks: Dict[str, asyncio.Task] = {}
        # Resolve per-provider timeout; prefer explicit env, then fall back to overall task timeout
        import os as _os
        try:
            # Configured timeout overrides env defaults when provided
            _per_provider_to = float(getattr(cfg, "timeout", 0.0) or 0.0)
            if not _per_provider_to:
                _per_provider_to = float(_os.getenv("SEARCH_PER_PROVIDER_TIMEOUT_SEC") or 0.0)
            if not _per_provider_to:
                _per_provider_to = float(_os.getenv("SEARCH_PROVIDER_TIMEOUT_SEC") or 0.0)
            if not _per_provider_to:
                _per_provider_to = float(_os.getenv("SEARCH_TASK_TIMEOUT_SEC", "25") or 25)
        except Exception:
            _per_provider_to = 25.0
        # Respect quota blocks
        now_ts = time.time()
        for name, api in self.apis.items():
            if self.quota_blocked.get(name, 0) > now_ts:
                # Provider is cooling down due to quota exhaustion
                if progress_callback and research_id:
                    try:
                        from services.websocket_service import connection_manager, WSMessage, WSEventType
                        await connection_manager.broadcast_to_research(
                            research_id,
                            WSMessage(
                                type=WSEventType.RATE_LIMIT_WARNING,
                                data={
                                    "research_id": research_id,
                                    "limit_type": "quota_exhausted",
                                    "provider": name,
                                    "message": f"Provider {name} temporarily disabled due to quota exhaustion",
                                    "cooldown_until": datetime.fromtimestamp(self.quota_blocked[name], tz=timezone.utc).isoformat(),
                                },
                            ),
                        )
                    except Exception:
                        pass
                continue
            async def _run_with_timeout(_api: BaseSearchAPI) -> List[SearchResult]:
                try:
                    _t0 = time.time()
                    with _otel_span(
                        "rag.provider.search",
                        {
                            "provider": getattr(_api, "__class__", type(_api)).__name__,
                            "seed_query": seed_query[:120],
                            "planned_count": len(planned),
                        },
                    ) as _sp:
                        res = await asyncio.wait_for(
                            _api.search_with_variations(seed_query, cfg, planned=planned),
                            timeout=_per_provider_to,
                        )
                        try:
                            if _sp:
                                _sp.set_attribute("results.count", len(res) if res else 0)
                                _sp.set_attribute("latency_ms", int((time.time() - _t0) * 1000))
                                _sp.set_attribute("success", True)
                        except Exception:
                            pass
                        return res
                except asyncio.TimeoutError:
                    logger.warning(f"Provider timeout ({_per_provider_to:.1f}s): {getattr(_api, '__class__', type(_api)).__name__}")
                    return []
            tasks[name] = asyncio.create_task(_run_with_timeout(api))
        all_res: List[SearchResult] = []

        if tasks:
            # Wait for all providers to finish within their individual budgets.
            name_by_task = {t: n for n, t in tasks.items()}
            done, pending = await asyncio.wait(set(tasks.values()), timeout=max(_per_provider_to + 0.1, 0.1))

            # Collect finished results
            for t in done:
                name = name_by_task.get(t, "unknown")
                try:
                    res = t.result()
                    # Provider-specific completion event
                    if progress_callback and research_id:
                        try:
                            count = len(res) if isinstance(res, list) else 0
                            await progress_callback.report_search_completed(research_id, seed_query, count)
                        except Exception:
                            pass
                    all_res.extend(res if isinstance(res, list) else [])
                except RateLimitedError:
                    # Quota exhausted for this provider – apply cooldown
                    try:
                        cooldown = float(_os.getenv("SEARCH_QUOTA_COOLDOWN_SEC", "3600") or 3600)
                    except Exception:
                        cooldown = 3600.0
                    self.quota_blocked[name] = time.time() + cooldown
                    logger.warning(f"{name} quota exhausted – cooling down for {int(cooldown)}s")
                    if progress_callback and research_id:
                        try:
                            from services.websocket_service import connection_manager, WSMessage, WSEventType
                            await connection_manager.broadcast_to_research(
                                research_id,
                                WSMessage(
                                    type=WSEventType.RATE_LIMIT_WARNING,
                                    data={
                                        "research_id": research_id,
                                        "limit_type": "quota_exhausted",
                                        "provider": name,
                                        "cooldown_sec": int(cooldown),
                                        "message": f"Provider {name} disabled due to quota exhaustion",
                                    },
                                ),
                            )
                        except Exception:
                            pass
                except Exception as e:
                    logger.error(f"{name} failed: {e}")

            # Cancel any truly stuck tasks (defensive; they should have timed out individually)
            for t in pending:
                name = name_by_task.get(t, "unknown")
                try:
                    t.cancel()
                except Exception:
                    pass
                logger.warning(f"{name} timed out; skipping")

            # Drain cancelled tasks to prevent warnings and resource leaks
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        else:
            all_res = []

        # Fetch full content concurrently
        async def _fetch(res: SearchResult):
            if res.content or not res.url:
                return
            session = self._any_session()
            async with self._fetch_sem:
                async def _emit_rate_limit(domain: str, provider: Optional[str] = None):
                    if progress_callback and research_id:
                        try:
                            from services.websocket_service import connection_manager, WSMessage, WSEventType
                            await connection_manager.broadcast_to_research(
                                research_id,
                                WSMessage(
                                    type=WSEventType.RATE_LIMIT_WARNING,
                                    data={
                                        "research_id": research_id,
                                        "limit_type": "search_provider",
                                        "provider": provider or res.source,
                                        "domain": domain,
                                        "message": f"Rate limit encountered on {provider or res.source} for {domain}",
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                    },
                                ),
                            )
                        except Exception:
                            pass
                text, meta = await self.fetcher.fetch_with_meta(session, res.url, on_rate_limit=lambda domain: _emit_rate_limit(domain, res.source))
            if text:
                res.content = text
            if meta:
                # Stash structured metadata and opportunistically fill fields
                try:
                    res.raw_data.setdefault("extracted_meta", meta)
                    if isinstance(meta, dict):
                        canonical_url = meta.get("canonical_url")
                        if isinstance(canonical_url, str) and canonical_url.strip():
                            try:
                                normalized = normalize_url(canonical_url)
                                if is_valid_url(normalized):
                                    res.url = normalized
                                    res.domain = extract_domain(normalized)
                            except Exception:
                                pass
                    pub = (meta.get("published_date") if isinstance(meta, dict) else None)
                    if pub and not getattr(res, "published_date", None):
                        dt = safe_parse_date(str(pub))
                        if dt:
                            res.published_date = dt
                    authors = meta.get("authors") if isinstance(meta, dict) else None
                    if authors and not getattr(res, "author", None):
                        if isinstance(authors, list):
                            res.author = ", ".join(authors)[:200]
                        elif isinstance(authors, str):
                            res.author = authors[:200]
                    content_type = None
                    try:
                        http_meta = meta.get("http") if isinstance(meta, dict) else None
                        if isinstance(http_meta, dict):
                            content_type = http_meta.get("content_type")
                    except Exception:
                        content_type = None
                    res.raw_data["source_category"] = categorize(
                        getattr(res, "domain", ""),
                        content_type,
                    )
                except Exception:
                    pass

        # Optional budgeted content fetch phase to avoid exceeding orchestrator timeouts
        enable_fetch = os.getenv("SEARCH_ENABLE_CONTENT_FETCH", "1").lower() in {"1", "true", "yes", "on"}
        if enable_fetch and all_res:
            # Determine total time budget for this fetch stage
            try:
                fetch_budget = float(os.getenv("SEARCH_FETCH_TOTAL_BUDGET_SEC", "0") or 0.0)
            except Exception:
                fetch_budget = 0.0
            if fetch_budget <= 0.0:
                # Dynamic content fetch budget based on result count
                # Minimum 10s to give content fetching a real chance
                # Scale with results: 0.5s per result up to 30s max
                result_count = len(all_res)
                fetch_budget = min(30.0, max(10.0, 0.5 * result_count))

            # Sort results by relevance score to prioritize top results
            sorted_results = sorted(all_res,
                                  key=lambda r: getattr(r, 'relevance_score', 0.5),
                                  reverse=True)

            # Launch fetch tasks with prioritization
            fetch_tasks = [asyncio.create_task(_fetch(r)) for r in sorted_results]
            done, pending = await asyncio.wait(set(fetch_tasks), timeout=fetch_budget)
            if pending:
                for t in pending:
                    try:
                        t.cancel()
                    except Exception:
                        pass
                # Drain cancellations
                try:
                    await asyncio.gather(*pending, return_exceptions=True)
                except Exception:
                    pass
                logger.warning("Content fetch budget exhausted after %.1fs; %d fetches cancelled", fetch_budget, len(pending))
        # Final normalization regardless of fetch
        for r in all_res:
            ensure_snippet_content(r)
            normalize_result_text_fields(r)

        # Provider-level Jaccard dedup removed; URL dedup happens upstream
        # and orchestrator-level ResultDeduplicator performs advanced dedup.
        return self.rfilter.filter(all_res, seed_query, cfg)

    async def search_with_plan(
        self,
        planned: Sequence["QueryCandidate"],
        config: SearchConfig = None
    ) -> Dict[str, List[SearchResult]]:
        """Execute planned queries across providers in a single prioritized call.

        This variant preserves the original call signature expected by tests:
        - Delegates once to search_with_priority(planned, config)
        - Returns a dict[label] -> List[SearchResult]
        - Ensures keys exist for all candidate labels (possibly with empty lists)
        """
        logger.info(
            "Starting search_with_plan execution",
            planned_count=len(planned) if planned else 0,
            queries=[getattr(c, "query", "")[:50] for c in (planned or [])[:3]]  # Log first 3 queries
        )
        if not planned:
            logger.warning("No planned queries provided, returning empty results")
            return {}

        if config is None:
            config = SearchConfig()

        # Pre-populate mapping for all labels so callers can rely on presence
        results_by_label: Dict[str, List[SearchResult]] = {
            getattr(c, "label", "unknown"): [] for c in planned
        }

        try:
            # Single call to priority path as expected by integration tests
            logger.info(
                "Executing search with priority",
                config_paradigm=getattr(config, "paradigm", None) if config else None
            )
            stage_results = await self.search_with_priority(planned, config)
            logger.info(
                "Search with priority completed",
                results_count=len(stage_results) if stage_results else 0
            )
        except Exception as e:
            # On any error, return empty mapping
            logger.error(
                "Search with priority failed",
                error=str(e),
                exc_info=True
            )
            return {}

        # Organize results by candidate label
        for result in stage_results or []:
            raw_data = result.raw_data if isinstance(result.raw_data, dict) else {}
            label = raw_data.get("query_label")
            stage = raw_data.get("query_stage")

            if not isinstance(result.raw_data, dict):
                result.raw_data = raw_data

            provider = (
                getattr(result, "source_api", None)
                or getattr(result, "source", None)
                or raw_data.get("provider")
                or raw_data.get("source")
                or "unknown"
            )

            result.raw_data["cost_attribution"] = {
                "stage": stage,
                "label": label,
                "provider": provider,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            if isinstance(label, str):
                results_by_label.setdefault(label, []).append(result)
            if isinstance(label, str) and isinstance(stage, str):
                result.raw_data["stage_label"] = f"{stage}:{label}"

        logger.info(
            "Search with plan completed",
            labels_with_results=list(results_by_label.keys()),
            total_results=sum(len(r) for r in results_by_label.values())
        )
        return results_by_label


# --------------------------------------------------------------------------- #
#                     FACTORY / QUICK DEMO                                    #
# --------------------------------------------------------------------------- #

def create_search_manager() -> SearchAPIManager:
    mgr = SearchAPIManager()

    # Brave as PRIMARY provider
    brave_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if brave_key and os.getenv("SEARCH_DISABLE_BRAVE", "0") not in {"1", "true", "yes"}:
        mgr.add_api("brave", BraveSearchAPI(brave_key), is_primary=True)

    # Google as FALLBACK provider
    # Support both canonical and legacy env var names for Google CSE
    g_key = os.getenv("GOOGLE_CSE_API_KEY") or os.getenv("GOOGLE_SEARCH_API_KEY")
    g_cx = os.getenv("GOOGLE_CSE_CX") or os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    if g_key and g_cx and os.getenv("SEARCH_DISABLE_GOOGLE", "0") not in {"1", "true", "yes"}:
        mgr.add_api("google_cse", GoogleCustomSearchAPI(g_key, g_cx), is_fallback=True)

    # Exa provider (configurable as primary or fallback)
    exa_key = os.getenv("EXA_API_KEY")
    if exa_key and os.getenv("SEARCH_DISABLE_EXA", "0").lower() not in {"1", "true", "yes"}:
        exa_api = ExaSearchAPI(exa_key, base_url=os.getenv("EXA_BASE_URL"))
        exa_primary = os.getenv("EXA_SEARCH_AS_PRIMARY", "0").lower() in {"1", "true", "yes"}
        mgr.add_api("exa", exa_api, is_primary=exa_primary, is_fallback=not exa_primary)

    # Academic/open providers
    if os.getenv("SEARCH_DISABLE_ARXIV", "0") not in {"1", "true", "yes"}:
        mgr.add_api("arxiv", ArxivAPI())
    if os.getenv("SEARCH_DISABLE_CROSSREF", "0") not in {"1", "true", "yes"}:
        mgr.add_api("crossref", CrossRefAPI())
    if os.getenv("SEARCH_DISABLE_SEMANTICSCHOLAR", "0") not in {"1", "true", "yes"}:
        mgr.add_api("semanticscholar", SemanticScholarAPI())
    pm_key = os.getenv("PUBMED_API_KEY")  # optional
    if os.getenv("SEARCH_DISABLE_PUBMED", "0") not in {"1", "true", "yes"}:
        mgr.add_api("pubmed", PubMedAPI(pm_key))

    return mgr


async def _demo():
    async with create_search_manager() as mgr:
        cfg = SearchConfig(max_results=10)
        from search.query_planner import QueryCandidate  # local import to avoid circular at module load
        planned = [
            QueryCandidate(
                query="artificial intelligence ethics",
                stage="context",
                label="demo",
            )
        ]
        res = await mgr.search_all(planned, cfg)
        logger = structlog.get_logger(__name__)
        for r in res[:5]:
            logger.info("demo result", title=r.title, url=r.url)


if __name__ == "__main__":
    asyncio.run(_demo())
