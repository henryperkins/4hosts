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
import logging
import os
import random
import re
import string
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union, cast, Callable, Awaitable
import html as _html
from urllib.parse import unquote, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import aiohttp
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None  # Optional; PDF parsing will be skipped if unavailable
import nltk
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Circuit breaker for resilient external API calls
from utils.circuit_breaker import with_circuit_breaker, CircuitOpenError


# --------------------------------------------------------------------------- #
#                        ENV / LOGGING / INITIALISATION                       #
# --------------------------------------------------------------------------- #

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

STOP_WORDS: Set[str]


def _ensure_nltk_ready() -> bool:
    """Return True if punkt/stopwords/wordnet are available (download if env allows)."""
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/stopwords")
        nltk.data.find("corpora/wordnet")
        return True
    except LookupError:
        if os.getenv("SEARCH_ALLOW_NLTK_DOWNLOADS") == "1":
            try:
                nltk.download("punkt", quiet=True)
                nltk.download("stopwords", quiet=True)
                nltk.download("wordnet", quiet=True)
                return True
            except Exception:
                return False
        return False


_use_nltk = _ensure_nltk_ready()
if _use_nltk:
    try:
        STOP_WORDS = set(nltk.corpus.stopwords.words("english"))
    except Exception:
        STOP_WORDS = {
            "the", "a", "an", "and", "or", "but", "of", "in", "on", "for", "to", "with", "by",
            "is", "are", "was", "were", "be", "as", "at", "it", "this", "that", "from",
        }
else:
    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "of", "in", "on", "for", "to", "with", "by",
        "is", "are", "was", "were", "be", "as", "at", "it", "this", "that", "from",
    }


def safe_parse_date(raw: str | None) -> Optional[datetime]:
    """Return timezone-aware datetime or None."""
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()
    if not raw:
        return None
    raw = raw.replace("Z", "+00:00")
    try:
        # Try ISO-style first
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass

    # Fallbacks: year-only or YYYY-MM
    m = re.match(r"^\s*(\d{4})\s*$", raw)
    if m:
        try:
            return datetime(int(m.group(1)), 1, 1, tzinfo=timezone.utc)
        except Exception:
            return None
    m = re.match(r"^\s*(\d{4})-(\d{1,2})\s*$", raw)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        try:
            return datetime(y, mo, 1, tzinfo=timezone.utc)
        except Exception:
            return None
    return None


def tokenize(text: str, *, lower=True) -> List[str]:
    text = text.lower() if lower else text
    tokens = re.findall(r"\w+", text)
    return [t for t in tokens if t not in STOP_WORDS]


def ngram_tokenize(text: str, n: int = 3) -> List[str]:
    toks = tokenize(text)
    return [" ".join(toks[i: i + n]) for i in range(max(0, len(toks) - n + 1))]


def extract_doi(text: str | None) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", text)
    return m.group(0) if m else None


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


def _strip_tags(text: str) -> str:
    """Lightweight HTML tag and entity cleanup used for titles/snippets.

    - Unescape HTML entities (so &lt; becomes <)
    - Remove all remaining <...> tags including publisher-specific like <jats:p>
    - Collapse whitespace
    """
    if not text:
        return ""
    try:
        import re as _re
        # Decode HTML entities first
        t = _html.unescape(str(text))
        # Remove tags
        t = _re.sub(r"<[^>]+>", " ", t)
        # Normalize whitespace
        t = _re.sub(r"\s+", " ", t).strip()
        return t
    except Exception:
        return str(text)


def normalize_result_text_fields(result: "SearchResult") -> None:
    """Normalize user-visible fields for safe display/logging.

    Called after result construction/fetch to ensure provider markup does not
    leak into progress logs or the UI (e.g. <jats:p>, Word <span data-...>).
    """
    try:
        max_title = int(os.getenv("SEARCH_TITLE_MAX_LEN", "300"))
        result.title = _strip_tags(result.title)[:max_title]
    except Exception:
        pass
    try:
        max_snippet = int(os.getenv("SEARCH_SNIPPET_MAX_LEN", "800"))
        result.snippet = _strip_tags(result.snippet)[:max_snippet]
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

class RateLimitedError(aiohttp.ClientError):
    """Raised on HTTP-429 to let tenacity back-off."""


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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, RateLimitedError)),
)
def _retry_after_to_seconds(val: str) -> float:
    """Parse Retry-After which may be seconds or HTTP-date."""
    try:
        if val.isdigit():
            return float(val)
    except Exception:
        pass
    # Try HTTP-date
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(val)
        if dt:
            return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return 0.0
    return 0.0

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
        link_canon = soup.find("link", rel=lambda v: v and "canonical" in str(v).lower())
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


def _derive_source_category(domain: str, content_type: Optional[str] = None) -> str:
    """Best-effort categorisation for UI grouping."""
    dom = (domain or "").lower()
    if dom.endswith(".edu") or "arxiv" in dom or "pubmed" in dom or "semanticscholar" in dom or "crossref" in dom:
        return "academic"
    if dom.endswith(".gov") or dom.endswith(".mil"):
        return "gov"
    if "youtube.com" in dom or "vimeo.com" in dom:
        return "video"
    if content_type and "pdf" in content_type.lower():
        return "pdf"
    if any(news in dom for news in [
        "nytimes.com",
        "washingtonpost.com",
        "theguardian.com",
        "wsj.com",
        "ft.com",
        "bloomberg.com",
        "bbc.co.uk",
        "cnn.com",
        "reuters.com",
    ]):
        return "news"
    if dom.endswith(".io") or dom.endswith(".dev"):
        return "blog"
    return "other"

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
        return _html_to_structured_text_legacy(soup, max_chars)
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
        return _html_to_structured_text_legacy(soup, max_chars)
    return out[:max_chars] if max_chars else out

def _html_to_structured_text_legacy(soup: BeautifulSoup, max_chars: int | None) -> str:
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

async def fetch_and_parse_url(session: aiohttp.ClientSession, url: str) -> str:
    """GET `url`, honour 429 back-off, return plaintext (PDF or HTML)."""
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
            retry_hdr = r.headers.get("retry-after", "")
            delay = _retry_after_to_seconds(retry_hdr) or random.uniform(5, 10)
            # Add jittered exponential floor via env knobs
            floor = float(os.getenv("SEARCH_429_MIN_DELAY", "3.0"))
            delay = max(delay, floor) * random.uniform(0.9, 1.2)
            _structured_log("warning", "rate_limited_fetch", {"url": url, "delay": round(delay,2)})
            await asyncio.sleep(delay)
            raise RateLimitedError()  # let tenacity retry
        if r.status != 200:
            return ""

        ctype = (r.headers.get("Content-Type") or "").lower()
        is_pdf = "application/pdf" in ctype or url.lower().endswith(".pdf")
        if is_pdf and fitz is not None:
            data = await r.read()
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
                    return joined[:max_chars]
            except Exception:
                # Fall back to HTML parse if PDF parse fails
                pass

        # HTML/text parse – prefer readability/main-content extractor with fallback
        html = await r.text()
        max_chars = int(os.getenv("SEARCH_HTML_MAX_CHARS", "250000"))
        mode = (os.getenv("SEARCH_HTML_MODE", "main") or "main").lower()
        # Optional: use readability-lxml if available and requested
        if mode in {"readability", "auto"}:
            try:
                from readability import Document  # type: ignore
                doc = Document(html)
                content_html = doc.summary() or ""
                if content_html:
                    soup = BeautifulSoup(content_html, "html.parser")
                    return _assemble_text_from_block(soup, max_chars)
            except Exception:
                # Fall through to main extractor
                pass
        # Heuristic main extractor
        if mode in {"main", "auto"}:
            try:
                return _extract_main_text(html, url, max_chars=max_chars)
            except Exception:
                pass
        # Legacy structured extractor
        try:
            soup = BeautifulSoup(html, "html.parser")
            return _html_to_structured_text_legacy(soup, max_chars)
        except Exception:
            return _strip_tags(html)[:max_chars]


async def fetch_and_parse_url_with_meta(session: aiohttp.ClientSession, url: str) -> Tuple[str, Dict[str, Any]]:
    """Like fetch_and_parse_url but also returns extracted metadata as a dict."""
    headers = {
        "User-Agent": "FourHostsResearch/1.0 (+https://github.com/four-hosts/research-bot)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "DNT": "1",
    }
    timeout_sec = float(os.getenv("SEARCH_FETCH_TIMEOUT_SEC", "25"))
    if "doi.org" in url or "ssrn.com" in url:
        acad_min = float(os.getenv("SEARCH_ACADEMIC_MIN_TIMEOUT", "20"))
        timeout_sec = max(timeout_sec, acad_min)
    timeout = aiohttp.ClientTimeout(total=timeout_sec)

    async with session.get(url, headers=headers, timeout=timeout, allow_redirects=True) as r:
        if r.status == 429:
            retry_hdr = r.headers.get("retry-after", "")
            delay = _retry_after_to_seconds(retry_hdr) or random.uniform(5, 10)
            floor = float(os.getenv("SEARCH_429_MIN_DELAY", "3.0"))
            delay = max(delay, floor) * random.uniform(0.9, 1.2)
            _structured_log("warning", "rate_limited_fetch", {"url": url, "delay": round(delay,2)})
            await asyncio.sleep(delay)
            raise RateLimitedError()
        if r.status != 200:
            return "", {}

        ctype = (r.headers.get("Content-Type") or "").lower()
        is_pdf = "application/pdf" in ctype or url.lower().endswith(".pdf")
        if is_pdf and fitz is not None:
            data = await r.read()
            text = ""
            try:
                with fitz.open(stream=data, filetype="pdf") as pdf:
                    max_pages = int(os.getenv("SEARCH_PDF_MAX_PAGES", "15"))
                    max_chars = int(os.getenv("SEARCH_PDF_MAX_CHARS", "200000"))
                    stop_at_refs = os.getenv("SEARCH_PDF_STOP_AT_REFERENCES", "1") in {"1", "true", "yes"}
                    parts: list[str] = []
                    for i, p in enumerate(pdf):
                        if i >= max_pages:
                            break
                        try:
                            d = p.get_text("dict")
                            for block in d.get("blocks", []):
                                for line in block.get("lines", []):
                                    for span in line.get("spans", []):
                                        txt = (span.get("text") or "").strip()
                                        if txt:
                                            parts.append(txt)
                        except Exception:
                            parts.append(p.get_text())
                        if sum(len(x) + 1 for x in parts) >= max_chars:
                            break
                    joined = "\n".join(parts)
                    if stop_at_refs:
                        import re as _re
                        m = _re.search(r"\n\s*(References|Bibliography|Acknowledg(e)?ments)\s*\n", joined, _re.I)
                        if m:
                            joined = joined[:m.start()].strip()
                    text = joined[:max_chars]
            except Exception:
                text = ""
            meta = {"http": {"content_type": ctype, "last_modified": r.headers.get("Last-Modified")}}
            return text, meta

        # HTML path
        html = await r.text()
        max_chars = int(os.getenv("SEARCH_HTML_MAX_CHARS", "250000"))
        mode = (os.getenv("SEARCH_HTML_MODE", "main") or "main").lower()
        # Content
        try:
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
                text = _html_to_structured_text_legacy(soup, max_chars)
            except Exception:
                text = _strip_tags(html)[:max_chars]

        # Metadata
        meta = _extract_metadata_from_html(html, headers=dict(r.headers)) if os.getenv("SEARCH_EXTRACT_METADATA", "1") in {"1", "true", "yes"} else {}
        return text, meta


# --------------------------------------------------------------------------- #
#                            DOMAIN HELPERS                                   #
# --------------------------------------------------------------------------- #

class URLNormalizer:
    @staticmethod
    def normalize_url(url: str) -> str:
        if not url:
            return ""
        url = url.strip()
        if url.startswith("10."):
            return f"https://doi.org/{url}"
        if "doi.org/" in url:
            doi = unquote(url.split("doi.org/", 1)[1])
            return f"https://doi.org/{doi}"
        p = urlparse(url if "://" in url else f"https://{url}")
        return urlunparse((p.scheme, p.netloc.lower(), p.path, p.params, p.query, p.fragment))

    @staticmethod
    def is_valid_url(url: str) -> bool:
        p = urlparse(url)
        return bool(p.scheme and p.netloc)


class CircuitBreaker:
    def __init__(self, threshold: int = 5, timeout_sec: int = 300, max_timeout_sec: int | None = None, backoff_factor: float = 2.0):
        self.threshold = threshold
        self.timeout = timeout_sec
        self.max_timeout = max_timeout_sec or int(os.getenv("CB_MAX_TIMEOUT_SEC", "1800") or 1800)
        self.backoff_factor = backoff_factor
        self.failures: Dict[str, int] = {}
        self.last_fail: Dict[str, float] = {}
        self.blocked: Set[str] = set()
        self.block_until: Dict[str, float] = {}

    def ok(self, domain: str) -> bool:
        if domain not in self.blocked:
            return True
        now = time.time()
        until = self.block_until.get(domain) or (self.last_fail.get(domain, 0) + self.timeout)
        if now >= until:
            # Auto-reset on expiry
            self.blocked.discard(domain)
            self.failures[domain] = 0
            self.block_until.pop(domain, None)
            return True
        return False

    def fail(self, domain: str):
        self.failures[domain] = self.failures.get(domain, 0) + 1
        self.last_fail[domain] = time.time()
        if self.failures[domain] >= self.threshold:
            self.blocked.add(domain)
            # Exponential backoff window increases with consecutive failures above threshold
            over = max(0, self.failures[domain] - self.threshold + 1)
            wait = min(self.timeout * (self.backoff_factor ** (over - 1)) if over > 0 else self.timeout, self.max_timeout)
            self.block_until[domain] = self.last_fail[domain] + wait

    def success(self, domain: str):
        self.failures[domain] = max(0, self.failures.get(domain, 0) - 1)
        # Gradually recover: when we succeed, allow requests and clear block state
        if domain in self.blocked:
            self.blocked.discard(domain)
            self.block_until.pop(domain, None)


class RespectfulFetcher:
    """robots.txt aware fetcher with 1 req/sec per domain + circuit breaker."""

    def __init__(self):
        self.robot_cache: Dict[str, RobotFileParser] = {}
        self.robot_checked: Dict[str, float] = {}
        self.last_fetch: Dict[str, float] = {}
        self.ua = "FourHostsResearch/1.0 (+https://github.com/four-hosts/research-bot)"
        self.circuit = CircuitBreaker()
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

    async def fetch(self, session: aiohttp.ClientSession, url: str, on_rate_limit: Optional[Callable[[str], Awaitable[None]]] = None) -> str | None:
        url = URLNormalizer.normalize_url(url)
        if not URLNormalizer.is_valid_url(url):
            return None
        domain = urlparse(url).netloc

        if domain in self.blocked_domains or any(domain.endswith("." + d) for d in self.blocked_domains):
            return None
        if not self.circuit.ok(domain):
            if on_rate_limit:
                try:
                    await on_rate_limit(domain)
                except Exception:
                    pass
            return None
        if not await self._can_fetch(url):
            return None

        elapsed = time.time() - self.last_fetch.get(domain, 0)
        if elapsed < 1.0:
            await asyncio.sleep(1 - elapsed)
        self.last_fetch[domain] = time.time()

        try:
            text = await fetch_and_parse_url(session, url)
            if text:
                self.circuit.success(domain)
            return text
        except Exception as e:
            if isinstance(e, RateLimitedError) and on_rate_limit:
                try:
                    await on_rate_limit(domain)
                except Exception:
                    pass
            self.circuit.fail(domain)
            return None

    async def fetch_with_meta(self, session: aiohttp.ClientSession, url: str, on_rate_limit: Optional[Callable[[str], Awaitable[None]]] = None) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        url = URLNormalizer.normalize_url(url)
        if not URLNormalizer.is_valid_url(url):
            return None, None
        domain = urlparse(url).netloc

        if domain in self.blocked_domains or any(domain.endswith("." + d) for d in self.blocked_domains):
            return None, None
        if not self.circuit.ok(domain):
            if on_rate_limit:
                try:
                    await on_rate_limit(domain)
                except Exception:
                    pass
            return None, None
        if not await self._can_fetch(url):
            return None, None

        elapsed = time.time() - self.last_fetch.get(domain, 0)
        if elapsed < 1.0:
            await asyncio.sleep(1 - elapsed)
        self.last_fetch[domain] = time.time()

        try:
            text, meta = await fetch_and_parse_url_with_meta(session, url)
            if text:
                self.circuit.success(domain)
            return text, meta
        except Exception as e:
            if isinstance(e, RateLimitedError) and on_rate_limit:
                try:
                    await on_rate_limit(domain)
                except Exception:
                    pass
            self.circuit.fail(domain)
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
                self.domain = urlparse(self.url).netloc.lower()
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
    min_relevance_score: float = 0.25

    # Brave-specific
    offset: int = 0
    result_filter: Optional[List[str]] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    units: Optional[str] = None
    goggles: Optional[str] = None
    extra_snippets: bool = False
    summary: bool = False


# --------------------------------------------------------------------------- #
#                         RATE LIMITER                                        #
# --------------------------------------------------------------------------- #

class RateLimiter:
    def __init__(self, calls_per_minute: int = 60):
        self.calls = deque()
        self.cpm = calls_per_minute

    async def wait(self):
        now = datetime.now(timezone.utc)
        while self.calls and (now - self.calls[0]) >= timedelta(minutes=1):
            self.calls.popleft()
        if len(self.calls) >= self.cpm:
            delay = 60 - (now - self.calls[0]).total_seconds()
            if delay > 0:
                await asyncio.sleep(delay)
        self.calls.append(datetime.now(timezone.utc))


# --------------------------------------------------------------------------- #
#                           Query Optimiser                                   #
# --------------------------------------------------------------------------- #
class QueryOptimizer:
    """
    Generates cleaned-up / expanded query strings that retain user intent
    while maximising recall (quoted-phrase protection, synonym expansion,
    domain-specific boosts, etc.).
    """

    def __init__(self):
        self.use_nltk = _use_nltk
        self.stop_words = STOP_WORDS

        # Terms that rarely influence retrieval quality
        self.noise_terms: Set[str] = {"information", "details", "find", "show", "tell"}

        # Common multi-word technical entities we want to protect (keep quoted)
        self.known_entities = [
            "context engineering",
            "web applications",
            "artificial intelligence",
            "machine learning",
            "deep learning",
            "neural network",
            "neural networks",
            "large language model",
            "large language models",
            "state of the art",
            "natural language processing",
            "computer vision",
            "reinforcement learning",
            "generative ai",
            "transformer models",
            "foundation models",
        ]

        # Paradigm-specific dictionaries (used by _add_domain_specific_terms)
        self.paradigm_terms = {
            "dolores": ["investigation", "expose", "systemic", "corruption"],
            "teddy": ["support", "community", "help", "care"],
            "bernard": ["research", "analysis", "data", "evidence"],
            "maeve": ["strategy", "business", "optimization", "market"],
        }

    # --------------------------------------------------------------------- #
    #                     Internal helper functions                         #
    # --------------------------------------------------------------------- #
    def _extract_entities(self, query: str) -> Tuple[List[str], str]:
        """
        1.  Keep quoted phrases intact.          ->  "climate change"
        2.  Detect known multi-token entities.   ->  machine learning
        3.  Grab obvious proper nouns.           ->  OpenAI, Maeve
        Returns (protected_entities, remaining_text)
        """
        protected: List[str] = []
        remainder = query

        # 1. quoted phrases -------------------------------------------------
        for phrase in re.findall(r'"([^"]+)"', query):
            protected.append(phrase)
            remainder = remainder.replace(f'"{phrase}"', " ")

        # 2. known entities --------------------------------------------------
        low = remainder.lower()
        for ent in self.known_entities:
            if ent in low:
                protected.append(ent)
                low = low.replace(ent, " ")
                remainder = re.sub(re.escape(ent), " ", remainder, flags=re.I)

        # 3. proper nouns heuristic -----------------------------------------
        proper = re.findall(r"\b[A-Z][a-z]{2,}\b", remainder)
        for p in proper:
            if p.lower() not in self.stop_words:
                protected.append(p)

        # Clean leftover text
        remainder = re.sub(r"\s+", " ", remainder).strip()
        return list(dict.fromkeys(protected)), remainder  # dedup while preserving order

    def _intelligent_stopword_removal(self, text: str) -> List[str]:
        tokens = tokenize(text)          # shared helper already removes stop-words
        return [t for t in tokens if t not in self.noise_terms]

    # --------------------------------------------------------------------- #
    #                       Public helper methods                           #
    # --------------------------------------------------------------------- #
    def get_key_terms(self, query: str) -> List[str]:
        ents, left = self._extract_entities(query)
        return [e.replace('"', "") for e in ents] + self._intelligent_stopword_removal(left)

    # --------------------------------------------------------------------- #
    #              Query-expansion / variation generation                   #
    # --------------------------------------------------------------------- #
    def generate_query_variations(
        self, query: str, paradigm: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Returns several variations keyed by a label:
            {
                "primary": "...",
                "semantic": "...",
                "question": "...",
                "synonym": "...",
                ...
            }
        We usually send only the first 2-3 variations to each provider.
        """
        ents, left = self._extract_entities(query)
        keywords = self._intelligent_stopword_removal(left)

        # PRIMARY (AND-joined) ---------------------------------------------
        if ents or keywords:
            primary = " AND ".join([f'"{e}"' for e in ents] + keywords)
        else:
            primary = query.strip()

        variations: Dict[str, str] = {"primary": primary}

        # SEMANTIC (OR between protected phrases) --------------------------
        if len(ents) > 1:
            quoted_ents = [f'"{e}"' for e in ents]
            variations["semantic"] = f"({' OR '.join(quoted_ents)}) AND {' '.join(keywords)}"
        else:
            variations["semantic"] = primary.replace(" AND ", " ")

        # QUESTION form ----------------------------------------------------
        wh = ["what is", "how does", "why is", "explain", "when did"]
        if not any(query.lower().startswith(w) for w in wh):
            # De-duplicate terms case-insensitively, drop connectors, cap length
            raw_terms = [e for e in ents] + keywords
            seen: set[str] = set()
            clean_terms: List[str] = []
            for t in raw_terms:
                tt = (t or "").strip()
                if not tt:
                    continue
                tl = tt.lower()
                if tl in {"and", "or", "the"}:
                    continue
                if tl not in seen:
                    seen.add(tl)
                    clean_terms.append(tt)
            # Limit to keep queries readable
            clean_terms = clean_terms[:6]
            if clean_terms:
                variations["question"] = (
                    f"what is the relationship between {' and '.join(clean_terms)}"
                )
            else:
                variations["question"] = f"what is {query}"  # fallback

        # SYNONYM expansion -------------------------------------------------
        syn_kw = self._expand_synonyms(keywords)
        if syn_kw != keywords:
            variations["synonym"] = " ".join([f'"{e}"' for e in ents] + syn_kw)

        # RELATED concepts --------------------------------------------------
        rel = self._get_related_concepts(keywords)
        if rel:
            variations["related"] = f"{primary} OR ({' OR '.join(rel)})"

        # DOMAIN-specific ---------------------------------------------------
        if paradigm:
            dom = self._add_domain_specific_terms(primary, paradigm)
            if dom != primary:
                variations["domain_specific"] = dom

        # BROAD (first 3 important terms) ----------------------------------
        broad_terms = ents + keywords
        if len(broad_terms) > 2:
            variations["broad"] = " ".join(broad_terms[:3])

        # EXACT phrase ------------------------------------------------------
        if len(query.split()) <= 6:
            variations["exact_phrase"] = f'"{query.strip()}"'

        return variations

    def optimize_query(self, query: str, paradigm: Optional[str] = None) -> str:
        """Return just the 'primary' variation for convenience."""
        return self.generate_query_variations(query, paradigm)["primary"]

    # --------------------------------------------------------------------- #
    #                       Private expansion helpers                       #
    # --------------------------------------------------------------------- #
    def _expand_synonyms(self, terms: List[str]) -> List[str]:
        if not (self.use_nltk and terms):
            return terms[:]  # unchanged

        from nltk.corpus import wordnet as wn

        expanded: List[str] = []
        for t in terms:
            expanded.append(t)
            try:
                syns = wn.synsets(t)
            except Exception:
                syns = []
            for syn in syns[:2]:                      # limit per term
                for lemma in syn.lemma_names()[:3]:
                    lemma = lemma.replace("_", " ")
                    if lemma.lower() != t.lower() and lemma not in expanded:
                        expanded.append(lemma)
            if len(expanded) >= len(terms) + 4:       # cap overall growth
                break
        return expanded

    def _get_related_concepts(self, terms: List[str]) -> List[str]:
        concept_map = {
            "ai": ["artificial intelligence", "machine learning", "deep learning"],
            "ml": ["machine learning", "algorithms", "models"],
            "ethic": ["responsible ai", "moral", "governance"],
            "security": ["cybersecurity", "privacy", "data protection"],
            "climate": ["global warming", "environmental impact"],
            "health": ["healthcare", "medical", "wellness"],
            "finance": ["economic", "investment", "financial"],
            "education": ["learning", "teaching", "academic"],
        }
        related: List[str] = []
        for t in terms:
            for key, vals in concept_map.items():
                if key in t.lower():
                    related.extend(vals)
        # Also plural/singular normalization triggers
        for t in terms:
            if t.endswith("s"):
                related.append(t[:-1])
        return list(dict.fromkeys(related))[:3]        # unique, max 3

    def _add_domain_specific_terms(self, query: str, paradigm: str) -> str:
        extra = self.paradigm_terms.get(paradigm.lower())
        if not extra:
            return query
        return f"{query} {' '.join(extra)}"


# --------------------------------------------------------------------------- #
#                       CONTENT RELEVANCE FILTER                              #
# --------------------------------------------------------------------------- #

class ContentRelevanceFilter:
    """Simple relevance scoring; uses shared utilities."""

    def __init__(self):
        self.qopt = QueryOptimizer()
        self.consensus_threshold = 0.7

    def _term_frequency(self, text: str, terms: List[str]) -> float:
        if not terms or not text:
            return 0.0
        return sum(1 for t in terms if t in text) / len(terms)

    def _title_relevance(self, title: str, terms: List[str]) -> float:
        if not terms or not title:
            return 0.0
        return sum(1 for t in terms if t in title) / len(terms)

    def _freshness(self, dt: Optional[datetime]) -> float:
        if not dt:
            return 0.5
        age = (datetime.now(timezone.utc) - dt).days
        if age <= 7:
            return 1.0
        if age <= 30:
            return 0.8
        if age <= 90:
            return 0.6
        if age <= 365:
            return 0.4
        return 0.2

    def score(self, res: SearchResult, query: str, key_terms: List[str], cfg: SearchConfig) -> float:
        text = (res.title or "").lower() + " " + (res.snippet or "").lower()
        score = 0.35 * self._term_frequency(text, key_terms)
        score += 0.25 * self._title_relevance((res.title or "").lower(), key_terms)
        score += 0.10 * self._freshness(res.published_date)
        score += 0.05  # metadata bonus plain
        return min(1.0, score)

    def filter(self, results: List[SearchResult], query: str, cfg: SearchConfig) -> List[SearchResult]:
        k = self.qopt.get_key_terms(query)
        for r in results:
            r.relevance_score = self.score(r, query, k, cfg)
        return [r for r in results if r.relevance_score >= cfg.min_relevance_score]


# --------------------------------------------------------------------------- #
#                       BASE SEARCH API                                       #
# --------------------------------------------------------------------------- #

class BaseSearchAPI:
    def __init__(self, api_key: str = "", rate: int = 60):
        self.api_key = api_key
        self.rate = RateLimiter(calls_per_minute=rate)
        self.session: Optional[aiohttp.ClientSession] = None
        self.qopt = QueryOptimizer()
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

    async def search(self, query: str, cfg: SearchConfig) -> List[SearchResult]:
        raise NotImplementedError

    async def search_with_variations(self, query: str, cfg: SearchConfig) -> List[SearchResult]:
        """Unified variation search; providers themselves handle rate limiting."""
        variations = self.qopt.generate_query_variations(query)
        # Prioritise variants for better recall/precision balance
        priority = [
            "primary",
            "domain_specific",
            "exact_phrase",
            "synonym",
            "related",
            "broad",
            "question",
        ]
        ordered = sorted(variations.items(), key=lambda kv: priority.index(kv[0]) if kv[0] in priority else 999)
        limit = int(os.getenv("SEARCH_QUERY_VARIATIONS_LIMIT", "5"))
        ordered = ordered[:limit]

        seen: Set[str] = set()
        out: List[SearchResult] = []

        # Execute variant searches with light concurrency while respecting provider rate limiter
        sem = asyncio.Semaphore(int(os.getenv("SEARCH_VARIANT_CONCURRENCY", "3")))

        async def _run_variant(vt: str, q: str) -> List[SearchResult]:
            async with sem:
                try:
                    return await self.search(q, cfg)
                except RateLimitedError:
                    # Bubble up to manager so it can apply quota cooldowns
                    raise
                except Exception as e:
                    logger.warning(f"{self.__class__.__name__}:{vt} failed: {e}")
                    return []

        tasks = [asyncio.create_task(_run_variant(vt, q)) for vt, q in ordered]
        results_by_variant = await asyncio.gather(*tasks, return_exceptions=False)
        for (vt, _), res in zip(ordered, results_by_variant):
            for r in res:
                if r.url and r.url not in seen:
                    seen.add(r.url)
                    r.raw_data["query_variant"] = vt
                    out.append(r)
        return out


# --------------------------------------------------------------------------- #
#                  INDIVIDUAL PROVIDER IMPLEMENTATIONS                        #
# --------------------------------------------------------------------------- #

class BraveSearchAPI(BaseSearchAPI):
    def __init__(self, api_key: str):
        super().__init__(api_key, rate=100)
        self.base = "https://api.search.brave.com/res/v1/web/search"

    @with_circuit_breaker("brave_search", failure_threshold=3, recovery_timeout=30)
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=10),
           retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)))
    async def search(self, query: str, cfg: SearchConfig) -> List[SearchResult]:
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
                return []
            data = await r.json()
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
                domain=urlparse(item.get("url", "")).netloc if item.get("url") else "",
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
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, RateLimitedError)),
    )
    async def search(self, query: str, cfg: SearchConfig) -> List[SearchResult]:
        await self.rate.wait()
        params = {
            "key": self.api_key,
            "cx": self.cx,
            "q": self.qopt.optimize_query(query),
            "num": min(cfg.max_results, 10),
            "safe": "active" if cfg.safe_search == "strict" else "off" if cfg.safe_search == "off" else "medium",
            "hl": cfg.language,
        }
        # Use a reasonable timeout for Google Search
        timeout = aiohttp.ClientTimeout(total=15)
        try:
            async with self._sess().get(self.BASE, params=params, timeout=timeout) as r:
                if r.status == 429:
                    # Honour rate limiting and let tenacity back off
                    raise RateLimitedError()
                if 500 <= r.status <= 599:
                    # Server errors – retryable
                    raise aiohttp.ClientError(f"Google CSE {r.status}")
                if r.status != 200:
                    # Non-OK without retry – return empty
                    return []
                data = await r.json()
                # Some Google CSE errors come back as 200 with an 'error' body
                if isinstance(data, dict) and data.get("error"):
                    err = data.get("error", {})
                    reasons = ",".join([e.get("reason", "?") for e in err.get("errors", []) if isinstance(e, dict)])
                    if "rateLimitExceeded" in reasons or "dailyLimitExceeded" in reasons:
                        raise RateLimitedError()
                    return []
        except asyncio.TimeoutError:
            logger.warning(f"Google Search timeout for query: {query[:50]}")
            return []
        items = data.get("items", []) or []
        results: List[SearchResult] = []
        for it in items:
            title = it.get("title", "") or ""
            url = it.get("link", "") or ""
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
            res = SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                source="google_cse",
                published_date=meta_dt,
                raw_data=it,
            )
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
    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(min=2, max=10),
           retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, RateLimitedError)))
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
            async with self._sess().get(self.BASE, params=params, headers=headers, timeout=timeout) as r:
                if r.status == 429:
                    self._rotate_key()
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
        self.fetcher = RespectfulFetcher()
        self.rfilter = ContentRelevanceFilter()
        self._fetch_sem = asyncio.Semaphore(int(os.getenv("SEARCH_FETCH_CONCURRENCY", "8")))
        self._fallback_session: Optional[aiohttp.ClientSession] = None
        # Quota exhaustion handling: provider -> unblock timestamp (epoch seconds)
        self.quota_blocked: Dict[str, float] = {}

    def add_api(self, name: str, api: BaseSearchAPI):
        self.apis[name] = api

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

    async def search_all(self, query: str, cfg: SearchConfig, progress_callback: Optional[Any] = None, research_id: Optional[str] = None) -> List[SearchResult]:
        # Provider-specific start events so UI shows "Searching Brave...", "Searching Google..."
        if progress_callback and research_id:
            try:
                total = max(1, len(self.apis))
                for idx, name in enumerate(self.apis.keys()):
                    await progress_callback.report_search_started(research_id, query, name, idx + 1, total)
            except Exception:
                pass

        # Launch one task per provider with an individual timeout budget
        # so a single slow provider cannot stall the overall search.
        tasks: Dict[str, asyncio.Task] = {}
        # Resolve per-provider timeout; prefer explicit env, then fall back to overall task timeout
        import os as _os
        try:
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
                    return await asyncio.wait_for(_api.search_with_variations(query, cfg), timeout=_per_provider_to)
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
                            await progress_callback.report_search_completed(research_id, query, count)
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
                                normalized = URLNormalizer.normalize_url(canonical_url)
                                if URLNormalizer.is_valid_url(normalized):
                                    res.url = normalized
                                    res.domain = urlparse(normalized).netloc.lower()
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
                    res.raw_data["source_category"] = _derive_source_category(
                        getattr(res, "domain", ""),
                        content_type,
                    )
                except Exception:
                    pass

        await asyncio.gather(*(_fetch(r) for r in all_res))
        for r in all_res:
            ensure_snippet_content(r)
            normalize_result_text_fields(r)

        # Deduplicate (Jaccard on 3-grams) with zero-division guard
        deduped: List[SearchResult] = []
        sigs: List[Set[str]] = []
        for r in all_res:
            sig = set(ngram_tokenize((r.title or "") + " " + (r.snippet or "")))
            is_dup = False
            for s in sigs:
                union = sig | s
                if not union:
                    continue
                if len(sig & s) / len(union) >= 0.7:
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(r)
                sigs.append(sig)

        return self.rfilter.filter(deduped, query, cfg)


# --------------------------------------------------------------------------- #
#                     FACTORY / QUICK DEMO                                    #
# --------------------------------------------------------------------------- #

def create_search_manager() -> SearchAPIManager:
    mgr = SearchAPIManager()

    brave_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if brave_key and os.getenv("SEARCH_DISABLE_BRAVE", "0") not in {"1", "true", "yes"}:
        mgr.add_api("brave", BraveSearchAPI(brave_key))

    g_key = os.getenv("GOOGLE_CSE_API_KEY")
    g_cx = os.getenv("GOOGLE_CSE_CX")
    if g_key and g_cx and os.getenv("SEARCH_DISABLE_GOOGLE", "0") not in {"1", "true", "yes"}:
        mgr.add_api("google_cse", GoogleCustomSearchAPI(g_key, g_cx))

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
        res = await mgr.search_all("artificial intelligence ethics", cfg)
        for r in res[:5]:
            print(r.title, "—", r.url)


if __name__ == "__main__":
    asyncio.run(_demo())
