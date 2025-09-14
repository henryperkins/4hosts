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
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union, cast
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
        t = _html.unescape(str(text))
        t = _re.sub(r"<[^>]+>", " ", t)
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
        result.title = _strip_tags(result.title)[:300]
    except Exception:
        pass
    try:
        result.snippet = _strip_tags(result.snippet)[:800]
    except Exception:
        pass


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
async def fetch_and_parse_url(session: aiohttp.ClientSession, url: str) -> str:
    """GET `url`, honour 429 back-off, return plaintext (PDF or HTML)."""
    headers = {
        "User-Agent": "FourHostsResearch/1.0 (+https://github.com/four-hosts/research-bot)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "DNT": "1",
    }

    # Use shorter timeout for DOI/SSRN to avoid long delays
    timeout_sec = float(os.getenv("SEARCH_FETCH_TIMEOUT_SEC", "25"))
    if "doi.org" in url or "ssrn.com" in url:
        timeout_sec = min(timeout_sec, 10.0)
    timeout = aiohttp.ClientTimeout(total=timeout_sec)

    async with session.get(url, headers=headers, timeout=timeout, allow_redirects=True) as r:
        if r.status == 429:
            retry_hdr = r.headers.get("retry-after")
            # Respect the retry-after header if present
            if retry_hdr and retry_hdr.isdigit():
                delay = min(float(retry_hdr), 30)  # Cap at 30 seconds
            else:
                delay = random.uniform(5, 10)  # Default longer backoff
            _structured_log("warning", "rate_limited_fetch", {"url": url, "delay": delay})
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
                    pages = []
                    for i, p in enumerate(pdf):
                        if i >= max_pages:
                            break
                        pages.append(p.get_text())
                    text = "".join(pages)
                    return text[:max_chars]
            except Exception:
                # Fall back to HTML parse if PDF parse fails
                pass

        # HTML/text fallback parse
        html = await r.text()
        soup = BeautifulSoup(html, "html.parser")
        for s in soup(["script", "style", "noscript"]):
            s.decompose()
        return soup.get_text(" ", strip=True)


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
    def __init__(self, threshold: int = 5, timeout_sec: int = 300):
        self.threshold, self.timeout = threshold, timeout_sec
        self.failures: Dict[str, int] = {}
        self.last_fail: Dict[str, float] = {}
        self.blocked: Set[str] = set()

    def ok(self, domain: str) -> bool:
        if domain not in self.blocked:
            return True
        if time.time() - self.last_fail.get(domain, 0) > self.timeout:
            self.blocked.discard(domain)
            self.failures[domain] = 0
            return True
        return False

    def fail(self, domain: str):
        self.failures[domain] = self.failures.get(domain, 0) + 1
        if self.failures[domain] >= self.threshold:
            self.blocked.add(domain)
        self.last_fail[domain] = time.time()

    def success(self, domain: str):
        self.failures[domain] = max(0, self.failures.get(domain, 0) - 1)


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

    async def fetch(self, session: aiohttp.ClientSession, url: str) -> str | None:
        url = URLNormalizer.normalize_url(url)
        if not URLNormalizer.is_valid_url(url):
            return None
        domain = urlparse(url).netloc

        if domain in self.blocked_domains or any(domain.endswith("." + d) for d in self.blocked_domains):
            return None
        if not self.circuit.ok(domain):
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
        except Exception:
            self.circuit.fail(domain)
            return None


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
        if self.title and self.snippet:
            sig = f"{self.title.lower().strip()}{self.snippet.lower().strip()}"
            try:
                self.content_hash = hashlib.md5(sig.encode()).hexdigest()
            except Exception:
                self.content_hash = None


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
        seen: Set[str] = set()
        out: List[SearchResult] = []
        for var_type, q in list(variations.items())[:3]:
            try:
                res = await self.search(q, cfg)
            except Exception as e:
                logger.warning(f"{self.__class__.__name__}:{var_type} failed: {e}")
                continue
            for r in res:
                if r.url and r.url not in seen:
                    seen.add(r.url)
                    r.raw_data["query_variant"] = var_type
                    out.append(r)
        return out


# --------------------------------------------------------------------------- #
#                  INDIVIDUAL PROVIDER IMPLEMENTATIONS                        #
# --------------------------------------------------------------------------- #

class BraveSearchAPI(BaseSearchAPI):
    def __init__(self, api_key: str):
        super().__init__(api_key, rate=100)
        self.base = "https://api.search.brave.com/res/v1/web/search"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=10),
           retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)))
    async def search(self, query: str, cfg: SearchConfig) -> List[SearchResult]:
        await self.rate.wait()
        params = {
            "q": self.qopt.optimize_query(query)[:400],
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

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(min=2, max=10),
           retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)))
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
                if r.status != 200:
                    return []
                data = await r.json()
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
        async with self._sess().get(f"{self.BASE}/esearch.fcgi", params=params) as r:
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
        async with self._sess().get(f"{self.BASE}/efetch.fcgi", params=params) as r:
            return await r.text() if r.status == 200 else ""

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

    async def search_all(self, query: str, cfg: SearchConfig) -> List[SearchResult]:
        tasks = {name: asyncio.create_task(api.search_with_variations(query, cfg))
                 for name, api in self.apis.items()}
        all_res: List[SearchResult] = []

        if tasks:
            # Bound overall wait per provider set to avoid blocking on a single slow API
            import os as _os
            try:
                # Prefer explicit provider timeout; fall back to overall task timeout if provided
                _prov_to = float(_os.getenv("SEARCH_PROVIDER_TIMEOUT_SEC") or 0.0)
                if not _prov_to:
                    # Increase default timeout to 25s to allow APIs to complete
                    _prov_to = float(_os.getenv("SEARCH_TASK_TIMEOUT_SEC", "25") or 25)
            except Exception:
                _prov_to = 25.0

            name_by_task = {t: n for n, t in tasks.items()}
            done, pending = await asyncio.wait(set(tasks.values()), timeout=_prov_to)

            # Collect finished results
            for t in done:
                name = name_by_task.get(t, "unknown")
                try:
                    res = t.result()
                    all_res.extend(res if isinstance(res, list) else [])
                except Exception as e:
                    logger.error(f"{name} failed: {e}")

            # Cancel any stragglers
            for t in pending:
                name = name_by_task.get(t, "unknown")
                try:
                    t.cancel()
                except Exception:
                    pass
                logger.warning(f"{name} timed out after {_prov_to:.1f}s; skipping")
        else:
            all_res = []

        # Fetch full content concurrently
        async def _fetch(res: SearchResult):
            if res.content or not res.url:
                return
            session = self._any_session()
            async with self._fetch_sem:
                fetched = await self.fetcher.fetch(session, res.url)
            if fetched:
                res.content = fetched

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
    if brave_key:
        mgr.add_api("brave", BraveSearchAPI(brave_key))

    g_key = os.getenv("GOOGLE_CSE_API_KEY")
    g_cx = os.getenv("GOOGLE_CSE_CX")
    if g_key and g_cx:
        mgr.add_api("google_cse", GoogleCustomSearchAPI(g_key, g_cx))

    # Academic/open providers
    mgr.add_api("arxiv", ArxivAPI())
    mgr.add_api("crossref", CrossRefAPI())
    mgr.add_api("semanticscholar", SemanticScholarAPI())
    pm_key = os.getenv("PUBMED_API_KEY")  # optional
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
