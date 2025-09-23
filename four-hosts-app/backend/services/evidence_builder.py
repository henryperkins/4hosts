"""
Evidence Builder
----------------
Select high-salience, quoted evidence from top web results to ground
the model's synthesis. Lightweight and dependency‑minimal.

Exports helpers:
    - build_evidence_bundle(query, results, max_docs=100, quotes_per_doc=10,
      include_full_content=True)
      -> EvidenceBundle
    - build_evidence_quotes(query, results, max_docs=100, quotes_per_doc=10)
      -> List[EvidenceQuote]
    - convert_quote_dicts_to_typed(quotes_raw) -> List[EvidenceQuote]
    - quotes_to_plain_dicts(quotes_typed) -> List[Dict]

Each returned quote dict has the shape:
    {
        "id": "q001",
        "url": str,
        "title": str,
        "domain": str,
        "quote": str,        # <= ~240 chars, sanitized
        "start": int,        # approximate char start in source text if known
        "end": int,
        "published_date": Any | None,
        "credibility_score": float | None,
    }
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Tuple

from utils.url_utils import extract_domain, canonicalize_url
from utils.source_normalization import dedupe_by_url
from models.evidence import EvidenceQuote, EvidenceBundle, EvidenceDocument
from utils.otel import otel_span as _otel_span

from utils.injection_hygiene import sanitize_snippet, flag_suspicious_snippet
from core.config import (
    EVIDENCE_MAX_DOCS_DEFAULT,
    EVIDENCE_QUOTES_PER_DOC_DEFAULT,
    EVIDENCE_QUOTE_MAX_CHARS,
    EVIDENCE_SEMANTIC_SCORING,
    EVIDENCE_BUDGET_TOKENS_DEFAULT,
)

# Reuse existing fetcher that handles HTML and PDFs
from services.search_apis import fetch_and_parse_url  # type: ignore
from services.rate_limiter import ClientRateLimiter  # type: ignore
from utils.token_budget import (
    estimate_tokens,
    trim_text_to_tokens,
    select_items_within_budget,
)


def _item_get(data: Any, key: str, default: Any = None) -> Any:
    """Helper to read dict-like or attribute-based search result fields."""
    if isinstance(data, dict):
        return data.get(key, default)
    return getattr(data, key, default)


_SENTENCE_SPLIT = re.compile(
    r"(?<=[\.!?])\s+(?=[A-Z0-9])|"
    r"\u2022|\u2023|\u25E6|\u2043|\u2219|\n|\r",
    re.VERBOSE,
)
_TOKEN = re.compile(r"[A-Za-z0-9]+")

try:
    # Optional semantic ranking using TF‑IDF cosine similarity
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

    _SK_OK = True
except Exception:
    _SK_OK = False

# In-process short-TTL cache for fetched evidence texts: canonical_url -> (text, expires_epoch)
_EVIDENCE_TEXT_CACHE: Dict[str, Tuple[str, float]] = {}


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN.findall(text or "") if len(t) > 2]


class EvidenceCircuitBreaker:
    """Simple circuit breaker for evidence fetching to prevent cascading failures."""
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open

    def call_success(self):
        """Record a successful call."""
        self.failure_count = 0
        self.state = "closed"

    def call_failure(self):
        """Record a failed call."""
        import time
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

    def allow_request(self) -> bool:
        """Check if request is allowed."""
        import time
        if self.state == "closed":
            return True
        elif self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                return True
            return False
        else:  # half-open
            return True


def _score_sentence(qtoks: set[str], sent: str) -> float:
    stoks = set(_tokens(sent))
    if not stoks:
        return 0.0
    overlap = len(qtoks & stoks)
    # Prefer sentences with numeric facts and moderate length
    has_num = 1 if re.search(r"\d", sent) else 0
    length_bonus = min(len(sent) / 200.0, 0.5)
    return overlap + 0.3 * has_num + length_bonus


def _semantic_scores(query: str, sentences: List[str]) -> List[float]:
    if not _SK_OK or not sentences:
        return [0.0] * len(sentences)
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        X = vec.fit_transform([query] + sentences)
        sims = cosine_similarity(X[0:1], X[1:]).flatten()
        # Normalize approximately to 0..1
        sims = [float(max(0.0, min(1.0, s))) for s in sims]
        return sims
    except Exception:
        return [0.0] * len(sentences)


def _domain_from(url: str) -> str:
    return extract_domain(url)


async def _fetch_texts(urls: List[str]) -> Dict[str, str]:
    import aiohttp  # local import to avoid hard dependency at import time
import structlog
    import os
    import time as _time

from logging_config import configure_logging

configure_logging()
logger = structlog.get_logger(__name__)

    out: Dict[str, str] = {}
    fetch_failures: List[str] = []

    # Use individual timeouts per URL for better success rate
    per_url_timeout = float(os.getenv("EVIDENCE_PER_URL_TIMEOUT", "15"))

    # Bounded concurrency to avoid overwhelming servers and improve stability
    max_concurrent = int(os.getenv("EVIDENCE_FETCH_CONCURRENCY", "5"))
    semaphore = asyncio.Semaphore(max_concurrent)

    # Initialize rate limiter and circuit breaker if enabled
    enable_rate_limiting = os.getenv("EVIDENCE_ENABLE_RATE_LIMITING", "1") == "1"
    rate_limiter = ClientRateLimiter(calls_per_minute=60) if enable_rate_limiting else None  # 60 calls per minute

    circuit_threshold = int(os.getenv("EVIDENCE_CIRCUIT_BREAKER_THRESHOLD", "5"))
    circuit_timeout = int(os.getenv("EVIDENCE_CIRCUIT_BREAKER_TIMEOUT", "60"))
    circuit_breaker = EvidenceCircuitBreaker(circuit_threshold, circuit_timeout)

    # Evidence fetch cache controls
    cache_enabled = os.getenv("EVIDENCE_FETCH_CACHE_ENABLE", "1").lower() in {"1", "true", "yes", "on"}
    use_redis_cache = os.getenv("EVIDENCE_FETCH_CACHE_USE_REDIS", "0").lower() in {"1", "true", "yes", "on"}
    ttl_sec = float(os.getenv("EVIDENCE_FETCH_CACHE_TTL_SEC", "600"))
    neg_ttl_sec = float(os.getenv("EVIDENCE_FETCH_CACHE_NEG_TTL_SEC", "120"))

    # Prepare canonical URL map and resolve cache hits upfront
    now = _time.time()
    canon_map: Dict[str, str] = {}
    to_fetch: List[str] = []
    cache_hits = 0
    cache_misses = 0

    # Optional Redis accessor
    async def _redis_get(canon: str) -> str | None:
        if not use_redis_cache:
            return None
        try:
            from services.cache import cache_manager  # lazy import
            key = f"evidence_text:{canon}"
            val = await cache_manager.get_kv(key)
            if isinstance(val, str):
                return val
            # Some cache backends may return bytes
            if isinstance(val, bytes):
                try:
                    return val.decode("utf-8", errors="replace")
                except Exception:
                    return None
            return None
        except Exception:
            return None

    async def _redis_set(canon: str, text: str, ttl: int) -> None:
        if not use_redis_cache:
            return
        try:
            from services.cache import cache_manager  # lazy import
            key = f"evidence_text:{canon}"
            await cache_manager.set_kv(key, text or "", ttl=max(1, int(ttl)))
        except Exception:
            # Best-effort only
            pass

    # Resolve in-memory/redis cache before issuing network requests
    if cache_enabled and urls:
        for url in urls:
            u = (url or "").strip()
            if not u:
                continue
            canon = canonicalize_url(u) or u
            canon_map[u] = canon

            # In-process cache first
            cached = _EVIDENCE_TEXT_CACHE.get(canon)
            if cached and cached[1] > now:
                out[u] = cached[0]
                cache_hits += 1
                continue

            # Optional Redis cache
            red = await _redis_get(canon)
            if isinstance(red, str) and red is not None and red != "":
                out[u] = red
                _EVIDENCE_TEXT_CACHE[canon] = (red, now + ttl_sec)
                cache_hits += 1
                continue

            # Miss -> schedule fetch
            to_fetch.append(u)
            cache_misses += 1
    else:
        to_fetch = list(urls or [])

    # Don't apply a session-wide timeout - let each URL have its own timeout
    headers = {"User-Agent": "FourHostsResearch/1.0 (+evidence-builder)"}
    async with aiohttp.ClientSession(headers=headers) as session:
        with _otel_span("evidence.fetch.batch", {"urls": len(urls)}) as _sp:
            _t0 = _time.time()

            # Create individual fetch tasks with their own timeouts
            async def fetch_single(url: str) -> None:
                async with semaphore:  # Limit concurrent fetches
                    # Check circuit breaker
                    if not circuit_breaker.allow_request():
                        out[url] = ""
                        fetch_failures.append(f"{url}: circuit breaker open")
                        return

                    # Apply rate limiting if enabled
                    if rate_limiter:
                        await rate_limiter.acquire(url)

                    try:
                        # Apply timeout using asyncio.wait_for for per-URL control
                        txt = await asyncio.wait_for(
                            fetch_and_parse_url(session, url),
                            timeout=per_url_timeout,
                        )
                        text_val = txt or ""
                        out[url] = text_val
                        if not txt:
                            fetch_failures.append(f"{url}: empty content")
                            circuit_breaker.call_failure()
                        else:
                            circuit_breaker.call_success()

                        # Populate caches
                        if cache_enabled:
                            canon = canon_map.get(url) or canonicalize_url(url) or url
                            expire = now + (ttl_sec if text_val else neg_ttl_sec)
                            _EVIDENCE_TEXT_CACHE[canon] = (text_val, expire)
                            # Fire-and-forget Redis set (await to preserve ordering; still cheap)
                            await _redis_set(canon, text_val, int(ttl_sec if text_val else neg_ttl_sec))

                    except asyncio.TimeoutError:
                        out[url] = ""
                        fetch_failures.append(
                            f"{url}: timeout after {per_url_timeout}s"
                        )
                        circuit_breaker.call_failure()
                    except aiohttp.ClientError as e:
                        out[url] = ""
                        fetch_failures.append(
                            f"{url}: network error - {type(e).__name__}"
                        )
                        circuit_breaker.call_failure()
                    except Exception as e:  # noqa: F841
                        out[url] = ""
                        fetch_failures.append(f"{url}: {type(e).__name__}")
                        circuit_breaker.call_failure()

            # Run fetches for cache misses only
            tasks = [asyncio.create_task(fetch_single(u)) for u in to_fetch]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # Set batch-level metrics
            try:
                if _sp:
                    _sp.set_attribute("latency_ms", int((_time.time() - _t0) * 1000))
                    _sp.set_attribute("success", True)
                    _sp.set_attribute("failures.count", len(fetch_failures))
                    _sp.set_attribute("cache.hits", int(cache_hits))
                    _sp.set_attribute("cache.misses", int(cache_misses))
            except Exception:
                pass

    # Expose cache hit/miss counters to telemetry (Prometheus if available)
    try:
        from services.telemetry_pipeline import telemetry_pipeline as _tp  # type: ignore
        _prom = getattr(_tp, "_prometheus", None)
        if _prom is not None:
            if cache_hits:
                _prom.evidence_cache_hits_total.inc(int(cache_hits))
            if cache_misses:
                _prom.evidence_cache_misses_total.inc(int(cache_misses))
    except Exception:
        pass

    # Also record to in-process metrics facade for internal dashboards
    try:
        from services.metrics import metrics as _metrics  # type: ignore
        if cache_hits:
            _metrics.increment("evidence_cache_hits", amount=int(cache_hits))
        if cache_misses:
            _metrics.increment("evidence_cache_misses", amount=int(cache_misses))
    except Exception:
        pass

    # Log fetch failures summary
    if fetch_failures:
        logger.warning(
            "Evidence fetch failures for %d/%d URLs. First 3 failures: %s",
            len(fetch_failures),
            len(urls),
            fetch_failures[:3],
        )
    elif urls:
        _success_count = sum(1 for v in out.values() if v)
        logger.debug(
            "Successfully fetched content from %d/%d URLs",
            _success_count,
            len(urls),
        )

    return out

    # NOTE: unreachable code below (return above) – ensure we log **before**
    # returning.  (Kept as comment for clarity.)


def _pick_docs(
    results: List[Dict[str, Any]],
    max_docs: int,
) -> List[Dict[str, Any]]:
    # Prefer high credibility and diversify by domain
    sorted_results = sorted(
        results or [],
        key=lambda r: float(_item_get(r, "credibility_score", 0.0) or 0.0),
        reverse=True,
    )
    picked: List[Dict[str, Any]] = []
    seen_domains: set[str] = set()
    domain_counts: dict[str, int] = {}

    for r in sorted_results:
        u = (_item_get(r, "url", "") or "").strip()
        if not u:
            continue
        metadata = _item_get(r, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        dom = metadata.get("domain") or _domain_from(u)

        # More forgiving domain diversity rules
        domain_count = domain_counts.get(dom, 0)
        # Allow up to 3 per domain early; unlimited later
        if domain_count >= 3 and len(picked) < max_docs // 2:
            continue  # Skip this result in early rounds

        picked.append(r)
        seen_domains.add(dom)
        domain_counts[dom] = domain_count + 1

        if len(picked) >= max_docs:
            break

    return picked


def _best_quotes_for_text(
    query: str,
    text: str,
    max_quotes: int = 3,
    max_len: int = EVIDENCE_QUOTE_MAX_CHARS,
    use_semantic: bool = EVIDENCE_SEMANTIC_SCORING,
) -> List[Tuple[str, int, int]]:
    import logging

    # Use module-level structured logger

    if not text:
        return []
    if not query:
        logger.debug("No query provided for quote extraction")
        return []

    try:
        qtoks = set(_tokens(query))
    except Exception as e:
        logger.debug(f"Failed to tokenize query: {e}")
        qtoks = set()

    # Break into sentences / list items
    try:
        parts = [
            p.strip()
            for p in _SENTENCE_SPLIT.split(text)
            if p and len(p.strip()) > 20
        ]
    except Exception as e:
        logger.debug(f"Failed to split text into sentences: {e}")
    # Fall back to empty
        parts = []
    scored: List[Tuple[str, int, int, float]] = []
    if use_semantic:
        sem = _semantic_scores(query, parts)
    else:
        sem = [0.0] * len(parts)
    for idx, p in enumerate(parts):
        # Trim long parts to a focused window around query terms when possible
        p_clean = sanitize_snippet(p, max_len=max_len * 2)
        kw = _score_sentence(qtoks, p_clean)
        sem_sc = sem[idx] if idx < len(sem) else 0.0
        # Blend: semantic (60%), keyword overlap & numerics (40%)
        score = 0.6 * sem_sc + 0.4 * kw
        if score <= 0:
            continue
        # Find approximate start offset (optional)
        try:
            start = text.find(p_clean[:60])
        except Exception:
            start = -1
        end = start + len(p_clean) if start >= 0 else -1
        if len(p_clean) > max_len:
            p_clean = p_clean[: max_len - 1] + "…"
        scored.append((p_clean, start, end, score))
    scored.sort(key=lambda x: x[3], reverse=True)
    top = [(q, s, e) for (q, s, e, _sc) in scored[:max_quotes]]
    return top


def _context_window_around(
    text: str,
    start: int,
    end: int,
    max_chars: int = 320,
) -> str:
    """Return a short context window (prev/next sentences) around a span.

    If sentence bounds are ambiguous, fall back to a fixed +/- window.
    Sanitized.
    """
    if (
        not text
        or start is None
        or end is None
        or start < 0
        or end < 0
        or start >= len(text)
    ):
        return ""
    try:
        # Search nearest sentence boundaries
        left = max(
            text.rfind(".", 0, start),
            text.rfind("!", 0, start),
            text.rfind("?", 0, start),
        )
    except Exception:
        left = -1
    try:
        right_candidates = [
            text.find(".", end),
            text.find("!", end),
            text.find("?", end),
        ]
        right_candidates = [c for c in right_candidates if c != -1]
        right = min(right_candidates) if right_candidates else -1
    except Exception:
        right = -1
    if left == -1:
        left = max(0, start - max_chars // 2)
    if right == -1:
        right = min(len(text), end + max_chars // 2)
    window = text[left:right].strip()
    if len(window) > max_chars:
        window = window[: max_chars - 1] + "…"
    return sanitize_snippet(window, max_len=max_chars)


def _summarize_text(
    query: str,
    text: str,
    max_sentences: int = 3,
    max_len: int = 500,
) -> str:
    """Lightweight extractive summary using TF‑IDF similarity to the query.

    Falls back to the first few sentences when semantic scoring is
    unavailable.
    """
    if not text:
        return ""
    parts = [
        p.strip()
        for p in _SENTENCE_SPLIT.split(text)
        if p and len(p.strip()) > 20
    ]
    if not parts:
        return sanitize_snippet(text[:max_len], max_len=max_len)
    sem = _semantic_scores(query, parts)
    idxs = list(range(len(parts)))

    def _sort_key(i: int) -> float:
        base = sem[i] if i < len(sem) else 0.0
        length_bonus = min(len(parts[i]) / 200.0, 0.5)
        return base + length_bonus

    # Rank by semantic score then length (prefer informative)
    idxs.sort(key=_sort_key, reverse=True)
    picked: List[str] = []
    for i in idxs[:max_sentences]:
        picked.append(
            sanitize_snippet(
                parts[i],
                max_len=max_len // max(1, max_sentences),
            )
        )
    out = " ".join(picked)
    return sanitize_snippet(out, max_len=max_len)


async def build_evidence_bundle(
    query: str,
    results: List[Dict[str, Any]],
    *,
    max_docs: int = EVIDENCE_MAX_DOCS_DEFAULT,
    quotes_per_doc: int = EVIDENCE_QUOTES_PER_DOC_DEFAULT,
    include_full_content: bool = True,
    full_text_budget: int = EVIDENCE_BUDGET_TOKENS_DEFAULT,
) -> EvidenceBundle:
    """Build comprehensive evidence bundle including quotes and documents."""
    import logging

    # Use module-level structured logger

    logger.debug(
        "Evidence builder invoked", 
        query=(query[:120] if query else ""),
        incoming_results=len(results),
        max_docs=max_docs,
        quotes_per_doc=quotes_per_doc,
        include_full_content=include_full_content,
        full_text_budget=full_text_budget,
    )

    if not results:
        logger.debug("Evidence builder: No input results provided")
        return EvidenceBundle()

    if not query:
        logger.warning(
            "Evidence builder: No query provided for evidence "
            "extraction"
        )

    docs = _pick_docs(results, max_docs=max_docs)

    logger.debug(
        "Evidence builder selected documents", 
        selected_docs=len(docs),
        candidate_results=len(results),
    )

    # FALLBACK: If no docs pass the picking criteria, use top N results
    if not docs and results:
        logger.warning(
            "Evidence builder: No documents met selection criteria from "
            "%d results. Using top %d as fallback.",
            len(results),
            min(max_docs, len(results)),
        )
        # Sort by credibility score if available, otherwise take first N
        has_cred = any(_item_get(r, "credibility_score") for r in results)
        if has_cred:
            sorted_results = sorted(
                results,
                key=lambda r: float(
                    _item_get(r, "credibility_score", 0.0) or 0.0
                ),
                reverse=True,
            )
        else:
            sorted_results = results

        # Take top N and mark them as fallback results
        docs = sorted_results[: min(max_docs, len(results))]
        for doc in docs:
            # Mark these as fallback docs for transparency
            metadata = _item_get(doc, "metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            metadata["evidence_fallback"] = True
            metadata["fallback_reason"] = "below_quality_threshold"
            if isinstance(doc, dict):
                doc["metadata"] = metadata

    if not docs:
        logger.warning(
            "Evidence builder: No valid documents selected from %d results",
            len(results),
        )
        return EvidenceBundle()

    logger.debug(
        "Evidence builder: Processing %d documents from %d search results",
        len(docs),
        len(results),
    )

    # Collect candidate URLs and drop unsupported schemes (e.g., exa://)
    urls_all = [
        (_item_get(d, "url", "") or "").strip()
        for d in docs
        if (_item_get(d, "url", "") or "").strip()
    ]
    urls = [u for u in urls_all if u.lower().startswith(("http://", "https://"))]

    # Log how many were skipped due to non-http schemes
    try:
        import logging as _logging
        # Module-level structured logger available as `logger`
        _log = logger  # reuse
        skipped = len(urls_all) - len(urls)
        if skipped > 0:
            _log.debug(
                "Evidence builder: skipped %d non-http(s) URLs (of %d)",
                skipped,
                len(urls_all),
            )
    except Exception:
        pass

    texts = await _fetch_texts(urls)

    quotes: List[EvidenceQuote] = []
    doc_entries: List[EvidenceDocument] = []
    total_doc_tokens = 0

    qid = 1
    doc_id = 1
    remaining_tokens = max(int(full_text_budget or 0), 0)

    for idx, raw_doc in enumerate(docs):
        url = (_item_get(raw_doc, "url", "") or "").strip()
        if not url:
            continue
        title = _item_get(raw_doc, "title", "") or ""
        metadata_raw = _item_get(raw_doc, "metadata", {}) or {}
        metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}
        domain = metadata.get("domain") or _item_get(raw_doc, "domain", "")
        if not domain:
            domain = _domain_from(url)
        snippet = _item_get(raw_doc, "snippet", "") or ""
        text = texts.get(url, "")
        doc_summary = (
            _summarize_text(query, text)
            if text
            else (sanitize_snippet(snippet, max_len=300) if snippet else "")
        )
        triples = _best_quotes_for_text(
            query,
            text,
            max_quotes=quotes_per_doc,
        )
        if not triples and snippet:
            triples = [(sanitize_snippet(snippet, 200), -1, -1)]

        credibility_val = _item_get(raw_doc, "credibility_score", None)
        try:
            credibility_score = (
                float(credibility_val) if credibility_val is not None else None
            )
        except Exception:
            credibility_score = None

        source_type = (
            metadata.get("result_type")
            or _item_get(raw_doc, "result_type", "web")
            or "web"
        )

        for quote, start, end in triples:
            ctx = (
                _context_window_around(
                    text,
                    start,
                    end,
                    max_chars=min(380, EVIDENCE_QUOTE_MAX_CHARS * 2),
                )
                if text and isinstance(start, int) and start >= 0
                else ""
            )
            quotes.append(
                EvidenceQuote(
                    id=f"q{qid:03d}",
                    url=url,
                    title=title,
                    domain=domain or "",
                    quote=quote,
                    start=int(start) if isinstance(start, int) else None,
                    end=int(end) if isinstance(end, int) else None,
                    published_date=metadata.get("published_date"),
                    credibility_score=credibility_score,
                    suspicious=bool(flag_suspicious_snippet(quote)),
                    doc_summary=doc_summary or None,
                    source_type=source_type,
                    context_window=ctx or None,
                )
            )
            qid += 1

        if include_full_content:
            base_content = text or _item_get(raw_doc, "content", "") or ""
            docs_left = max(1, len(docs) - idx)
            allocation = 0
            if remaining_tokens > 0 and base_content:
                allocation = min(
                    remaining_tokens,
                    max(
                        200,
                        (remaining_tokens // docs_left) or remaining_tokens,
                    ),
                )

            if allocation > 0 and base_content:
                content_for_doc = trim_text_to_tokens(base_content, allocation)
                truncated = len(content_for_doc) < len(base_content)
            else:
                fallback_candidates = [
                    doc_summary,
                    snippet,
                    base_content,
                    title,
                    url,
                ]
                content_for_doc = next(
                    (c for c in fallback_candidates if c),
                    "",
                )
                truncated = False

            token_count = estimate_tokens(content_for_doc) if content_for_doc else 0
            if remaining_tokens > 0 and token_count:
                remaining_tokens = max(0, remaining_tokens - token_count)
            total_doc_tokens += token_count

            doc_metadata = dict(metadata)
            if doc_summary and "summary" not in doc_metadata:
                doc_metadata["summary"] = doc_summary

            doc_entries.append(
                EvidenceDocument(
                    id=f"d{doc_id:03d}",
                    url=url,
                    title=title,
                    domain=domain or "",
                    content=content_for_doc or "",
                    token_count=token_count,
                    word_count=len((content_for_doc or "").split()),
                    truncated=truncated,
                    credibility_score=credibility_score,
                    published_date=doc_metadata.get("published_date"),
                    source_type=source_type,
                    metadata=doc_metadata,
                )
            )
            doc_id += 1

    # Balance quotes by domain: keep at most 4 per domain overall
    by_domain: Dict[str, int] = {}
    balanced: List[EvidenceQuote] = []
    for q in quotes:
        dom = (getattr(q, "domain", "") or "").lower()
        count = by_domain.get(dom, 0)
        if count >= 4:
            continue
        balanced.append(q)
        by_domain[dom] = count + 1

    balanced = balanced[: max_docs * quotes_per_doc]

    return EvidenceBundle(
        quotes=balanced,
        matches=[],
        by_domain=by_domain,
        focus_areas=[],
        documents=doc_entries,
        documents_token_count=total_doc_tokens,
    )


async def build_evidence_quotes(
    query: str,
    results: List[Dict[str, Any]],
    *,
    max_docs: int = EVIDENCE_MAX_DOCS_DEFAULT,
    quotes_per_doc: int = EVIDENCE_QUOTES_PER_DOC_DEFAULT,
) -> List[EvidenceQuote]:
    """Retained wrapper for legacy callers returning only evidence quotes."""
    bundle = await build_evidence_bundle(
        query,
        results,
        max_docs=max_docs,
        quotes_per_doc=quotes_per_doc,
        include_full_content=False,
    )
    return list(bundle.quotes)


def convert_quote_dicts_to_typed(quotes_raw: List[Dict[str, Any]]) -> List[EvidenceQuote]:
    out: List[EvidenceQuote] = []
    for q in quotes_raw or []:
        try:
            out.append(EvidenceQuote.model_validate(q))
        except Exception:
            try:
                out.append(
                    EvidenceQuote(
                        id=str(q.get("id", f"q{len(out)+1:03d}")),
                        url=q.get("url", ""),
                        title=q.get("title", ""),
                        domain=q.get("domain", ""),
                        quote=q.get("quote", ""),
                        start=q.get("start"),
                        end=q.get("end"),
                        published_date=q.get("published_date"),
                        credibility_score=q.get("credibility_score"),
                        suspicious=q.get("suspicious"),
                        doc_summary=q.get("doc_summary"),
                        source_type=q.get("source_type"),
                    )
                )
            except Exception:
                continue
    return out


def quotes_to_plain_dicts(quotes_typed: List[EvidenceQuote]) -> List[Dict[str, Any]]:
    return [q.model_dump() for q in quotes_typed or []]


async def build_evidence_pipeline(
    query: str,
    results: List[Dict[str, Any]],
    *,
    max_docs: int = EVIDENCE_MAX_DOCS_DEFAULT,
    quotes_per_doc: int = EVIDENCE_QUOTES_PER_DOC_DEFAULT,
    include_full_content: bool = True,
    full_text_budget: int = EVIDENCE_BUDGET_TOKENS_DEFAULT,
) -> tuple[EvidenceBundle, List[Dict[str, Any]]]:
    """
    Build end-to-end evidence bundle with centralized dedupe and budgeting.

    - Dedupe sources by canonical URL while preserving order
    - Select within token budget using utils.token_budget
    - Delegate quote extraction and (optional) full text to build_evidence_bundle
    - Return typed EvidenceBundle and plain quote dicts for payload embedding
    """
    # Fast path: nothing to do
    if not results:
        return EvidenceBundle(), []

    try:
        deduped = dedupe_by_url(results)
    except Exception:
        deduped = list(results)

    try:
        selected, _used, _dropped = select_items_within_budget(
            deduped,
            max_tokens=int(full_text_budget or 0),
        )
    except Exception:
        selected = list(deduped)

    bundle = await build_evidence_bundle(
        query,
        selected,
        max_docs=max_docs,
        quotes_per_doc=quotes_per_doc,
        include_full_content=include_full_content,
        full_text_budget=full_text_budget,
    )

    quotes_plain: List[Dict[str, Any]] = quotes_to_plain_dicts(
        list(getattr(bundle, "quotes", []) or [])
    )
    return bundle, quotes_plain
