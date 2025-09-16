"""
Evidence Builder
----------------
Select high-salience, quoted evidence from top web results to ground
the model's synthesis. Lightweight and dependency‑minimal.

Exports helpers:
    - build_evidence_bundle(query, results, max_docs=100, quotes_per_doc=10, include_full_content=True)
      -> EvidenceBundle
    - build_evidence_quotes(query, results, max_docs=100, quotes_per_doc=10) -> List[EvidenceQuote]
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
from models.evidence import EvidenceQuote, EvidenceBundle, EvidenceDocument

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
from utils.token_budget import estimate_tokens, trim_text_to_tokens


def _item_get(data: Any, key: str, default: Any = None) -> Any:
    """Helper to read dict-like or attribute-based search result fields."""
    if isinstance(data, dict):
        return data.get(key, default)
    return getattr(data, key, default)


_SENTENCE_SPLIT = re.compile(r"(?<=[\.!?])\s+(?=[A-Z0-9])|"
                             r"\u2022|\u2023|\u25E6|\u2043|\u2219|\n|\r",
                             re.VERBOSE)
_TOKEN = re.compile(r"[A-Za-z0-9]+")

try:
    # Optional semantic ranking using TF‑IDF cosine similarity
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    _SK_OK = True
except Exception:
    _SK_OK = False


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN.findall(text or "") if len(t) > 2]


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
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


async def _fetch_texts(urls: List[str]) -> Dict[str, str]:
    import aiohttp  # local import to avoid hard dependency at import time
    import logging
    import os

    logger = logging.getLogger(__name__)

    out: Dict[str, str] = {}
    fetch_failures = []

    # Use individual timeouts per URL for better success rate
    per_url_timeout = float(os.getenv("EVIDENCE_PER_URL_TIMEOUT", "10"))

    # Bounded concurrency to avoid overwhelming servers and improve stability
    max_concurrent = int(os.getenv("EVIDENCE_FETCH_CONCURRENCY", "10"))
    semaphore = asyncio.Semaphore(max_concurrent)

    # Don't apply a session-wide timeout - let each URL have its own timeout
    headers = {"User-Agent": "FourHostsResearch/1.0 (+evidence-builder)"}
    async with aiohttp.ClientSession(headers=headers) as session:
        # Create individual fetch tasks with their own timeouts
        async def fetch_single(url: str) -> None:
            async with semaphore:  # Limit concurrent fetches
                try:
                    # Apply timeout using asyncio.wait_for for per-URL control
                    txt = await asyncio.wait_for(
                        fetch_and_parse_url(session, url),
                        timeout=per_url_timeout
                    )
                    out[url] = txt or ""
                    if not txt:
                        fetch_failures.append(f"{url}: empty content")
                except asyncio.TimeoutError:
                    out[url] = ""
                    fetch_failures.append(f"{url}: timeout after {per_url_timeout}s")
                except aiohttp.ClientError as e:
                    out[url] = ""
                    fetch_failures.append(f"{url}: network error - {type(e).__name__}")
                except Exception as e:
                    out[url] = ""
                    fetch_failures.append(f"{url}: {type(e).__name__}")

        # Run all fetches in parallel with bounded concurrency via semaphore
        tasks = [asyncio.create_task(fetch_single(u)) for u in urls]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # Log fetch failures summary
    if fetch_failures:
        logger.warning(
            f"Evidence fetch failures for {len(fetch_failures)}/{len(urls)} URLs. "
            f"First 3 failures: {fetch_failures[:3]}"
        )
    elif urls:
        logger.debug(f"Successfully fetched content from {len([v for v in out.values() if v])}/{len(urls)} URLs")

    return out


def _pick_docs(results: List[Dict[str, Any]], max_docs: int) -> List[Dict[str, Any]]:
    # Prefer high credibility and diversify by domain
    sorted_results = sorted(
        results or [],
        key=lambda r: float(_item_get(r, "credibility_score", 0.0) or 0.0),
        reverse=True,
    )
    picked: List[Dict[str, Any]] = []
    seen_domains: set[str] = set()
    for r in sorted_results:
        u = (_item_get(r, "url", "") or "").strip()
        if not u:
            continue
        metadata = _item_get(r, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        dom = metadata.get("domain") or _domain_from(u)
        if dom in seen_domains and len(picked) < max_docs // 2:
            # Early rounds: prioritize new domains
            continue
        picked.append(r)
        seen_domains.add(dom)
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
    logger = logging.getLogger(__name__)

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
        parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p and len(p.strip()) > 20]
    except Exception as e:
        logger.debug(f"Failed to split text into sentences: {e}")
        parts = []
    scored = []
    sem = _semantic_scores(query, parts) if use_semantic else [0.0] * len(parts)
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

def _context_window_around(text: str, start: int, end: int, max_chars: int = 320) -> str:
    """Return a short context window (prev/next sentences) around a span.

    If sentence bounds are ambiguous, fall back to a fixed +/- window. Sanitized.
    """
    if not text or start is None or end is None or start < 0 or end < 0 or start >= len(text):
        return ""
    try:
        # Search nearest sentence boundaries
        left = max(text.rfind('.', 0, start), text.rfind('!', 0, start), text.rfind('?', 0, start))
    except Exception:
        left = -1
    try:
        right_candidates = [text.find('.', end), text.find('!', end), text.find('?', end)]
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
        window = window[: max_chars - 1] + '…'
    return sanitize_snippet(window, max_len=max_chars)


def _summarize_text(query: str, text: str, max_sentences: int = 3, max_len: int = 500) -> str:
    """Lightweight extractive summary using TF‑IDF similarity to the query.

    Falls back to the first few sentences when semantic scoring is unavailable.
    """
    if not text:
        return ""
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p and len(p.strip()) > 20]
    if not parts:
        return sanitize_snippet(text[:max_len], max_len=max_len)
    sem = _semantic_scores(query, parts)
    idxs = list(range(len(parts)))
    # Rank by semantic score then length (prefer informative)
    idxs.sort(key=lambda i: (sem[i] if i < len(sem) else 0.0) + min(len(parts[i]) / 200.0, 0.5), reverse=True)
    picked = []
    for i in idxs[:max_sentences]:
        picked.append(sanitize_snippet(parts[i], max_len=max_len // max(1, max_sentences)))
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
    """Build comprehensive evidence bundle including quotes and (optionally) full documents."""
    import logging
    logger = logging.getLogger(__name__)

    if not results:
        logger.debug("Evidence builder: No input results provided")
        return EvidenceBundle()

    if not query:
        logger.warning("Evidence builder: No query provided for evidence extraction")

    docs = _pick_docs(results, max_docs=max_docs)
    if not docs:
        logger.warning(f"Evidence builder: No valid documents selected from {len(results)} results")
        return EvidenceBundle()

    logger.debug(f"Evidence builder: Processing {len(docs)} documents from {len(results)} search results")

    urls = [(_item_get(d, "url", "") or "").strip() for d in docs if (_item_get(d, "url", "") or "").strip()]
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
        domain = metadata.get("domain") or (_item_get(raw_doc, "domain", "") or _domain_from(url))
        snippet = _item_get(raw_doc, "snippet", "") or ""
        text = texts.get(url, "")
        doc_summary = _summarize_text(query, text) if text else (sanitize_snippet(snippet, max_len=300) if snippet else "")
        triples = _best_quotes_for_text(query, text, max_quotes=quotes_per_doc)
        if not triples:
            if snippet:
                triples = [(sanitize_snippet(snippet, 200), -1, -1)]

        credibility_val = _item_get(raw_doc, "credibility_score", None)
        try:
            credibility_score = float(credibility_val) if credibility_val is not None else None
        except Exception:
            credibility_score = None

        source_type = metadata.get("result_type") or _item_get(raw_doc, "result_type", "web") or "web"

        for quote, start, end in triples:
            ctx = _context_window_around(
                text,
                start,
                end,
                max_chars=min(380, EVIDENCE_QUOTE_MAX_CHARS * 2),
            ) if text and isinstance(start, int) and start >= 0 else ""
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
                    max(200, (remaining_tokens // docs_left) or remaining_tokens),
                )

            if allocation > 0 and base_content:
                content_for_doc = trim_text_to_tokens(base_content, allocation)
                truncated = len(content_for_doc) < len(base_content)
            else:
                fallback_candidates = [doc_summary, snippet, base_content, title, url]
                content_for_doc = next((c for c in fallback_candidates if c), "")
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
                out.append(EvidenceQuote(
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
                ))
            except Exception:
                continue
    return out


def quotes_to_plain_dicts(quotes_typed: List[EvidenceQuote]) -> List[Dict[str, Any]]:
    return [q.model_dump() for q in quotes_typed or []]
