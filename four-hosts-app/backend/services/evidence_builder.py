"""
Evidence Builder
----------------
Select high-salience, quoted evidence from top web results to ground
the model's synthesis. Lightweight and dependency‑minimal.

Exports a single async helper:
    build_evidence_quotes(query, results, max_docs=8, quotes_per_doc=3) -> List[Dict]

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

from utils.injection_hygiene import sanitize_snippet, flag_suspicious_snippet

# Reuse existing fetcher that handles HTML and PDFs
from services.search_apis import fetch_and_parse_url  # type: ignore


_SENTENCE_SPLIT = re.compile(r"(?<=[\.!?])\s+(?=[A-Z0-9])|
                               \u2022|\u2023|\u25E6|\u2043|\u2219|\n|\r",
                             re.VERBOSE)
_TOKEN = re.compile(r"[A-Za-z0-9]+")


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


def _domain_from(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


async def _fetch_texts(urls: List[str]) -> Dict[str, str]:
    import aiohttp  # local import to avoid hard dependency at import time

    out: Dict[str, str] = {}
    timeout = aiohttp.ClientTimeout(total=30)
    headers = {"User-Agent": "FourHostsResearch/1.0 (+evidence-builder)"}
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        tasks = []
        for u in urls:
            async def run(u=u):
                try:
                    txt = await fetch_and_parse_url(session, u)
                    out[u] = txt or ""
                except Exception:
                    out[u] = ""
            tasks.append(asyncio.create_task(run()))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    return out


def _pick_docs(results: List[Dict[str, Any]], max_docs: int) -> List[Dict[str, Any]]:
    # Prefer high credibility and diversify by domain
    sorted_results = sorted(
        results or [],
        key=lambda r: float(r.get("credibility_score", 0.0) or 0.0),
        reverse=True,
    )
    picked: List[Dict[str, Any]] = []
    seen_domains: set[str] = set()
    for r in sorted_results:
        u = (r.get("url") or "").strip()
        if not u:
            continue
        dom = (r.get("metadata", {}) or {}).get("domain") or _domain_from(u)
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
    max_len: int = 240,
) -> List[Tuple[str, int, int]]:
    if not text:
        return []
    qtoks = set(_tokens(query))
    # Break into sentences / list items
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p and len(p.strip()) > 20]
    scored = []
    for p in parts:
        # Trim long parts to a focused window around query terms when possible
        p_clean = sanitize_snippet(p, max_len=max_len * 2)
        score = _score_sentence(qtoks, p_clean)
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


async def build_evidence_quotes(
    query: str,
    results: List[Dict[str, Any]],
    *,
    max_docs: int = 8,
    quotes_per_doc: int = 3,
) -> List[Dict[str, Any]]:
    """Return prioritized quotes across high-credibility, diverse sources."""
    if not results:
        return []

    docs = _pick_docs(results, max_docs=max_docs)
    urls = [d.get("url", "") for d in docs if d.get("url")]
    texts = await _fetch_texts(urls)

    quotes: List[Dict[str, Any]] = []
    qid = 1
    for d in docs:
        url = d.get("url", "")
        title = d.get("title", "")
        md = d.get("metadata", {}) or {}
        domain = md.get("domain") or _domain_from(url)
        text = texts.get(url, "")
        triples = _best_quotes_for_text(query, text, max_quotes=quotes_per_doc)
        if not triples:
            # fallback to snippet
            snip = d.get("snippet", "")
            if snip:
                triples = [(sanitize_snippet(snip, 200), -1, -1)]
        for quote, start, end in triples:
            item = {
                "id": f"q{qid:03d}",
                "url": url,
                "title": title,
                "domain": domain,
                "quote": quote,
                "start": int(start) if isinstance(start, int) else -1,
                "end": int(end) if isinstance(end, int) else -1,
                "published_date": md.get("published_date"),
                "credibility_score": d.get("credibility_score"),
                "suspicious": bool(flag_suspicious_snippet(quote)),
            }
            quotes.append(item)
            qid += 1

    # Balance by domain: keep at most 4 per domain overall
    by_domain: Dict[str, int] = {}
    balanced: List[Dict[str, Any]] = []
    for q in quotes:
        dom = (q.get("domain") or "").lower()
        c = by_domain.get(dom, 0)
        if c >= 4:
            continue
        balanced.append(q)
        by_domain[dom] = c + 1

    # Cap hard limit to protect prompt
    return balanced[: max_docs * quotes_per_doc]

