"""
Brave AI Grounding client
-------------------------
Lightweight async wrapper around Brave's Chat Completions-style endpoint to
retrieve grounded answers and extract citations for credibility checks.

Notes
- Requires `BRAVE_API_KEY` (aka subscription token).
- Defaults to non-streaming requests.
- Tries Chat Completions first; falls back to Summarizer Search for citations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlencode

import aiohttp

from .cache import cache_manager

logger = logging.getLogger(__name__)


@dataclass
class BraveCitation:
    title: Optional[str]
    url: str
    hostname: str


class BraveGroundingClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.search.brave.com/res/v1",
        country: str = "us",
        language: str = "en",
        timeout_seconds: int = 20,
    ) -> None:
        self.api_key = api_key or os.getenv("BRAVE_API_KEY") or os.getenv("BRAVE_SEARCH_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.country = country
        self.language = language
        self.timeout_seconds = timeout_seconds

        # Session is created on demand (so module import doesn’t fail without network)
        self._session: Optional[aiohttp.ClientSession] = None

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def aclose(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ────────────────────────────────────────────────────────────
    #  Public API
    # ────────────────────────────────────────────────────────────
    async def fetch_citations(
        self,
        query: str,
        *,
        enable_research: bool = False,
        force_refresh: bool = False,
    ) -> List[BraveCitation]:
        """Return a list of citations for the query using Brave.

        Tries Chat Completions with `enable_citations`; if no structured
        citations are found, falls back to Summarizer Search.
        """
        if not self.is_configured():
            logger.debug("BraveGroundingClient not configured; skipping")
            return []

        # Check cache
        cache_key = f"brave:citations:{hash((query, enable_research))}"
        if not force_refresh:
            cached = await cache_manager.get_kv(cache_key)
            if cached and isinstance(cached, list):
                return [BraveCitation(**c) for c in cached if isinstance(c, dict) and "url" in c]

        citations = await self._citations_via_chat(query, enable_research=enable_research)
        if not citations:
            citations = await self._citations_via_summarizer(query)

        # Cache
        if citations:
            await cache_manager.set_kv(
                cache_key,
                [c.__dict__ for c in citations],
                ttl=24 * 3600,
            )
        return citations

    # ────────────────────────────────────────────────────────────
    #  Internal helpers
    # ────────────────────────────────────────────────────────────
    async def _citations_via_chat(self, query: str, *, enable_research: bool) -> List[BraveCitation]:
        """Attempt to fetch citations using Chat Completions endpoint.

        The response shape is OpenAI-like; Brave may attach citations either at
        top-level or within enrichment/context payloads. We inspect common
        placements and normalize.
        """
        try:
            session = await self._ensure_session()
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self.api_key,
            }
            body: Dict[str, Any] = {
                "model": "brave",
                "stream": False,
                "messages": [{"role": "user", "content": query}],
                "country": self.country,
                "language": self.language,
                "enable_entities": True,
                "enable_citations": True,
            }
            if enable_research:
                body["enable_research"] = True

            async with session.post(url, headers=headers, data=json.dumps(body)) as resp:
                if resp.status != 200:
                    logger.debug("Brave chat status=%s text=%s", resp.status, await resp.text())
                    return []
                data = await resp.json()
        except Exception as e:
            logger.debug(f"Brave chat error: {e}")
            return []

        return self._extract_citations_from_any(data)

    async def _citations_via_summarizer(self, query: str) -> List[BraveCitation]:
        """Fallback to Summarizer Search API to harvest citations."""
        try:
            session = await self._ensure_session()
            url = f"{self.base_url}/summarizer/search"
            params = {
                "q": query,
                "country": self.country,
                "search_lang": self.language,
                "summary": True,
                "include_ans": True,
                "summary_type": "paragraph",
                "enable_citations": True,  # Some deployments use this flag
            }
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self.api_key,
            }
            async with session.get(f"{url}?{urlencode(params)}", headers=headers) as resp:
                if resp.status != 200:
                    logger.debug("Brave summarizer status=%s text=%s", resp.status, await resp.text())
                    return []
                data = await resp.json()
        except Exception as e:
            logger.debug(f"Brave summarizer error: {e}")
            return []

        return self._extract_citations_from_any(data)

    def _extract_citations_from_any(self, data: Dict[str, Any]) -> List[BraveCitation]:
        """Best-effort extraction of citations from varied Brave payloads."""
        citations: List[BraveCitation] = []

        # 1) Explicit top-level citations
        top = data.get("citations") if isinstance(data, dict) else None
        if isinstance(top, list):
            for item in top:
                url = (item or {}).get("url")
                if url:
                    citations.append(self._make_citation(url, item.get("title")))

        # 2) Enrichments/context path (summarizer + some chat responses)
        enrich = (data.get("enrichments") or {}) if isinstance(data, dict) else {}
        ctx = enrich.get("context") if isinstance(enrich, dict) else None
        if isinstance(ctx, list):
            for c in ctx:
                url = (c or {}).get("url")
                if url:
                    citations.append(self._make_citation(url, c.get("title")))

        # 3) Choices metadata (chat-completions variant)
        choices = data.get("choices") if isinstance(data, dict) else None
        if isinstance(choices, list):
            for ch in choices:
                msg = (ch or {}).get("message") or {}
                for key in ("citations", "context", "references"):
                    block = msg.get(key)
                    if isinstance(block, list):
                        for item in block:
                            url = (item or {}).get("url")
                            if url:
                                citations.append(self._make_citation(url, item.get("title")))

        # 4) De‑duplicate by URL
        seen: set[str] = set()
        deduped: List[BraveCitation] = []
        for c in citations:
            if c.url not in seen:
                seen.add(c.url)
                deduped.append(c)
        return deduped

    @staticmethod
    def _make_citation(url: str, title: Optional[str]) -> BraveCitation:
        try:
            host = urlparse(url).netloc
        except Exception:
            host = ""
        return BraveCitation(title=title, url=url, hostname=host)


# Convenience singleton accessor (lazy)
_singleton: Optional[BraveGroundingClient] = None


def brave_client() -> BraveGroundingClient:
    global _singleton
    if _singleton is None:
        _singleton = BraveGroundingClient()
    return _singleton

