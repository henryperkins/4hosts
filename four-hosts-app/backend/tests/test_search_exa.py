import asyncio
import os
import sys
import types
from datetime import datetime, timezone

import pytest

# Ensure backend root is on sys.path for 'services' imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.search_apis import (
    ExaSearchAPI,
    SearchConfig,
    SearchAPIManager,
    RateLimitedError,
)
from services.query_planning.types import QueryCandidate


class DummyResponse:
    def __init__(self, status=200, json_data=None):
        self.status = status
        self._json = json_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        if isinstance(self._json, Exception):
            raise self._json
        return self._json


class DummySession:
    def __init__(self, responses):
        self._responses = list(responses)
        self._idx = 0
        self.last_json = None
        self.last_headers = None

    def post(self, url, headers=None, json=None, timeout=None):
        # Return the next response; clamp to last when exhausted
        if not self._responses:
            return DummyResponse(status=200, json_data={"results": []})
        resp = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        self.last_json = json
        self.last_headers = headers
        return resp


@pytest.mark.asyncio
async def test_exa_mapping_and_content_toggle(monkeypatch):
    api = ExaSearchAPI(api_key="test")

    payload = {
        "results": [
            {
                "title": "Sample Title",
                "url": "https://example.com/a",
                "publishedDate": "2024-05-15",
                "author": "Author Name",
                "id": "doc-1",
                "score": 0.87,
                "image": "https://cdn/img.jpg",
                "favicon": "https://cdn/icon.ico",
                "highlights": [
                    "First highlight snippet.",
                    "Second highlight snippet."
                ],
                "highlightScores": [0.5, 0.4],
                "text": "Full page text from Exa."
            }
        ]
    }

    # 1) EXA_INCLUDE_TEXT=0 => no direct content, snippet used for ensure_snippet_content
    monkeypatch.setenv("EXA_INCLUDE_TEXT", "0")
    api._sess = lambda: DummySession([DummyResponse(status=200, json_data=payload)])  # type: ignore[assignment]

    res = await api.search("q", SearchConfig(max_results=5, min_relevance_score=0.0))
    assert res and res[0].source == "exa"
    r = res[0]
    assert r.title == "Sample Title"
    assert r.url == "https://example.com/a"
    assert r.published_date and r.published_date.year == 2024
    assert r.author == "Author Name"
    # snippet uses first highlight
    assert r.snippet.startswith("First highlight")
    # content should be populated only via snippet helper (not full text)
    assert r.content and r.content.startswith("Summary from search results:")
    # raw exa fields are preserved
    exa_raw = r.raw_data.get("exa", {})
    assert exa_raw.get("id") == "doc-1"
    assert exa_raw.get("score") == 0.87
    assert exa_raw.get("image") and exa_raw.get("favicon")
    assert exa_raw.get("highlights") and len(exa_raw["highlights"]) == 2

    # 2) EXA_INCLUDE_TEXT=1 => direct content set
    monkeypatch.setenv("EXA_INCLUDE_TEXT", "1")
    api2 = ExaSearchAPI(api_key="test")
    api2._sess = lambda: DummySession([DummyResponse(status=200, json_data=payload)])  # type: ignore[assignment]
    res2 = await api2.search("q", SearchConfig(max_results=5, min_relevance_score=0.0))
    assert res2 and res2[0].content == "Full page text from Exa."


@pytest.mark.asyncio
async def test_exa_no_highlights_snippet_behavior(monkeypatch):
    api = ExaSearchAPI(api_key="test")
    payload = {"results": [{"title": "T", "url": "https://e.com", "publishedDate": "2023-01-01"}]}
    api._sess = lambda: DummySession([DummyResponse(status=200, json_data=payload)])  # type: ignore[assignment]
    res = await api.search("q", SearchConfig(max_results=3, min_relevance_score=0.0))
    assert res and res[0].snippet == ""  # no highlights -> empty snippet
    # ensure_snippet_content only adds content when snippet exists; content stays None
    assert res[0].content is None


@pytest.mark.asyncio
async def test_exa_429_propagates_and_manager_sets_cooldown(monkeypatch):
    # Force consecutive 429s
    api = ExaSearchAPI(api_key="k")
    api._sess = lambda: DummySession([DummyResponse(status=429), DummyResponse(status=429), DummyResponse(status=429)])  # type: ignore[assignment]

    # Direct call should raise after retries
    with pytest.raises(RateLimitedError):
        await api.search("q", SearchConfig(max_results=2))

    # Manager applies cooldown on RateLimitedError
    mgr = SearchAPIManager()
    mgr.add_api("exa", api, is_fallback=True)
    monkeypatch.setenv("SEARCH_QUOTA_COOLDOWN_SEC", "5")

    planned = [QueryCandidate(query="q", stage="rule_based", label="seed")]
    with pytest.raises(RateLimitedError):
        await mgr._search_single_provider("exa", api, planned, SearchConfig(max_results=2), None, None)

    assert "exa" in mgr.quota_blocked and mgr.quota_blocked["exa"] > 0


@pytest.mark.asyncio
async def test_exa_handles_snake_case_payload(monkeypatch):
    api = ExaSearchAPI(api_key="test")
    payload = {
        "results": [
            {
                "title": "Snake",
                "url": "https://example.com/snake",
                "published_date": "2024-05-05",
                "highlights": ["shed skin"],
                "highlight_scores": [0.42],
            }
        ]
    }
    api._sess = lambda: DummySession([DummyResponse(status=200, json_data=payload)])  # type: ignore[assignment]
    res = await api.search("q", SearchConfig(max_results=1, min_relevance_score=0.0))
    assert res and res[0].published_date and res[0].published_date.year == 2024
    assert res[0].snippet == "shed skin"
    exa_raw = res[0].raw_data.get("exa") or {}
    assert exa_raw.get("highlight_scores") == [0.42]


@pytest.mark.asyncio
async def test_exa_request_payload_matches_spec(monkeypatch):
    api = ExaSearchAPI(api_key="key")
    dummy = DummySession([DummyResponse(status=200, json_data={"results": []})])
    api._sess = lambda: dummy  # type: ignore[assignment]
    monkeypatch.setenv("EXA_INCLUDE_HIGHLIGHTS", "0")
    cfg = SearchConfig(
        max_results=7,
        authority_whitelist=["exa.ai"],
        authority_blacklist=["spam.example"],
    )

    await api.search("spec", cfg)

    assert dummy.last_json["num_results"] == 7
    assert dummy.last_json["include_domains"] == ["exa.ai"]
    assert dummy.last_json["exclude_domains"] == ["spam.example"]
    assert "highlights" not in dummy.last_json


@pytest.mark.asyncio
async def test_factory_wiring_honors_env(monkeypatch):
    # Ensure Brave/Google do not interfere in this test
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_CX", raising=False)

    # Exa present as fallback by default
    monkeypatch.setenv("EXA_API_KEY", "x")
    monkeypatch.setenv("SEARCH_DISABLE_EXA", "0")
    monkeypatch.setenv("EXA_SEARCH_AS_PRIMARY", "0")
    from services.search_apis import create_search_manager

    mgr = create_search_manager()
    assert "exa" in mgr.apis
    assert mgr.primary_api != "exa"

    # Promote to primary
    monkeypatch.setenv("EXA_SEARCH_AS_PRIMARY", "1")
    mgr2 = create_search_manager()
    assert mgr2.primary_api == "exa"
