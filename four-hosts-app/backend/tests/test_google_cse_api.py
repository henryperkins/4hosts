import os
import sys
from typing import Any, Dict

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.search_apis import GoogleCustomSearchAPI, SearchConfig  # noqa: E402


class DummyResponse:
    def __init__(self, status: int = 200, payload: Dict[str, Any] | None = None):
        self.status = status
        self._payload = payload or {"items": []}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self._payload


class DummySession:
    def __init__(self, response: DummyResponse):
        self._response = response
        self.last_params: Dict[str, Any] | None = None
        self.last_timeout = None

    def get(self, url, params=None, timeout=None):
        self.last_params = params
        self.last_timeout = timeout
        return self._response


@pytest.mark.asyncio
async def test_google_cse_parameter_mapping(monkeypatch):
    api = GoogleCustomSearchAPI("api-key", "cx-id")
    dummy = DummySession(DummyResponse())
    api._sess = lambda: dummy  # type: ignore[assignment]

    cfg = SearchConfig(
        max_results=25,
        language="fr-CA",
        region="CA",
        date_range="w2",
        exclusion_keywords=["spam", "foo"],
        offset=10,
        safe_search="strict",
    )

    await api.search("solar energy", cfg)

    params = dummy.last_params
    assert params is not None
    assert params["key"] == "api-key"
    assert params["cx"] == "cx-id"
    assert params["num"] == 10  # capped to API maximum
    assert params["start"] == 11
    assert params["safe"] == "active"
    assert params["hl"] == "fr-CA"
    assert params["lr"] == "lang_fr"
    assert params["gl"] == "ca"
    assert params["dateRestrict"] == "w2"
    # excludeTerms sorted and de-duplicated
    assert params["excludeTerms"] == "foo spam"


@pytest.mark.asyncio
async def test_google_cse_start_clamped(monkeypatch):
    api = GoogleCustomSearchAPI("k", "cx")
    dummy = DummySession(DummyResponse())
    api._sess = lambda: dummy  # type: ignore[assignment]

    cfg = SearchConfig(max_results=5, offset=500)
    await api.search("query", cfg)

    params = dummy.last_params
    assert params is not None
    assert params["num"] == 5
    assert params["start"] == 96  # 100 - num + 1


@pytest.mark.asyncio
async def test_google_cse_item_mapping(monkeypatch):
    api = GoogleCustomSearchAPI("k", "cx")
    payload = {
        "items": [
            {
                "title": "Example PDF",
                "formattedUrl": "https://example.com/report.pdf",
                "displayLink": "example.com",
                "snippet": "Summary",
                "fileFormat": "PDF/Adobe Acrobat",
                "mime": "application/pdf",
                "cacheId": "CACHE123",
                "pagemap": {"metatags": [{"datepublished": "2024-03-01"}]},
            }
        ]
    }
    dummy = DummySession(DummyResponse(payload=payload))
    api._sess = lambda: dummy  # type: ignore[assignment]

    results = await api.search("query", SearchConfig(max_results=1))
    assert results and results[0].result_type == "file"
    assert results[0].domain == "example.com"
    raw = results[0].raw_data
    assert raw.get("cacheId") == "CACHE123"
    assert raw.get("google_meta", {}).get("file_format") == "PDF/Adobe Acrobat"
    assert raw.get("google_meta", {}).get("mime") == "application/pdf"
