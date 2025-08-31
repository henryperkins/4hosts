import asyncio
import types
import pytest
from services import search_apis


class DummyResponse:
    def __init__(self, status=200, headers=None, json_data=None, text_data="", raise_json=False):
        self.status = status
        self._headers = {k.lower(): v for k, v in (headers or {}).items()}
        self._json_data = json_data
        self._text = text_data
        self._raise_json = raise_json

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    @property
    def headers(self):
        return self._headers

    async def json(self):
        if self._raise_json:
            raise ValueError("Invalid JSON")
        return self._json_data

    async def text(self):
        return self._text

    async def read(self):
        return self._text.encode("utf-8", errors="replace")


class DummySession:
    def __init__(self, responses):
        self._responses = responses
        self._index = 0

    def get(self, url, params=None, headers=None):
        # Return the next response, clamp to last when out of range
        resp = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return resp


@pytest.mark.asyncio
async def test_ss_parse_valid_and_types():
    api = search_apis.SemanticScholarAPI()
    data = {
        "data": [
            {
                "paperId": "xyz",
                "title": "Test",
                "abstract": "Abstract",
                "year": 2021,
                "citationCount": 5,
                "influentialCitationCount": 1,
            }
        ]
    }
    results = api._parse_semantic_scholar_results(data)
    assert results, "No results parsed"
    r = results[0]
    assert r.title == "Test"
    assert r.url.endswith("/xyz")
    assert r.published_date and r.published_date.year == 2021
    assert r.source == "semantic_scholar"
    assert r.result_type == "academic"


@pytest.mark.asyncio
async def test_ss_parse_alternate_keys_and_type_coercions():
    api = search_apis.SemanticScholarAPI()
    data = {
        "results": [
            {
                "paperId": "abc",
                "title": "Alt",
                "abstract": None,
                "year": "2020",
                "citationCount": "2",
                "influentialCitationCount": "0",
            }
        ]
    }
    res = api._parse_semantic_scholar_results(data)
    assert res and res[0].title == "Alt"
    assert res[0].published_date and res[0].published_date.year == 2020
    # No abstract -> snippet built from counts or blank allowed
    assert isinstance(res[0].snippet, str)


@pytest.mark.asyncio
async def test_ss_parse_top_level_list_and_missing_fields():
    api = search_apis.SemanticScholarAPI()
    data = [
        {
            "paperId": "p1"
            # missing title/abstract/year/etc.
        }
    ]
    res = api._parse_semantic_scholar_results(data)  # type: ignore
    assert res and res[0].url.endswith("/p1")
    assert res[0].title == ""  # defaulted


@pytest.mark.asyncio
async def test_ss_invalid_payload_type_returns_empty():
    api = search_apis.SemanticScholarAPI()
    res = api._parse_semantic_scholar_results("oops")  # type: ignore
    assert res == []


@pytest.mark.asyncio
async def test_ss_search_handles_non_json_200_gracefully(monkeypatch):
    # 200 OK but HTML/non-JSON body -> should log and return []
    api = search_apis.SemanticScholarAPI()
    session = DummySession(
        [
            DummyResponse(
                status=200,
                headers={"Content-Type": "text/html"},
                text_data="<html>error</html>",
                raise_json=True,  # .json() will raise
            )
        ]
    )
    # Monkeypatch the session getter to return our dummy session
    api._get_session = lambda: session  # type: ignore[assignment]
    results = await api.search("q", search_apis.SearchConfig(max_results=3))
    assert results == []


@pytest.mark.asyncio
async def test_ss_search_retries_on_429_with_backoff(monkeypatch):
    # Make sleep deterministic and fast
    delays = []

    async def fake_sleep(d):
        delays.append(d)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    # Deterministic jitter
    def fake_uniform(a, b):
        return b

    monkeypatch.setattr(search_apis, "random", types.SimpleNamespace(uniform=fake_uniform))

    ok_payload = {
        "data": [
            {
                "paperId": "z1",
                "title": "T",
                "abstract": "A",
                "year": 2022,
                "citationCount": 1,
                "influentialCitationCount": 0,
            }
        ]
    }

    session = DummySession(
        [
            DummyResponse(status=429, headers={"Retry-After": "5"}),
            DummyResponse(status=429, headers={"Retry-After": "1"}),
            DummyResponse(status=200, headers={"Content-Type": "application/json"}, json_data=ok_payload),
        ]
    )

    api = search_apis.SemanticScholarAPI()
    api._get_session = lambda: session  # type: ignore[assignment]

    # Control backoff math
    monkeypatch.setenv("SEARCH_RATE_LIMIT_BASE_DELAY", "1")
    monkeypatch.setenv("SEARCH_RATE_LIMIT_BACKOFF_FACTOR", "2")
    monkeypatch.setenv("SEARCH_RATE_LIMIT_MAX_DELAY", "10")
    monkeypatch.setenv("SEARCH_RATE_LIMIT_JITTER", "full")

    results = await api.search("x", search_apis.SearchConfig(max_results=3))
    assert len(results) == 1 and results[0].title == "T"

    # Ensure attempt attribute tracked on session
    attempt_attrs = [a for a in dir(session) if a.startswith("_ss_rate_attempts_")]
    assert attempt_attrs
    assert getattr(session, attempt_attrs[0]) == 2

    # attempts: 1 => computed=1, server=5 -> upper=1 -> delay=1
    # attempts: 2 => computed=2, server=1 -> upper=1 -> delay=1
    assert len(delays) >= 2
    assert delays[0] == 1
    assert delays[1] == 1
