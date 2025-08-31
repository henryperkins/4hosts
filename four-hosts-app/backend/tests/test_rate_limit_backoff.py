import asyncio
import os
import time
import types
import pytest
from services import search_apis

class DummyResponse:
    def __init__(self, status=429, headers=None):
        self.status = status
        self.headers = headers or {}
        self.headers = {k.lower(): v for k, v in self.headers.items()}
        self._text = ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self):
        return self._text

    async def read(self):
        return b"PDFDATA"

    @property
    def headers(self):
        return self._headers

    @headers.setter
    def headers(self, value):
        self._headers = value

class DummySession:
    def __init__(self, responses):
        self._responses = responses
        self._index = 0

    def get(self, url, timeout=None, headers=None, allow_redirects=True):
        # Return next response
        resp = self._responses[min(self._index, len(self._responses)-1)]
        self._index += 1
        return resp

@pytest.mark.asyncio
async def test_exponential_backoff_jitter(monkeypatch):
    # Ensure deterministic jitter by patching random.uniform
    delays = []
    def fake_uniform(a, b):
        # record requested upper bound
        delays.append(b)
        return b  # choose max to make assertion easier

    monkeypatch.setenv('SEARCH_RATE_LIMIT_BASE_DELAY', '1')
    monkeypatch.setenv('SEARCH_RATE_LIMIT_BACKOFF_FACTOR', '2')
    monkeypatch.setenv('SEARCH_RATE_LIMIT_MAX_DELAY', '5')
    monkeypatch.setenv('SEARCH_RATE_LIMIT_JITTER', 'full')
    monkeypatch.setattr(search_apis, 'random', types.SimpleNamespace(uniform=fake_uniform))

    # Two 429 responses then a 200
    class OKResponse(DummyResponse):
        def __init__(self):
            super().__init__(status=200, headers={})
        async def text(self):
            return '<html><body>ok</body></html>'

    session = DummySession([
        DummyResponse(status=429, headers={'Retry-After': '10'}),
        DummyResponse(status=429, headers={'Retry-After': '10'}),
        OKResponse()
    ])

    start = time.time()
    # Call the internal fetch function directly
    await search_apis.fetch_and_parse_url(session, 'http://example.com')
    await search_apis.fetch_and_parse_url(session, 'http://example.com')
    # After two attempts, attempt counter should be 2 stored on session
    attempt_attr = [attr for attr in dir(session) if attr.startswith('_rate_attempts_')]
    assert attempt_attr, 'Attempt attribute not set on session'
    attempts_recorded = getattr(session, attempt_attr[0])
    assert attempts_recorded == 2
    # Validate delays captured (upper bounds used due to fake_uniform)
    # First attempt upper bound 1, second attempt upper bound 2 (capped by server header 10 and max 5)
    assert delays[0] == 1
    assert delays[1] == 2

    # Ensure total elapsed at least sum of delays (approx) allowing scheduling overhead
    elapsed = time.time() - start
    assert elapsed >= 3 - 0.2

@pytest.mark.asyncio
async def test_backoff_respects_max_and_server_retry(monkeypatch):
    delays = []
    def fake_uniform(a, b):
        delays.append(b)
        return b

    monkeypatch.setenv('SEARCH_RATE_LIMIT_BASE_DELAY', '2')
    monkeypatch.setenv('SEARCH_RATE_LIMIT_BACKOFF_FACTOR', '3')
    monkeypatch.setenv('SEARCH_RATE_LIMIT_MAX_DELAY', '4')
    monkeypatch.setenv('SEARCH_RATE_LIMIT_JITTER', 'full')
    monkeypatch.setattr(search_apis, 'random', types.SimpleNamespace(uniform=fake_uniform))

    class OKResponse(DummyResponse):
        def __init__(self):
            super().__init__(status=200, headers={})
        async def text(self):
            return '<html><body>ok</body></html>'

    session = DummySession([
        DummyResponse(status=429, headers={'Retry-After': '100'}),  # server asks 100, but capped to 4
        DummyResponse(status=429, headers={'Retry-After': '1'}),    # server asks 1, should use 1
        OKResponse()
    ])

    await search_apis.fetch_and_parse_url(session, 'http://example.com')
    await search_apis.fetch_and_parse_url(session, 'http://example.com')

    # Computed delays before jitter: attempt1=2, attempt2=6 -> capped to 4. Server retry 100 is ignored (capped), server retry 1 overrides computed 4 -> 1
    assert delays[0] == 2  # after cap logic with server 100 => 2 vs 4? Wait: base 2^0=2, min(2,4)=2
    assert delays[1] == 1  # server asks 1 which is less than computed 4
