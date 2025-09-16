# flake8: noqa: E402
import asyncio
import pytest
from types import SimpleNamespace

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from services.research_orchestrator import (
    ResearchOrchestrator,
    EarlyRelevanceFilter,
)
from services.search_apis import SearchConfig
from models.context_models import HostParadigm


class DummySM:
    def __init__(self, providers):
        self.apis = {p: object() for p in providers}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass

    async def search_all(
        self, query, config, progress_callback=None, research_id=None
    ):
        # two results from two providers
        return [
            SimpleNamespace(
                title="A",
                url="https://x.test/a",
                snippet="q",
                source_api="google",
                domain="x.test",
                content="c",
            ),
            SimpleNamespace(
                title="B",
                url="https://y.test/b",
                snippet="q",
                source_api="brave",
                domain="y.test",
                content="c",
            ),
        ]

    def _any_session(self):
        return None

    class fetcher:
        @staticmethod
        async def fetch(session, url):
            return "body"


@pytest.mark.asyncio
async def test_cost_attribution_counts_seen_providers(monkeypatch):
    o = ResearchOrchestrator()
    o.search_manager = DummySM(["google", "brave"])

    # stub cost monitor to collect calls
    seen = []

    async def _track(api, n):
        seen.append((api, n))
        return 0.1

    monkeypatch.setattr(o.cost_monitor, "track_search_cost", _track)

    class Ctx:
        role = "PRO"
        source_limit = 5

    user = Ctx()

    async def not_cancelled():
        return False

    _ = await o._execute_searches_deterministic(
        ["hello"],
        HostParadigm.BERNARD,
        user,
        None,
        None,
        not_cancelled,
        cost_accumulator={},
        metrics={},
    )
    # cost attribution should be called for 'google' and 'brave',
    # not 'aggregate'
    assert set([api for api, _ in seen]) == {"google", "brave"}


@pytest.mark.asyncio
async def test_cancelled_error_propagates(monkeypatch):
    o = ResearchOrchestrator()
    sm = DummySM(["google"])

    async def boom(*a, **k):
        raise asyncio.CancelledError()

    sm.search_all = boom
    o.search_manager = sm

    with pytest.raises(asyncio.CancelledError):
        await o._perform_search(
            "q",
            "aggregate",
            SearchConfig(
                max_results=1,
                language="en",
                region="us",
                min_relevance_score=0.1,
            ),
        )


def test_early_filter_keyword_path_no_nameerror():
    f = EarlyRelevanceFilter()
    R = SimpleNamespace(
        title="hello world",
        snippet="about keyword foo with more text",
        domain="example.com",
        language="en",
    )
    assert f.is_relevant(R, "foo", "bernard") is True