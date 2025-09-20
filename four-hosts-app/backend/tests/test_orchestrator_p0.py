# flake8: noqa: E402
import asyncio
import pytest
from types import SimpleNamespace
pytest.importorskip("pydantic")

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
from search.query_planner import QueryCandidate


class DummySM:
    def __init__(self, providers):
        self.apis = {p: object() for p in providers}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass

    async def search_all(
        self, planned, config, progress_callback=None, research_id=None
    ):
        # two results from two providers
        assert planned, "planner must supply candidates"
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
        [
            QueryCandidate(
                query="hello",
                stage="rule_based",
                label="primary",
            )
        ],
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
            QueryCandidate(query="q", stage="rule_based", label="primary"),
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


@pytest.mark.asyncio
async def test_agentic_followups_trigger(monkeypatch):
    o = ResearchOrchestrator()
    o.search_manager = DummySM(["brave"])
    o.agentic_config.update(
        {
            "enabled": True,
            "max_iterations": 1,
            "coverage_threshold": 0.9,
            "max_new_queries_per_iter": 2,
        }
    )

    base_candidate = QueryCandidate(
        query="community mental health benefits",
        stage="rule_based",
        label="primary",
    )
    follow_candidate = QueryCandidate(
        query="community support resources case study",
        stage="agentic",
        label="followup_1",
    )

    class DummyPlanner:
        def __init__(self, cfg):
            self.cfg = cfg

        async def initial_plan(self, **_kwargs):
            return [base_candidate]

        async def followups(self, **_kwargs):
            return [follow_candidate]

    monkeypatch.setattr("services.research_orchestrator.QueryPlanner", DummyPlanner)

    calls: list[list[str]] = []

    async def fake_execute(self, candidates, *args, **kwargs):
        calls.append([cand.query for cand in candidates])
        if len(calls) == 1:
            return {
                candidates[0].query: [
                    SimpleNamespace(
                        title="General overview",
                        url="https://example.com/a",
                        snippet="high level summary",
                        source_api="brave",
                        domain="example.com",
                        content="",
                    )
                ]
            }
        return {
            cand.query: [
                SimpleNamespace(
                    title="Community support resources",
                    url=f"https://example.org/{idx}",
                    snippet="community support resources success story",
                    source_api="brave",
                    domain="example.org",
                    content="",
                )
            ]
            for idx, cand in enumerate(candidates)
        }

    monkeypatch.setattr(
        ResearchOrchestrator,
        "_execute_searches_deterministic",
        fake_execute,
        raising=False,
    )

    classification = SimpleNamespace(
        query="Community support expansion",
        primary_paradigm=HostParadigm.TEDDY,
        secondary_paradigm=None,
        confidence=0.8,
        reasoning={HostParadigm.TEDDY: ["community focus"]},
    )

    context_engineered = SimpleNamespace(
        original_query="Community support expansion",
        refined_queries=["Community support expansion"],
        write_output=SimpleNamespace(key_themes=["community support resources"]),
        isolate_output=SimpleNamespace(focus_areas=[]),
    )

    user_context = SimpleNamespace(role="PRO", source_limit=5)

    result = await o.execute_research(
        classification=classification,
        context_engineered=context_engineered,
        user_context=user_context,
        progress_callback=None,
        research_id="agentic-test",
        enable_deep_research=False,
        synthesize_answer=False,
        answer_options={},
    )

    assert len(calls) == 2, "follow-up iteration should execute"
    assert follow_candidate.query in calls[1]
    assert result["metadata"]["queries_executed"] == 2
