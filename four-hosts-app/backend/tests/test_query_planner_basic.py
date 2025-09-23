import pytest
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("aiohttp")

from search.query_planner import QueryPlanner, PlannerConfig, QueryCandidate
from services.search_apis import BaseSearchAPI, SearchConfig, SearchResult


@pytest.mark.asyncio
async def test_query_planner_deduplicates_variations():
    cfg = PlannerConfig(max_candidates=4, stage_order=["rule_based"], enable_llm=False)
    planner = QueryPlanner(cfg)
    plan = await planner.initial_plan(
        seed_query="battery recycling market",
        paradigm="bernard",
        additional_queries=["battery recycling market", "Battery recycling market"],
    )
    queries = [candidate.query for candidate in plan]
    assert queries, "planner should return candidates"
    assert len(queries) == len(set(queries)), "planner must deduplicate queries"


class _PlannerStubAPI(BaseSearchAPI):
    def __init__(self):
        super().__init__(api_key="stub", rate=100)

    async def search(self, query: str, cfg: SearchConfig):
        return [
            SearchResult(
                title=f"{query} result",
                url=f"https://example.com/{query.replace(' ', '_')}",
                snippet="",
                source="stub",
                domain="example.com",
                raw_data={},
            )
        ]


@pytest.mark.asyncio
async def test_provider_uses_planned_candidates(monkeypatch):
    api = _PlannerStubAPI()

    async def _boom(*args, **kwargs):  # pragma: no cover - safety
        raise AssertionError("generate_query_variations should not be called")

    monkeypatch.setattr(api.qopt, "generate_query_variations", _boom)

    planned = [
        QueryCandidate(query="alpha beta", stage="rule_based", label="primary"),
        QueryCandidate(query="gamma delta", stage="paradigm", label="pattern"),
    ]
    cfg = SearchConfig(max_results=5)
    results = await api.search_with_variations(planned[0].query, cfg, planned=planned)
    assert {r.raw_data.get("query_variant") for r in results} == {
        "rule_based:primary",
        "paradigm:pattern",
    }
    assert {r.url for r in results} == {
        "https://example.com/alpha_beta",
        "https://example.com/gamma_delta",
    }


@pytest.mark.asyncio
async def test_planner_respects_stage_order():
    cfg = PlannerConfig(
        max_candidates=6,
        stage_order=["paradigm", "rule_based"],
        enable_llm=False,
        per_stage_caps={
            "paradigm": 3,
            "rule_based": 3,
            "llm": 0,
            "agentic": 0,
            "context": 0,
        },
    )
    planner = QueryPlanner(cfg)
    plan = await planner.initial_plan(
        seed_query="battery recycling market growth 2024",
        paradigm="maeve",
        additional_queries=["battery recycling market growth"],
    )
    assert plan, "planner must return candidates"
    paradigm_seen = [cand for cand in plan if cand.stage == "paradigm"]
    rule_seen = [cand for cand in plan if cand.stage == "rule_based"]
    assert paradigm_seen, "expected paradigm stage candidates"
    assert rule_seen, "expected rule-based candidates"
    assert plan[0].stage == "paradigm"
    # Respect per-stage caps
    assert len(paradigm_seen) <= cfg.per_stage_caps["paradigm"]
    assert len(rule_seen) <= cfg.per_stage_caps["rule_based"]


@pytest.mark.asyncio
async def test_default_stage_order_includes_context_and_agentic(monkeypatch):
    cfg = PlannerConfig(enable_llm=False)
    planner = QueryPlanner(cfg)

    async def _fake_rule(**_kwargs):
        return [
            QueryCandidate(query="rule", stage="rule_based", label="primary")
        ]

    async def _fake_paradigm(**_kwargs):
        return [
            QueryCandidate(query="paradigm", stage="paradigm", label="pattern")
        ]

    async def _fake_context(additional_queries, _cfg):  # pragma: no cover - guard
        assert additional_queries == ["ctx"], "context stage should receive refined queries"
        return [
            QueryCandidate(query="ctx", stage="context", label="context_1")
        ]

    async def _fake_agentic(**_kwargs):
        return [
            QueryCandidate(query="followup", stage="agentic", label="followup_1")
        ]

    monkeypatch.setattr(planner.rule_stage, "generate", _fake_rule)
    monkeypatch.setattr(planner.paradigm_stage, "generate", _fake_paradigm)
    monkeypatch.setattr(planner.context_stage, "generate", _fake_context)
    monkeypatch.setattr(planner.agentic_stage, "generate", _fake_agentic)

    plan = await planner.initial_plan(
        seed_query="seed",
        paradigm="bernard",
        additional_queries=["ctx"],
    )

    stages = {candidate.stage for candidate in plan}
    assert {"rule_based", "paradigm", "context", "agentic"}.issubset(stages)
