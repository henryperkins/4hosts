import types
import os
import sys
import pytest

# Ensure backend package root on path for direct test execution
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from services.research_orchestrator import research_orchestrator
from services.classification_engine import HostParadigm
from services.search_apis import SearchResult


@pytest.mark.asyncio
async def test_insufficient_data_gates_synthesis_and_tracks_query_effectiveness(monkeypatch):
    # Arrange: stub searches to produce two queries with no results
    async def fake_execute_searches_deterministic(*args, **kwargs):
        return {
            "q-original": [],
            "q-rewrite": [],
        }

    # And make processing return 0 results with minimal required metadata
    async def fake_process_results(*args, **kwargs):
        return {
            "results": [],
            "metadata": {"processing_time_seconds": 0.01, "agent_trace": []},
            "sources_used": [],
            "credibility_summary": {"average_score": 0.0},
            "dedup_stats": {"original_count": 0, "final_count": 0, "duplicates_removed": 0},
            "contradictions": {"count": 0, "examples": []},
        }

    monkeypatch.setattr(research_orchestrator, "_execute_searches_deterministic", fake_execute_searches_deterministic)
    monkeypatch.setattr(research_orchestrator, "_process_results", fake_process_results)

    # Tighten the gate so it definitely trips
    monkeypatch.setenv("MIN_RESULTS_FOR_SYNTHESIS", "2")
    monkeypatch.setenv("MIN_AVG_CREDIBILITY", "0.9")

    # Minimal inputs
    classification = types.SimpleNamespace(primary_paradigm=HostParadigm.BERNARD, secondary_paradigm=None, confidence=0.8, query="test insufficient data")
    context = types.SimpleNamespace(original_query="test insufficient data", refined_queries=["q-original", "q-rewrite"])
    user_ctx = types.SimpleNamespace(role="PRO", source_limit=5, enable_real_search=False)

    # Bypass fail-fast path that requires configured search providers
    research_orchestrator.search_manager = types.SimpleNamespace(apis={"mock": object()})

    # Act
    resp = await research_orchestrator.execute_research(
        classification=classification,
        context_engineered=context,
        user_context=user_ctx,
        progress_callback=None,
        research_id="test_gate_1",
        enable_deep_research=False,
        synthesize_answer=True,
        answer_options={"max_length": 200},
    )

    # Assert: query effectiveness is present and summarizes both queries
    md = resp.get("metadata", {})
    eff = md.get("query_effectiveness")
    assert isinstance(eff, list) and len(eff) == 2
    assert all("query" in e and "results" in e for e in eff)

    # Assert: insufficient_data is marked in metadata and no answer field exists
    insuff = md.get("insufficient_data")
    assert isinstance(insuff, dict)
    assert resp.get("answer") is None
