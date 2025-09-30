import types
import asyncio
import pytest

from services.research_orchestrator import research_orchestrator
from services.classification_engine import HostParadigm


@pytest.mark.asyncio
async def test_e2e_smoke_evidence_bundle(monkeypatch):
    # 1) Stub deep research to return one citation with content span
    from services.deep_research_service import (
        deep_research_service,
        DeepResearchResult,
        ResponseStatus,
    )
    from services.openai_responses_client import Citation

    async def fake_execute_deep_research(**kwargs):
        content = "Deep content shows important fact about example.com"
        citations = [
            Citation(url="https://deep.example/s1", title="Deep Source", start_index=0, end_index=25)
        ]
        return DeepResearchResult(
            research_id=kwargs.get("research_id") or "test",
            status=ResponseStatus("completed"),
            content=content,
            citations=citations,
            tool_calls=[],
            web_search_calls=[],
            paradigm_analysis=None,
            cost_info={"total": 0.0},
        )

    monkeypatch.setattr(deep_research_service, "execute_deep_research", fake_execute_deep_research)

    # 2) Stub search path to avoid network and heavy processing
    async def fake_execute_searches_deterministic(*args, **kwargs):
        return {}

    async def fake_process_results(*args, **kwargs):
        return {
            "results": [],
            "metadata": {"processing_time_seconds": 0.01, "agent_trace": []},
            "sources_used": [],
            "credibility_summary": {},
            "dedup_stats": {},
            "contradictions": {"count": 0, "examples": []},
        }

    monkeypatch.setattr(research_orchestrator, "_execute_searches_deterministic", fake_execute_searches_deterministic)
    monkeypatch.setattr(research_orchestrator, "_process_results", fake_process_results)

    # 3) Return one typed quote from evidence_builder
    from models.evidence import EvidenceQuote, EvidenceBundle
    import services.evidence_builder as eb

    async def fake_build_evidence_bundle(query, results, max_docs=5, quotes_per_doc=1, include_full_content=True, paradigm=None, **_):
        quote = EvidenceQuote(
            id="q001",
            url="https://example.com/a",
            title="Example A",
            domain="example.com",
            quote="Example quote",
        )
        return EvidenceBundle(quotes=[quote], matches=[], by_domain={"example.com": 1}, focus_areas=["theme"], documents=[], documents_token_count=0)

    monkeypatch.setattr(eb, "build_evidence_bundle", fake_build_evidence_bundle)

    # 4) Minimal generator returning an object with metadata to be populated by orchestrator
    from services.answer_generator import answer_orchestrator

    class DummyAnswer:
        def __init__(self):
            self.metadata = {}
            self.sections = []
            self.citations = {}
            self.summary = ""

    async def fake_generate_answer(*args, **kwargs):
        return DummyAnswer()

    monkeypatch.setattr(answer_orchestrator, "generate_answer", fake_generate_answer)

    # 5) Execute orchestrator (synthesis + deep research enabled)
    classification = types.SimpleNamespace(primary_paradigm=HostParadigm.BERNARD, secondary_paradigm=None)
    context = types.SimpleNamespace(original_query="test deep research query", refined_queries=[])
    user_ctx = types.SimpleNamespace(role="PRO", source_limit=10)

    resp = await research_orchestrator.execute_research(
        classification=classification,
        context_engineered=context,
        user_context=user_ctx,
        progress_callback=None,
        research_id="test_e2e_1",
        enable_deep_research=True,
        synthesize_answer=True,
        answer_options={"max_length": 800},
    )

    ans = resp.get("answer")
    assert ans is not None and hasattr(ans, "metadata")
    meta = ans.metadata
    assert "evidence_bundle" in meta
    ebundle = meta["evidence_bundle"]
    assert isinstance(ebundle, dict)
    assert "quotes" in ebundle and isinstance(ebundle["quotes"], list)
    # Ensure there is at least one deep_research quote merged
    assert any((q.get("source_type") == "deep_research") for q in ebundle["quotes"]) or any(
        (q.get("domain") or "").startswith("deep.") for q in ebundle["quotes"] if isinstance(q, dict)
    )
