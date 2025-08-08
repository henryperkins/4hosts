import asyncio
import pytest

from services.enhanced_integration import EnhancedAnswerGenerationOrchestrator
from services.classification_engine import HostParadigm
from models.synthesis_models import SynthesisContext
from backend.contracts import ResearchStatus, GeneratedAnswer


@pytest.mark.asyncio
async def test_zero_sources_returns_failed_no_sources():
    orch = EnhancedAnswerGenerationOrchestrator()
    ctx = SynthesisContext(
        query="What is context engineering?",
        paradigm="bernard",
        search_results=[],
        context_engineering={},
    )

    out = await orch.generate_answer(ctx, HostParadigm.BERNARD)
    assert isinstance(out, GeneratedAnswer)
    assert out.status is ResearchStatus.FAILED_NO_SOURCES
    assert out.content_md == ""


@pytest.mark.asyncio
async def test_generator_exception_maps_to_tool_error(monkeypatch):
    orch = EnhancedAnswerGenerationOrchestrator()
    # Provide minimal non-empty results
    results = [
        {
            "url": "https://example.com/a",
            "title": "A",
            "snippet": "alpha",
            "credibility_score": 0.8,
        }
    ]
    ctx = SynthesisContext(
        query="Q",
        paradigm="bernard",
        search_results=results,
        context_engineering={},
    )

    async def boom(_):
        raise Exception("boom")

    monkeypatch.setattr(
        orch.generators[HostParadigm.BERNARD], "generate_answer", boom
    )

    out = await orch.generate_answer(ctx, HostParadigm.BERNARD)
    assert isinstance(out, GeneratedAnswer)
    assert out.status is ResearchStatus.TOOL_ERROR
    assert out.diagnostics.get("error")
    # Citations should reference sources (by url)
    assert out.citations and out.citations[0].url == "https://example.com/a"


@pytest.mark.asyncio
async def test_generator_timeout_maps_to_timeout(monkeypatch):
    orch = EnhancedAnswerGenerationOrchestrator()
    results = [
        {"url": "https://example.com/a", "title": "A", "snippet": "alpha"}
    ]
    ctx = SynthesisContext(
        query="Q",
        paradigm="bernard",
        search_results=results,
        context_engineering={},
    )

    async def timeout(_):
        raise asyncio.TimeoutError()

    monkeypatch.setattr(
        orch.generators[HostParadigm.BERNARD], "generate_answer", timeout
    )

    out = await orch.generate_answer(ctx, HostParadigm.BERNARD)
    assert isinstance(out, GeneratedAnswer)
    assert out.status is ResearchStatus.TIMEOUT


@pytest.mark.asyncio
async def test_happy_path_passthrough(monkeypatch):
    orch = EnhancedAnswerGenerationOrchestrator()
    results = [
        {"url": "https://example.com/a", "title": "A", "snippet": "alpha"}
    ]
    ctx = SynthesisContext(
        query="Q",
        paradigm="bernard",
        search_results=results,
        context_engineering={},
    )

    async def ok(_):
        return {"content": "ok", "paradigm": "bernard", "citations": []}

    monkeypatch.setattr(orch.generators[HostParadigm.BERNARD], "generate_answer", ok)

    out = await orch.generate_answer(ctx, HostParadigm.BERNARD)
    # Normal path returns dict intact (back-compat)
    assert isinstance(out, dict)
    assert out.get("content") == "ok"

