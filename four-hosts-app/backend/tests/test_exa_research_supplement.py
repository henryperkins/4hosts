import asyncio
from types import SimpleNamespace
import pathlib
import sys

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from services.research_orchestrator import research_orchestrator
from services.search_apis import SearchResult
from services.exa_research import ExaResearchOutput
from models.context_models import (
    ClassificationResultSchema,
    ContextEngineeredQuerySchema,
    QueryFeaturesSchema,
)
from services.classification_engine import HostParadigm


@pytest.mark.asyncio
async def test_exa_supplement_appends_result(monkeypatch):
    processed = {
        "results": [
            SearchResult(
                title="Brave Result",
                url="https://example.com/brave",
                snippet="Initial brave snippet",
                source="brave",
                content="Original content",
                relevance_score=0.6,
            )
        ],
        "metadata": {},
        "sources_used": ["brave"],
    }

    classification = ClassificationResultSchema(
        query="test query",
        primary_paradigm=HostParadigm.BERNARD,
        distribution={HostParadigm.BERNARD: 0.9},
        confidence=0.8,
        features=QueryFeaturesSchema(
            text="test query",
            tokens=["test"],
            intent_signals=[],
            domain="science",
            urgency_score=0.0,
            complexity_score=0.0,
            emotional_valence=0.0,
        ),
        reasoning={HostParadigm.BERNARD: ["reason"]},
    )

    context = ContextEngineeredQuerySchema(
        original_query="test query",
        refined_query="test query refined",
        paradigm=HostParadigm.BERNARD,
        enhancements=[],
        context_additions=[],
        focus_areas=[],
    )

    user_ctx = SimpleNamespace(source_limit=5, query="test query")

    async def fake_supplement(query: str, highlights, focus=None):
        return ExaResearchOutput(
            summary="Exa summary",
            key_findings=["Finding A"],
            supplemental_sources=[{"title": "Exa Source", "url": "https://exa.ai", "snippet": "Details"}],
        )

    monkeypatch.setenv("ENABLE_EXA_RESEARCH_SUPPLEMENT", "1")
    monkeypatch.setattr(
        "services.exa_research.exa_research_client.is_configured",
        lambda: True,
    )
    monkeypatch.setattr(
        "services.exa_research.exa_research_client.supplement_with_research",
        fake_supplement,
    )

    await research_orchestrator._augment_with_exa_research(
        processed,
        classification,
        context,
        user_ctx,
        progress_callback=None,
        research_id=None,
    )

    assert processed["results"][0].source == "exa_research"
    assert processed["metadata"]["exa_research"]["summary"] == "Exa summary"
    assert "exa_research" in processed["sources_used"]
