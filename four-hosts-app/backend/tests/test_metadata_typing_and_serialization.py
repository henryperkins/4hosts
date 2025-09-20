import pytest
from typing import Any, Dict, List

pytest.importorskip("pydantic")

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.research_orchestrator import ResearchOrchestrator
from models.context_models import (
    ClassificationResultSchema,
    QueryFeaturesSchema,
    ContextEngineeredQuerySchema,
)
from services.classification_engine import HostParadigm


@pytest.mark.asyncio
async def test_classification_details_and_metrics_shapes():
    # Arrange: orchestrator with stubbed search manager that returns no results
    orch = ResearchOrchestrator()

    class _StubSearchManager:
        apis = {"google": True}

        async def search_all(self, planned, config, progress_callback=None, research_id=None):
            assert planned, "planner must supply candidates"
            return []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    orch.search_manager = _StubSearchManager()

    classification = ClassificationResultSchema(
        query="test",
        primary_paradigm=HostParadigm.BERNARD,
        secondary_paradigm=HostParadigm.DOLORES,
        distribution={
            HostParadigm.BERNARD: 0.7,
            HostParadigm.DOLORES: 0.3,
        },
        confidence=0.85,
        features=QueryFeaturesSchema(text="test"),
        reasoning={
            HostParadigm.BERNARD: ["analytical signals"],
            HostParadigm.DOLORES: ["revolutionary signals"],
        },
    )

    ctx = ContextEngineeredQuerySchema(
        original_query="test",
        refined_query="test refined",
        paradigm=HostParadigm.BERNARD,
        enhancements=[],
        context_additions=[],
        focus_areas=[],
    )

    class _UserCtx:
        role = "PRO"
        source_limit = 5

    # Act
    resp = await orch.execute_research(
        classification=classification,
        context_engineered=ctx,
        user_context=_UserCtx(),
        progress_callback=None,
        research_id="unit_test_meta",
        enable_deep_research=False,
        synthesize_answer=False,
        answer_options={},
    )

    # Assert: response container
    assert isinstance(resp, dict)

    # metadata present and mapping-like
    md = resp.get("metadata") or {}
    assert isinstance(md, dict)

    # classification_details is present and mapping-like
    cls_details = md.get("classification_details")
    assert isinstance(cls_details, dict)

    # distribution must be Dict[str, float] with at least one entry
    dist = cls_details.get("distribution")
    assert isinstance(dist, dict) and len(dist) > 0, "distribution should be a non-empty dict"
    assert all(isinstance(k, str) for k in dist.keys()), "distribution keys must be strings"
    assert all(isinstance(v, (int, float)) for v in dist.values()), "distribution values must be numeric"

    # reasoning must be Dict[str, List[str]] (allowing empty)
    reasoning = cls_details.get("reasoning")
    assert isinstance(reasoning, dict), "reasoning must be a dict"
    if reasoning:
        assert all(isinstance(k, str) for k in reasoning.keys())
        assert all(isinstance(v, list) for v in reasoning.values())

    # search_metrics.apis_used must always be a list (never a set)
    sm = md.get("search_metrics")
    assert isinstance(sm, dict), "search_metrics must be a dict"
    apis_used = sm.get("apis_used")
    assert isinstance(apis_used, list), "search_metrics.apis_used must be a list"
