import os
import pytest
import asyncio

from backend.services.context_engineering import OptimizeLayer
from backend.services.classification_engine import ClassificationResult, HostParadigm, QueryFeatures


def _cls(query: str = "effective ways to cut cloud costs") -> ClassificationResult:
    feats = QueryFeatures(
        text=query,
        tokens=query.split(),
        entities=[],
        intent_signals=[],
        domain=None,
        urgency_score=0.5,
        complexity_score=0.5,
        emotional_valence=0.0,
    )
    return ClassificationResult(
        query=query,
        primary_paradigm=HostParadigm.BERNARD,
        secondary_paradigm=None,
        distribution={HostParadigm.BERNARD: 1.0},
        confidence=0.8,
        features=feats,
        reasoning={HostParadigm.BERNARD: ["stub"]},
        signals={},
    )


@pytest.mark.asyncio
async def test_llm_query_optimizer_integration(monkeypatch):
    # Enable feature flag
    os.environ["ENABLE_QUERY_LLM"] = "1"

    # Stub propose_semantic_variations to avoid external LLM
    called = {"n": 0}

    async def fake_propose(query: str, paradigm: str = "bernard", *, max_variants: int = 4, key_terms=None):
        called["n"] += 1
        return [
            "optimize kubernetes spend with right-sizing",
            "reduce cloud costs using autoscaling and spot instances",
        ]

    monkeypatch.setenv("ENABLE_QUERY_LLM", "1")
    import importlib
    mod = importlib.import_module("backend.services.llm_query_optimizer")
    monkeypatch.setattr(mod, "propose_semantic_variations", fake_propose)

    layer = OptimizeLayer()
    out = await layer.process(_cls(), None)

    # Expect LLM variations merged
    vars = out.get("variations", {})
    assert any(v.startswith("optimize kubernetes") for v in vars.values())
    assert any("reduce cloud costs" in v for v in vars.values())

    # Metrics counter should increment
    from backend.services.metrics import metrics
    snap = metrics.get_counters()
    assert any(k == () for k in metrics._counters.get("query_optimizer_llm_invocations", {})), "counter missing"


@pytest.mark.asyncio
async def test_llm_query_optimizer_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_QUERY_LLM", "0")
    # Even if propose exists, it shouldn't be called when flag off
    layer = OptimizeLayer()
    out = await layer.process(_cls(), None)
    # variations exist from heuristic path; but no llm_* keys expected
    vars = out.get("variations", {})
    assert not any(k.startswith("llm_") for k in vars.keys())

