import importlib

import pytest


class _FakeClassification:
    def __init__(self, primary_paradigm):
        self.primary_paradigm = primary_paradigm


@pytest.mark.asyncio
async def test_build_system_prompt_variant_v2(monkeypatch):
    # Force experiments on and v2
    monkeypatch.setenv("ENABLE_PROMPT_AB", "1")
    monkeypatch.setenv("PROMPT_AB_WEIGHTS", '{"v2": 1.0}')

    from backend.services.deep_research_service import (
        DeepResearchService,
        DeepResearchConfig,
        HostParadigm,
    )
    drs = importlib.import_module("backend.services.deep_research_service")
    importlib.reload(drs)

    svc = DeepResearchService()
    cfg = DeepResearchConfig()
    prompt = svc._build_system_prompt(
        query="test",
        classification=_FakeClassification(HostParadigm.BERNARD),
        context_engineering=None,
        config=cfg,
        research_id="rid-1",
    )
    assert cfg.prompt_variant == "v2"
    assert "STRICT: inline citations for all statistics" in prompt


@pytest.mark.asyncio
async def test_build_system_prompt_variant_v1(monkeypatch):
    # Force experiments on and v1
    monkeypatch.setenv("ENABLE_PROMPT_AB", "1")
    monkeypatch.setenv("PROMPT_AB_WEIGHTS", '{"v1": 1.0}')

    from backend.services.deep_research_service import (
        DeepResearchService,
        DeepResearchConfig,
        HostParadigm,
    )
    drs = importlib.import_module("backend.services.deep_research_service")
    importlib.reload(drs)

    svc = DeepResearchService()
    cfg = DeepResearchConfig()
    prompt = svc._build_system_prompt(
        query="test",
        classification=_FakeClassification(HostParadigm.BERNARD),
        context_engineering=None,
        config=cfg,
        research_id="rid-2",
    )
    assert cfg.prompt_variant == "v1"
    assert "rigorous analytical and empirical focus" in prompt
