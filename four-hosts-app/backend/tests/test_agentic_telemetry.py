import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from services.agentic_process import run_followups


def _make_context(themes=None, focus=None):
    return SimpleNamespace(
        write_output=SimpleNamespace(key_themes=themes or []),
        isolate_output=SimpleNamespace(focus_areas=focus or []),
    )


@pytest.mark.asyncio
async def test_agentic_loop_telemetry_logs_iterations():
    context = _make_context(themes=["alpha"], focus=["beta"])
    planner = Mock()
    planner.followups = AsyncMock(return_value=[SimpleNamespace(query="follow-1")])

    with patch("services.agentic_process.logger") as mock_logger:
        await run_followups(
            original_query="alpha",
            context_engineered=context,
            paradigm_code="bernard",
            planner=planner,
            seed_query="alpha",
            executed_queries=set(),
            coverage_sources=[],
            max_iterations=1,
            coverage_threshold=0.9,
        )

    info_calls = [call for call in mock_logger.info.call_args_list]
    messages = [call.args[0] for call in info_calls if call.args]

    assert "agentic_coverage_evaluation" in messages
    assert "agentic_loop_complete" in messages
    planner.followups.assert_awaited()


@pytest.mark.asyncio
async def test_agentic_loop_high_coverage_skips_followups():
    coverage_sources = [
        {"title": "Alpha Insights", "snippet": "Detailed alpha analysis"}
    ]
    context = _make_context(themes=["alpha"], focus=[])
    planner = Mock()
    planner.followups = AsyncMock()

    new_candidates, followup_results, coverage_ratio, missing_terms, _ = await run_followups(
        original_query="alpha",
        context_engineered=context,
        paradigm_code="teddy",
        planner=planner,
        seed_query="alpha",
        executed_queries=set(),
        coverage_sources=coverage_sources,
        max_iterations=2,
        coverage_threshold=0.5,
    )

    assert not new_candidates
    assert followup_results == {}
    assert coverage_ratio >= 0.5
    assert not missing_terms
    planner.followups.assert_not_awaited()
