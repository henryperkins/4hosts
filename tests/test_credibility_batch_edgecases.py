"""Edge-case tests for the refactored credibility batch analysis.

These tests patch the expensive network-bound ``get_source_credibility``
function so we can validate batch-level behaviour deterministically and
quickly.
"""

import asyncio
from types import SimpleNamespace

import pytest


# Path to the module under test
import importlib

# Import the public wrapper; this adjusts sys.path for backend modules.
from source_credibility import analyze_source_credibility_batch, CredibilityScore  # noqa: E501

# Now we can safely import the real implementation module and patch it.
cred_mod = importlib.import_module("services.credibility")


@pytest.mark.asyncio
async def test_batch_all_failures_returns_zero(monkeypatch):
    """If every individual credibility check fails, the aggregator returns zeros."""

    async def _always_fail(*args, **kwargs):  # noqa: D401
        raise RuntimeError("boom")

    monkeypatch.setattr(cred_mod, "get_source_credibility", _always_fail)

    sources = [{"domain": f"example{i}.com"} for i in range(3)]

    stats = await analyze_source_credibility_batch(sources, paradigm="bernard")

    assert stats["total_sources"] == 3
    assert stats["average_credibility"] == 0.0
    assert stats["high_credibility_sources"] == 0


@pytest.mark.asyncio
async def test_progress_tracker_called(monkeypatch):
    """Progress tracker receives one update per completed source."""

    # Return deterministic credibility scores without network I/O.
    async def _fake_score(domain, paradigm, **kwargs):  # noqa: D401
        return CredibilityScore(domain=domain, overall_score=0.8)

    monkeypatch.setattr(cred_mod, "get_source_credibility", _fake_score)

    class _Tracker(SimpleNamespace):
        def __init__(self):  # noqa: D401
            super().__init__()
            self.calls: list[dict] = []

        async def update_progress(self, research_id, **kwargs):  # noqa: D401
            self.calls.append(kwargs)

    tracker = _Tracker()

    sources = [{"domain": f"site{i}.org"} for i in range(4)]

    await analyze_source_credibility_batch(
        sources,
        paradigm="bernard",
        progress_tracker=tracker,
        research_id="RID123",
    )

    # Expect exactly one call per source
    assert len(tracker.calls) == len(sources)
