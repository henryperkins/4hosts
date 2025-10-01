from typing import Any, Dict
from unittest.mock import AsyncMock

import pytest

from services.triage import TriageLane, TriageManager


@pytest.mark.asyncio
async def test_snapshot_includes_all_lanes(monkeypatch) -> None:
    """Empty snapshots should enumerate every lane."""

    manager = TriageManager()

    monkeypatch.setattr(
        manager,
        "_cache_key",
        "triage:test",
        raising=False,
    )

    state: Dict[str, Dict[str, Any]] = {}

    async def fake_get(key: str):
        return state.get(key)

    async def fake_set(key: str, value: Dict[str, Any], ttl: int):
        state[key] = value

    monkeypatch.setattr("services.triage.cache_manager.get_kv", fake_get)
    monkeypatch.setattr("services.triage.cache_manager.set_kv", fake_set)
    snapshot = await manager.snapshot()

    assert snapshot["entry_count"] == 0
    assert set(snapshot["lanes"].keys()) == {lane.value for lane in TriageLane}


@pytest.mark.asyncio
async def test_mark_blocked_records_reason(monkeypatch) -> None:
    """Blocked updates should persist the provided reason in metadata."""

    manager = TriageManager()

    broadcaster = AsyncMock()
    manager.register_broadcaster(broadcaster)

    state: Dict[str, Dict[str, Any]] = {}

    async def fake_get(key: str):
        return state.get(key)

    async def fake_set(key: str, value: Dict[str, Any], ttl: int):
        state[key] = value

    monkeypatch.setattr("services.triage.cache_manager.get_kv", fake_get)
    monkeypatch.setattr("services.triage.cache_manager.set_kv", fake_set)
    monkeypatch.setattr(
        "services.triage.research_store.update_fields",
        AsyncMock(),
    )

    entry = await manager.initialize_entry(
        research_id="res-test",
        user_id="user-1",
        user_role="pro",
        depth="deep_research",
        paradigm="maeve",
        query="impact of quantum computing on logistics",
    )
    assert entry.lane == TriageLane.INTAKE

    await manager.mark_blocked("res-test", reason="timeout")

    snapshot_blocked = await manager.snapshot()
    blocked_lane = snapshot_blocked["lanes"][TriageLane.BLOCKED.value]
    assert len(blocked_lane) == 1
    assert blocked_lane[0]["metadata"]["blocked_reason"] == "timeout"

    await manager.mark_completed("res-test")

    snapshot_done = await manager.snapshot()
    assert snapshot_done["lanes"][TriageLane.BLOCKED.value] == []
    done_lane = snapshot_done["lanes"][TriageLane.DONE.value]
    assert len(done_lane) == 1
    assert done_lane[0]["lane"] == TriageLane.DONE.value

    # Broadcaster should have been invoked for each mutation
    assert broadcaster.await_count >= 3
