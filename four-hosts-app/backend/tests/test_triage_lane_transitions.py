"""Integration-style test validating triage lane transitions via progress updates."""

from __future__ import annotations

import sys
import types
from enum import Enum
from typing import Any, Dict

import pytest
from unittest.mock import AsyncMock, MagicMock

# Provide a lightweight stub for services.auth_service to avoid heavy DB imports.
auth_stub = types.ModuleType("services.auth_service")
websocket_auth_stub = types.ModuleType("services.websocket_auth")


class _TokenData:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _UserRole(str, Enum):
    FREE = "free"


def _decode_token(_token: str) -> _TokenData:  # pragma: no cover - simple stub
    return _TokenData(user_id="stub-user", role=_UserRole.FREE)


auth_stub.decode_token = _decode_token
auth_stub.TokenData = _TokenData
auth_stub.UserRole = _UserRole
sys.modules.setdefault("services.auth_service", auth_stub)


async def _async_noop(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - trivial stub
    return None


def _identity_decorator(func):  # pragma: no cover - trivial stub
    return func


websocket_auth_stub.authenticate_websocket = _async_noop
websocket_auth_stub.verify_websocket_rate_limit = _async_noop
websocket_auth_stub.check_websocket_message_rate = _async_noop
websocket_auth_stub.check_websocket_subscription_limit = _async_noop
websocket_auth_stub.cleanup_websocket_connection = _async_noop
websocket_auth_stub.secure_websocket_endpoint = _identity_decorator
sys.modules.setdefault("services.websocket_auth", websocket_auth_stub)

from services.triage import TriageManager
from services.websocket_service import ProgressTracker


@pytest.mark.asyncio
async def test_triage_lane_progression_through_websocket_updates(monkeypatch) -> None:
    """Research progress events should advance the triage entry through every lane."""

    # In-memory cache backing so the triage manager avoids Redis in tests.
    cache_state: Dict[str, Dict[str, Any]] = {}

    async def fake_get(key: str):
        return cache_state.get(key)

    async def fake_set(key: str, value: Dict[str, Any], ttl: int):
        cache_state[key] = value

    monkeypatch.setattr("services.triage.cache_manager.get_kv", fake_get)
    monkeypatch.setattr("services.triage.cache_manager.set_kv", fake_set)
    monkeypatch.setattr(
        "services.triage.research_store.update_fields",
        AsyncMock(),
    )

    triage_instance = TriageManager()
    # Swap the global triage manager reference used by the websocket service.
    monkeypatch.setattr("services.triage.triage_manager", triage_instance, raising=False)
    monkeypatch.setattr("services.websocket_service.triage_manager", triage_instance)

    connection_manager = MagicMock()
    connection_manager.broadcast_to_research = AsyncMock()
    tracker = ProgressTracker(connection_manager)

    research_id = "res-triage-001"
    await tracker.start_research(
        research_id=research_id,
        user_id="user-123",
        query="impact of agentic research on triage delivery",
        paradigm="bernard",
        depth="deep_research",
        user_role="pro",
    )

    async def current_lane() -> str | None:
        snapshot = await triage_instance.snapshot()
        for lane, entries in snapshot["lanes"].items():
            for entry in entries:
                if entry["research_id"] == research_id:
                    return lane
        return None

    assert await current_lane() == "intake"

    phase_expectations = [
        ("classification", "classification"),
        ("context", "context"),
        ("search", "search"),
        ("analysis", "analysis"),
        ("synthesis", "synthesis"),
        ("review", "review"),
        ("complete", "done"),
    ]

    for phase, expected_lane in phase_expectations:
        await tracker.update_progress(
            research_id,
            phase=phase,
            message=f"progressing to {phase}",
        )
        assert await current_lane() == expected_lane

    await tracker._cleanup(research_id)
