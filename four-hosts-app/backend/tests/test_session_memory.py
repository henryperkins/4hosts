import asyncio

import pytest

from services.session_memory import (
    SessionManager,
    SummarizingSession,
    TrimmingSession,
)


@pytest.mark.asyncio
async def test_trimming_session_keeps_last_turns():
    session = TrimmingSession("s1", max_turns=2)
    await session.add_items([
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "three"},
    ])

    history = await session.get_items()
    assert len(history) == 3  # last two turns => user(2)/assistant + user(3)
    assert history[0]["content"].startswith("two")
    assert history[-1]["content"].startswith("three")


class _DummySummariser:
    async def summarize(self, messages):
        return "Summarise", f"Summary for {len(messages)} messages"


@pytest.mark.asyncio
async def test_summarizing_session_inserts_summary():
    summariser = _DummySummariser()
    session = SummarizingSession(
        "s2",
        keep_last_n_turns=1,
        context_limit=2,
        summarizer=summariser,
    )

    await session.add_items([
        {"role": "user", "content": "issue"},
        {"role": "assistant", "content": "step1"},
        {"role": "user", "content": "followup"},
        {"role": "assistant", "content": "step2"},
        {"role": "user", "content": "final"},
    ])

    history = await session.get_items()
    # First message should be synthetic summary prompt
    assert history[0]["role"] == "user"
    assert "Summarise" in history[0]["content"]
    assert history[1]["role"] == "assistant"
    assert "Summary" in history[1]["content"]
    # Newest turn kept verbatim
    assert history[-1]["content"] == "final"


@pytest.mark.asyncio
async def test_session_manager_returns_same_instance():
    manager = SessionManager()
    s_a = await manager.get_or_create("abc", policy="trim", max_turns=1)
    s_b = await manager.get_or_create("abc", policy="trim", max_turns=5)
    assert s_a is s_b

