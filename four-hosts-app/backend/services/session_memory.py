"""Session memory helpers for managing short-term context.

The classes in this module provide two policies for retaining dialogue state
across long-running agent sessions:

* ``TrimmingSession`` keeps the last *N* user turns verbatim.
* ``SummarizingSession`` keeps the last *N* user turns and replaces earlier
  content with an LLM-generated summary.

Both policies mirror the patterns outlined in the OpenAI Agents SDK
documentation but avoid adding a hard dependency on that SDK so the existing
stack can begin experimenting immediately.  The implementations are intentionally
minimal: they manage conversation history, expose async methods compatible with
the rest of the service layer, and provide metadata for observability.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)

Message = Dict[str, Any]


def _is_user_message(item: Message) -> bool:
    role = (item.get("role") or item.get("type"))
    if role is None:
        return False
    role_value = str(role).lower()
    return role_value == "user"


class MemorySession:
    """Interface for memory sessions."""

    async def get_items(self, limit: Optional[int] = None) -> List[Message]:  # pragma: no cover - interface
        raise NotImplementedError

    async def add_items(self, items: Iterable[Message]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def pop_item(self) -> Optional[Message]:  # pragma: no cover - interface
        raise NotImplementedError

    async def clear_session(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class TrimmingSession(MemorySession):
    """Keep only the most recent ``max_turns`` user turns."""

    def __init__(self, session_id: str, max_turns: int = 8):
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self.session_id = session_id
        self.max_turns = max_turns
        self._items: deque[Message] = deque()
        self._lock = asyncio.Lock()

    async def get_items(self, limit: Optional[int] = None) -> List[Message]:
        async with self._lock:
            trimmed = self._trim_locked()
        return trimmed[-limit:] if limit else trimmed

    async def add_items(self, items: Iterable[Message]) -> None:
        collection = list(items or [])
        if not collection:
            return
        async with self._lock:
            self._items.extend(collection)
            self._items = deque(self._trim_locked())

    async def pop_item(self) -> Optional[Message]:
        async with self._lock:
            if not self._items:
                return None
            return self._items.pop()

    async def clear_session(self) -> None:
        async with self._lock:
            self._items.clear()

    async def set_max_turns(self, max_turns: int) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        async with self._lock:
            self.max_turns = max_turns
            self._items = deque(self._trim_locked())

    async def raw_items(self) -> List[Message]:
        async with self._lock:
            return list(self._items)

    # Internal helpers -------------------------------------------------

    def _trim_locked(self) -> List[Message]:
        if not self._items:
            return []

        user_indices: List[int] = []
        items_list = list(self._items)
        for idx in range(len(items_list) - 1, -1, -1):
            if _is_user_message(items_list[idx]):
                user_indices.append(idx)
                if len(user_indices) >= self.max_turns:
                    break

        if len(user_indices) < self.max_turns:
            return items_list

        start_index = user_indices[-1]
        return items_list[start_index:]


SUMMARY_PROMPT = (
    "You are a senior support analyst. Summarise the earlier conversation into"
    " a concise, factual briefing covering devices, reported issues, steps"
    " attempted, blockers, and next recommended actions. Use bullet lists"
    " where appropriate and do not invent new facts."
)


class LLMSummarizer:
    """Minimal LLM backed summariser used by :class:`SummarizingSession`."""

    def __init__(
        self,
        completion_model: str = "o3-mini",
        *,
        prompt: str = SUMMARY_PROMPT,
        max_tokens: int = 400,
    ) -> None:
        self.model = completion_model
        self.prompt = prompt
        self.max_tokens = max_tokens

        try:
            # Import lazily to avoid circular imports at module load
            from services.llm_client import llm_client  # type: ignore

            self._llm_client = llm_client
        except Exception:
            self._llm_client = None

    async def summarize(self, messages: List[Message]) -> Tuple[str, str]:
        if not self._llm_client:
            raise RuntimeError("LLM client is unavailable for summarisation")

        # Convert conversation to a compact transcript for the prompt
        lines: List[str] = []
        for message in messages:
            role = message.get("role") or message.get("type") or "assistant"
            text = message.get("content")
            if isinstance(text, list):
                # Some events may store a list of annotations; keep a minimal view
                text = " ".join(str(t) for t in text if isinstance(t, str))
            if not isinstance(text, str):
                text = str(text)
            if text:
                lines.append(f"{role.upper()}: {text.strip()}")

        transcript = "\n".join(lines)
        prompt = f"{self.prompt}\n\nTranscript:\n{transcript}".strip()

        completion = await self._llm_client.generate_completion(
            prompt,
            paradigm="bernard",
            max_tokens=self.max_tokens,
            model=self.model,
            temperature=0.2,
        )

        user_shadow = "Summarise the conversation so far."
        return user_shadow, completion.strip()


@dataclass
class SessionRecord:
    message: Message
    metadata: Dict[str, Any]


class SummarizingSession(MemorySession):
    """Hybrid session that summarises older turns while keeping recent ones."""

    def __init__(
        self,
        session_id: str,
        *,
        keep_last_n_turns: int = 3,
        context_limit: int = 6,
        summarizer: Optional[LLMSummarizer] = None,
    ) -> None:
        if context_limit <= 0:
            raise ValueError("context_limit must be positive")
        if keep_last_n_turns < 0 or keep_last_n_turns > context_limit:
            raise ValueError("keep_last_n_turns must be between 0 and context_limit")

        self.session_id = session_id
        self.keep_last_n_turns = keep_last_n_turns
        self.context_limit = context_limit
        self.summarizer = summarizer or LLMSummarizer()
        self._records: deque[SessionRecord] = deque()
        self._lock = asyncio.Lock()

    async def get_items(self, limit: Optional[int] = None) -> List[Message]:
        async with self._lock:
            data = [rec.message for rec in self._records]
        return data[-limit:] if limit else data

    async def add_items(self, items: Iterable[Message]) -> None:
        async with self._lock:
            for item in items or []:
                self._records.append(SessionRecord(message=dict(item), metadata={}))

            need_summary, boundary = self._should_summarize_locked()

        if not need_summary:
            return

        await self._summarize_and_apply(boundary)

    async def pop_item(self) -> Optional[Message]:
        async with self._lock:
            if not self._records:
                return None
            return self._records.pop().message

    async def clear_session(self) -> None:
        async with self._lock:
            self._records.clear()

    async def get_full_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        async with self._lock:
            data = [
                {"message": dict(rec.message), "metadata": dict(rec.metadata)}
                for rec in self._records
            ]
        return data[-limit:] if limit else data

    # Internal helpers -------------------------------------------------

    async def _summarize_and_apply(self, boundary: int) -> None:
        async with self._lock:
            snapshot = list(self._records)
        prefix = [rec.message for rec in snapshot[:boundary]]

        try:
            shadow_user, summary = await self.summarizer.summarize(prefix)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("summary_generation_failed", error=str(exc))
            return

        async with self._lock:
            # Re-check in case new items arrived while the summary was built
            need_summary, new_boundary = self._should_summarize_locked()
            if not need_summary:
                return

            retained = list(self._records)[new_boundary:]
            self._records.clear()
            self._records.append(
                SessionRecord(
                    message={"role": "user", "content": shadow_user},
                    metadata={"synthetic": True, "kind": "history_summary_prompt"},
                )
            )
            self._records.append(
                SessionRecord(
                    message={"role": "assistant", "content": summary},
                    metadata={"synthetic": True, "kind": "history_summary"},
                )
            )
            self._records.extend(retained)

    def _should_summarize_locked(self) -> Tuple[bool, int]:
        user_turns = [idx for idx, rec in enumerate(self._records) if _is_user_message(rec.message) and not rec.metadata.get("synthetic")]
        if len(user_turns) <= self.context_limit:
            return False, -1
        if self.keep_last_n_turns == 0:
            return True, len(self._records)
        boundary_index = user_turns[-self.keep_last_n_turns]
        if boundary_index <= 0:
            return False, -1
        return True, boundary_index


class SessionManager:
    """Factory/registry that returns per-ID session instances."""

    def __init__(self) -> None:
        self._sessions: Dict[str, MemorySession] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        session_id: str,
        *,
        policy: str = "trim",
        **kwargs: Any,
    ) -> MemorySession:
        async with self._lock:
            existing = self._sessions.get(session_id)
            if existing:
                return existing

            if policy == "trim":
                session = TrimmingSession(session_id, max_turns=int(kwargs.get("max_turns", 8)))
            elif policy == "summary":
                keep_last = int(kwargs.get("keep_last_n_turns", 3))
                context_limit = int(kwargs.get("context_limit", 6))
                summarizer = kwargs.get("summarizer")
                if summarizer is None:
                    summarizer = LLMSummarizer()
                session = SummarizingSession(
                    session_id,
                    keep_last_n_turns=keep_last,
                    context_limit=context_limit,
                    summarizer=summarizer,
                )
            else:
                raise ValueError(f"Unknown session policy: {policy}")

            self._sessions[session_id] = session
            return session

    async def clear(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session:
            await session.clear_session()


# Shared manager instance for convenience
session_manager = SessionManager()

