"""Triage manager for research intake coordination and Kanban exports.

This module computes a lightweight priority rubric for incoming research
requests, tracks their lane as the orchestrator advances through phases,
and exposes a board snapshot that can be streamed to the frontend.

The implementation deliberately avoids any direct dependency on the
websocket layer to prevent circular imports.  Callers may register an
asynchronous broadcaster (typically wired up inside
``services.websocket_service``) which will be invoked whenever the board
changes so that updates can be pushed to subscribers.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import structlog

from services.cache import cache_manager
from services.research_store import research_store
from utils.date_utils import get_current_iso, get_current_utc


logger = structlog.get_logger(__name__)


class TriagePriority(str, Enum):
    """Priority band derived from heuristics."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TriageLane(str, Enum):
    """Kanban lanes used by the operations team."""

    INTAKE = "intake"
    CLASSIFICATION = "classification"
    CONTEXT = "context"
    SEARCH = "search"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    REVIEW = "review"
    BLOCKED = "blocked"
    DONE = "done"


PRIORITY_WEIGHTS: Dict[TriagePriority, int] = {
    TriagePriority.HIGH: 3,
    TriagePriority.MEDIUM: 2,
    TriagePriority.LOW: 1,
}


_DEFAULT_BOARD: Dict[str, Any] = {
    "entries": {},
    "updated_at": get_current_iso(),
}


@dataclass
class TriageEntry:
    research_id: str
    priority: TriagePriority
    score: float
    lane: TriageLane
    user_id: str
    user_role: str
    depth: str
    paradigm: str
    query: str
    created_at: str = field(default_factory=get_current_iso)
    updated_at: str = field(default_factory=get_current_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "research_id": self.research_id,
            "priority": self.priority.value,
            "score": round(self.score, 2),
            "lane": self.lane.value,
            "user_id": self.user_id,
            "user_role": self.user_role,
            "depth": self.depth,
            "paradigm": self.paradigm,
            "query": self.query,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


class TriageManager:
    """Coordinates intake prioritisation and Kanban board state."""

    def __init__(self) -> None:
        self._cache_key = "triage:board"
        self._lock = asyncio.Lock()
        self._broadcaster: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None

    def register_broadcaster(
        self, broadcaster: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> None:
        """Register an async callback invoked after each board mutation."""

        self._broadcaster = broadcaster

    async def initialize_entry(
        self,
        *,
        research_id: str,
        user_id: str,
        user_role: Optional[str],
        depth: Optional[str],
        paradigm: Optional[str],
        query: str,
        triage_context: Optional[Dict[str, Any]] = None,
    ) -> TriageEntry:
        """Create or refresh the triage entry for an incoming request."""

        context = triage_context or {}
        priority, score, reasons = self._score_priority(
            user_role=user_role,
            depth=depth,
            options=context,
            query=query,
        )

        entry = TriageEntry(
            research_id=research_id,
            priority=priority,
            score=score,
            lane=TriageLane.INTAKE,
            user_id=user_id,
            user_role=(user_role or "unknown").lower(),
            depth=(depth or "standard").lower(),
            paradigm=(paradigm or "unknown").lower(),
            query=query,
            metadata={
                "reasons": reasons,
                "context": context,
            },
        )

        async with self._lock:
            board = await self._load_board()
            board["entries"][research_id] = entry.as_dict()
            board["updated_at"] = get_current_iso()
            await self._persist_board(board)

        await research_store.update_fields(research_id, {"triage": entry.as_dict()})
        await self._broadcast(board)
        return entry

    async def update_lane(
        self,
        research_id: str,
        *,
        phase: Optional[str] = None,
        status: Optional[str] = None,
        metadata_update: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Advance the triage lane when orchestration phases change."""

        lane = self._lane_for_phase(phase, status=status)
        if lane is None:
            return

        async with self._lock:
            board = await self._load_board()
            entry = board["entries"].get(research_id)
            if not entry:
                return
            if entry.get("lane") == lane.value and status is None:
                return

            entry["lane"] = lane.value
            entry["updated_at"] = get_current_iso()
            if metadata_update:
                metadata = entry.get("metadata")
                if not isinstance(metadata, dict):
                    metadata = {}
                metadata.update(metadata_update)
                entry["metadata"] = metadata
            board["entries"][research_id] = entry
            board["updated_at"] = entry["updated_at"]
            await self._persist_board(board)

        await research_store.update_fields(research_id, {"triage": entry})
        await self._broadcast(board)

    async def mark_completed(self, research_id: str) -> None:
        """Mark a research request as completed."""

        await self.update_lane(research_id, status="completed")

    async def mark_complete(self, research_id: str) -> None:
        """Backward compatible alias for mark_completed."""

        await self.mark_completed(research_id)

    async def mark_failed(self, research_id: str) -> None:
        await self.update_lane(research_id, status="failed")

    async def mark_cancelled(self, research_id: str) -> None:
        await self.update_lane(research_id, status="cancelled")

    async def mark_blocked(self, research_id: str, *, reason: Optional[str] = None) -> None:
        """Mark a research request as blocked and optionally record the reason."""

        metadata = {"blocked_reason": reason} if reason else None
        await self.update_lane(research_id, status="blocked", metadata_update=metadata)

    async def remove_entry(self, research_id: str) -> None:
        async with self._lock:
            board = await self._load_board()
            if research_id in board["entries"]:
                board["entries"].pop(research_id)
                board["updated_at"] = get_current_iso()
                await self._persist_board(board)
            else:
                return

        await self._broadcast(board)

    async def snapshot(self) -> Dict[str, Any]:
        """Return the rendered board grouped by lane."""

        board = await self._load_board()
        return self._render_board(board)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _load_board(self) -> Dict[str, Any]:
        cached = await cache_manager.get_kv(self._cache_key)
        if not cached:
            return {"entries": {}, "updated_at": get_current_iso()}
        try:
            entries = cached.get("entries", {}) if isinstance(cached, dict) else {}
            updated_at = cached.get("updated_at") if isinstance(cached, dict) else get_current_iso()
            return {
                "entries": dict(entries),
                "updated_at": updated_at or get_current_iso(),
            }
        except Exception:
            return {"entries": {}, "updated_at": get_current_iso()}

    async def _persist_board(self, board: Dict[str, Any]) -> None:
        await cache_manager.set_kv(self._cache_key, board, ttl=24 * 3600)

    async def _broadcast(self, board: Dict[str, Any]) -> None:
        if not self._broadcaster:
            return
        try:
            payload = self._render_board(board)
            await self._broadcaster(payload)
        except Exception as exc:
            logger.debug("triage.broadcast_failed", error=str(exc))

    def _render_board(self, board: Dict[str, Any]) -> Dict[str, Any]:
        lanes: Dict[str, List[Dict[str, Any]]] = {}
        entries: Dict[str, Dict[str, Any]] = board.get("entries", {})
        for entry in entries.values():
            lane = entry.get("lane", TriageLane.INTAKE.value)
            lanes.setdefault(lane, []).append(entry)

        def _sort_key(item: Dict[str, Any]) -> Tuple[int, float, str]:
            try:
                priority_enum = TriagePriority(item.get("priority", TriagePriority.LOW.value))
            except ValueError:
                priority_enum = TriagePriority.LOW
            priority = PRIORITY_WEIGHTS.get(priority_enum, 1)
            score = float(item.get("score", 0.0))
            return (-priority, -score, item.get("created_at", ""))

        sorted_lanes: Dict[str, List[Dict[str, Any]]] = {}
        for lane_enum in TriageLane:
            bucket = lanes.get(lane_enum.value, [])
            if bucket:
                sorted_lanes[lane_enum.value] = sorted(bucket, key=_sort_key)
            else:
                sorted_lanes[lane_enum.value] = []

        return {
            "updated_at": board.get("updated_at", get_current_iso()),
            "lanes": sorted_lanes,
            "entry_count": len(entries),
        }

    def _score_priority(
        self,
        *,
        user_role: Optional[str],
        depth: Optional[str],
        options: Dict[str, Any],
        query: str,
    ) -> Tuple[TriagePriority, float, List[str]]:
        role = (user_role or "unknown").lower()
        depth_val = (depth or "standard").lower()

        score = 1.0
        reasons: List[str] = []

        if role in {"enterprise", "admin"}:
            score += 3
            reasons.append("enterprise_tier")
        elif role == "pro":
            score += 2
            reasons.append("pro_tier")
        elif role == "basic":
            score += 1
            reasons.append("basic_tier")

        if depth_val == "deep_research":
            score += 3
            reasons.append("deep_research")
        elif depth_val == "deep":
            score += 2
            reasons.append("deep_analysis")
        elif depth_val == "quick":
            score -= 0.5
            reasons.append("quick_mode")

        try:
            max_sources = int(options.get("max_sources", 0) or 0)
            if max_sources >= 200:
                score += 1.5
                reasons.append("high_source_budget")
            elif max_sources >= 120:
                score += 1
                reasons.append("elevated_source_budget")
        except Exception:
            pass

        if not bool(options.get("enable_real_search", True)):
            score -= 0.5
            reasons.append("simulated_search")

        if options.get("paradigm_override"):
            score += 0.5
            reasons.append("paradigm_override")

        query_length = len(query or "")
        if query_length >= 180:
            score += 1.5
            reasons.append("long_query")
        elif query_length >= 120:
            score += 1
            reasons.append("detailed_query")

        sentiment = options.get("priority_tags") or []
        if isinstance(sentiment, (list, tuple)) and "escalation" in [str(s).lower() for s in sentiment]:
            score += 1.5
            reasons.append("escalation_tag")

        score = max(0.0, score)

        if score >= 6:
            priority = TriagePriority.HIGH
        elif score >= 3:
            priority = TriagePriority.MEDIUM
        else:
            priority = TriagePriority.LOW

        return priority, score, reasons

    def _lane_for_phase(
        self,
        phase: Optional[str],
        *,
        status: Optional[str] = None,
    ) -> Optional[TriageLane]:
        if status:
            status_norm = status.lower()
            if status_norm in {"completed", "partial"}:
                return TriageLane.DONE
            if status_norm in {"failed", "error", "blocked"}:
                return TriageLane.BLOCKED
            if status_norm == "cancelled":
                return TriageLane.BLOCKED

        if not phase:
            return None

        normalized = phase.lower()
        if normalized in {"initialization", "intake"}:
            return TriageLane.INTAKE
        if normalized == "classification":
            return TriageLane.CLASSIFICATION
        if normalized in {"context", "context_engineering"}:
            return TriageLane.CONTEXT
        if normalized == "search":
            return TriageLane.SEARCH
        if normalized in {"analysis", "agentic_loop", "agentic"}:
            return TriageLane.ANALYSIS
        if normalized == "synthesis":
            return TriageLane.SYNTHESIS
        if normalized == "review":
            return TriageLane.REVIEW
        if normalized in {"complete"}:
            return TriageLane.DONE
        return None


triage_manager = TriageManager()


__all__ = [
    "TriageManager",
    "TriageLane",
    "TriagePriority",
    "TriageEntry",
    "triage_manager",
]
