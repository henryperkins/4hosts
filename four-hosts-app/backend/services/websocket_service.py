"""
WebSocket Support for Real-time Progress Tracking
Phase 5: Production-Ready Features
"""

import asyncio
import os
import contextlib
import json
import logging
import structlog
from datetime import datetime, timezone, timedelta
from typing import Dict, Set, Any, Optional, List
from enum import Enum
import uuid
import time
from weakref import WeakKeyDictionary

from utils.date_utils import iso_or_none

from core.config import PROGRESS_WS_TIMEOUT_MS, RESULTS_POLL_TIMEOUT_MS

from fastapi import WebSocket, WebSocketDisconnect, Header
from pydantic import BaseModel, Field

from services.auth_service import decode_token, TokenData, UserRole
from services.triage import triage_manager

# Configure logging
logger = structlog.get_logger(__name__)

# --- WebSocket Event Types ---


class WSEventType(str, Enum):
    """WebSocket event types"""

    # Connection events
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"

    # Research events
    RESEARCH_STARTED = "research.started"
    RESEARCH_PROGRESS = "research.progress"
    RESEARCH_PHASE_CHANGE = "research.phase_change"
    RESEARCH_COMPLETED = "research.completed"
    RESEARCH_FAILED = "research.failed"
    RESEARCH_CANCELLED = "research.cancelled"

    # Source events
    SOURCE_FOUND = "source.found"
    SOURCE_ANALYZING = "source.analyzing"
    SOURCE_ANALYZED = "source.analyzed"

    # Search events
    SEARCH_STARTED = "search.started"
    SEARCH_COMPLETED = "search.completed"
    SEARCH_RETRY = "search.retry"

    # Synthesis events
    SYNTHESIS_STARTED = "synthesis.started"
    SYNTHESIS_PROGRESS = "synthesis.progress"
    SYNTHESIS_COMPLETED = "synthesis.completed"

    # Analysis events
    CREDIBILITY_CHECK = "credibility.check"
    DEDUPLICATION = "deduplication.progress"

    # MCP events
    MCP_TOOL_EXECUTING = "mcp.tool_executing"
    MCP_TOOL_COMPLETED = "mcp.tool_completed"

    # System events
    RATE_LIMIT_WARNING = "rate_limit.warning"
    SYSTEM_NOTIFICATION = "system.notification"
    # Evidence events
    EVIDENCE_BUILDER_SKIPPED = "evidence_builder.skipped"
    TRIAGE_BOARD_UPDATE = "triage.board_update"


class WSMessage(BaseModel):
    """WebSocket message structure"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: WSEventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


# --- Connection Manager ---


class ConnectionManager:
    """Manages WebSocket connections and message routing"""

    def __init__(self):
        # Active connections by user ID
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Connection metadata keyed by WebSocket; weak ref to prevent leaks
        self.connection_metadata: "WeakKeyDictionary[WebSocket, Dict[str, Any]]" = WeakKeyDictionary()
        # Research subscriptions (research_id -> set of websockets)
        self.research_subscriptions: Dict[str, Set[WebSocket]] = {}
        # Message history for reconnection (research_id -> List[WSMessage])
        self.message_history: Dict[str, List[WSMessage]] = {}
        self.history_limit = 100

        # Bound history by time to avoid unbounded growth on long sessions
        try:
            self.history_ttl_sec: int = int(os.getenv("WS_HISTORY_TTL_SEC", "900") or 900)
        except Exception:
            self.history_ttl_sec = 900

        # Keepalive task for long-lived websockets behind proxies
        self._keepalive_task: Optional[asyncio.Task] = None
        # Lower default keepalive to better survive strict proxies; override via env.
        self._keepalive_interval_sec: int = int(os.getenv("WS_KEEPALIVE_INTERVAL_SEC", "20") or 20)

        # Retention for completed research progress (seconds) – used by ProgressTracker cleanup
        self._progress_retention_sec: int = int(os.getenv("WS_PROGRESS_RETENTION_SEC", "300") or 300)

        # Cache timeout to surface in resume heartbeats for clients
        self.progress_ws_timeout_ms: int = PROGRESS_WS_TIMEOUT_MS

        # Optional write-pump to decouple slow clients (opt-in via env)
        self._use_write_pump: bool = os.getenv("WS_WRITE_PUMP", "0") == "1"
        self._send_queues: Dict[WebSocket, asyncio.Queue] = {}
        self._write_pumps: Dict[WebSocket, asyncio.Task] = {}

        # Timestamp for last global history sweep
        self._last_history_sweep_ts: float = 0.0

    def _all_live_websockets(self) -> Set[WebSocket]:
        conns: Set[WebSocket] = set()
        for _uid, ws_set in self.active_connections.items():
            conns |= set(ws_set)
        return conns

    async def _keepalive_loop(self):
        try:
            while True:
                await asyncio.sleep(max(10, self._keepalive_interval_sec))
                # Emit a lightweight ping message to each connected client
                payload = WSMessage(
                    type=WSEventType.PING,
                    data={"server_time": datetime.now(timezone.utc).isoformat()},
                )
                for ws in list(self._all_live_websockets()):
                    try:
                        await ws.send_json(self._transform_for_frontend(payload))
                    except Exception:
                        try:
                            await self.disconnect(ws)
                        except Exception:
                            pass
        except asyncio.CancelledError:
            return

    def start_keepalive(self):
        if self._keepalive_task and not self._keepalive_task.done():
            return
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def stop_keepalive(self):
        task = self._keepalive_task
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(Exception):
                await task

    async def connect(
        self,
        websocket: WebSocket,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Accept and register a new WebSocket connection"""
        # Note: WebSocket is already accepted in secure_websocket_endpoint
        # await websocket.accept()

        # Add to active connections
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        self.active_connections[user_id].add(websocket)

        # Store metadata
        self.connection_metadata[websocket] = {
            "user_id": user_id,
            "connected_at": datetime.now(timezone.utc),
            "client_info": metadata or {},
            "subscriptions": set(),
        }

        # Optionally create a per-connection writer pump to isolate slow clients
        if self._use_write_pump and websocket not in self._send_queues:
            q: asyncio.Queue = asyncio.Queue(maxsize=1000)
            self._send_queues[websocket] = q
            self._write_pumps[websocket] = asyncio.create_task(self._writer_pump(websocket, q))

        # Send connection confirmation
        await self.send_to_websocket(
            websocket,
            WSMessage(
                type=WSEventType.CONNECTED,
                data={
                    "user_id": user_id,
                    "connection_id": id(websocket),
                    "server_time": datetime.now(timezone.utc).isoformat(),
                },
            ),
        )

        logger.info(f"WebSocket connected for user {user_id}")

    async def disconnect(self, websocket: WebSocket):
        """Disconnect and cleanup WebSocket connection"""
        metadata = self.connection_metadata.get(websocket)
        if not metadata:
            return

        user_id = metadata["user_id"]

        # Remove from active connections
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]

        # Remove from research subscriptions
        for research_id in list(metadata["subscriptions"]):
            await self.unsubscribe_from_research(websocket, research_id)

        # Cleanup metadata (weak dict suppress if absent)
        with contextlib.suppress(Exception):
            del self.connection_metadata[websocket]

        # Tear down writer pump if present
        task = self._write_pumps.pop(websocket, None)
        if task and not task.done():
            task.cancel()

        self._send_queues.pop(websocket, None)

        logger.info(f"WebSocket disconnected for user {user_id}")

    async def subscribe_to_research(self, websocket: WebSocket, research_id: str):
        """Subscribe a WebSocket to research updates"""
        if research_id not in self.research_subscriptions:
            self.research_subscriptions[research_id] = set()

        self.research_subscriptions[research_id].add(websocket)

        # Update connection metadata
        if websocket in self.connection_metadata:
            self.connection_metadata[websocket]["subscriptions"].add(research_id)

        # Send subscription confirmation
        await self.send_to_websocket(
            websocket,
            WSMessage(
                type=WSEventType.SYSTEM_NOTIFICATION,
                data={
                    "message": f"Subscribed to research {research_id}",
                    "research_id": research_id,
                },
            ),
        )

        # Send any missed messages
        if research_id in self.message_history:
            for message in self.message_history[research_id][-10:]:  # Last 10 messages
                await self.send_to_websocket(websocket, message)

        # Emit a lightweight resume heartbeat so the client knows the stream is active
        try:
            await self.send_to_websocket(
                websocket,
                WSMessage(
                    type=WSEventType.SYSTEM_NOTIFICATION,
                    data={
                        "research_id": research_id,
                        "message": "Progress stream resumed",
                        "heartbeat": True,
                        "resumed": True,
                        "progress_ws_timeout_ms": self.progress_ws_timeout_ms,
                    },
                ),
            )
        except Exception as exc:
            logger.debug("ws.resume_heartbeat_failed", research_id=research_id, error=str(exc))

    async def unsubscribe_from_research(self, websocket: WebSocket, research_id: str):
        """Unsubscribe a WebSocket from research updates"""
        if research_id in self.research_subscriptions:
            self.research_subscriptions[research_id].discard(websocket)
            if not self.research_subscriptions[research_id]:
                del self.research_subscriptions[research_id]

        # Update connection metadata
        if websocket in self.connection_metadata:
            self.connection_metadata[websocket]["subscriptions"].discard(research_id)

    async def send_to_websocket(self, websocket: WebSocket, message: WSMessage):
        """Send a message to a specific WebSocket"""
        try:
            await self._send_json(websocket, self._transform_for_frontend(message))
        except Exception as e:
            logger.error(f"Error sending to WebSocket: {e}")
            await self.disconnect(websocket)

    async def send_to_user(self, user_id: str, message: WSMessage):
        """Send a message to all connections for a user"""
        disconnected = []
        if user_id in self.active_connections:
            for websocket in list(self.active_connections[user_id]):
                try:
                    await self._send_json(websocket, self._transform_for_frontend(message))
                except Exception as e:
                    logger.error(f"Error sending to user {user_id}: {e}")
                    disconnected.append(websocket)

        # Clean up disconnected websockets
        for ws in disconnected:
            await self.disconnect(ws)

    async def broadcast_to_research(self, research_id: str, message: WSMessage):
        """Broadcast a message to all subscribers of a research"""
        # Store in history
        if research_id not in self.message_history:
            self.message_history[research_id] = []

        self.message_history[research_id].append(message)
        if len(self.message_history[research_id]) > self.history_limit:
            self.message_history[research_id] = self.message_history[research_id][
                -self.history_limit :
            ]
        # TTL-based eviction
        try:
            if self.history_ttl_sec > 0:
                cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.history_ttl_sec)
                self.message_history[research_id] = [m for m in self.message_history[research_id] if getattr(m, "timestamp", cutoff) >= cutoff]
        except Exception:
            pass

        # Send to subscribers
        disconnected = []
        if research_id in self.research_subscriptions:
            for websocket in list(self.research_subscriptions[research_id]):
                try:
                    await self._send_json(websocket, self._transform_for_frontend(message))
                except Exception as e:
                    logger.error(f"Error broadcasting to research {research_id}: {e}")
                    disconnected.append(websocket)

        # Clean up disconnected websockets
        for ws in disconnected:
            await self.disconnect(ws)

    # ------------------------------------------------------------------
    # Internal helpers for write-pump and JSON serialization
    # ------------------------------------------------------------------

    async def _writer_pump(self, websocket: WebSocket, q: asyncio.Queue):
        """Background task draining a send queue for a websocket.

        Helps prevent head-of-line blocking when a single slow client cannot
        keep up with the broadcast rate. If the connection closes or the task
        is cancelled, it exits quietly.
        """
        try:
            while True:
                payload = await q.get()
                try:
                    if isinstance(payload, str):
                        await websocket.send_text(payload)
                    else:
                        await websocket.send_json(payload)
                except Exception as e:
                    logger.error(f"Writer pump send failed: {e}")
                    await self.disconnect(websocket)
                    return
        except asyncio.CancelledError:
            return

    async def _send_json(self, websocket: WebSocket, payload: Dict[str, Any]):
        """Efficient JSON send with optional write-pump buffering."""
        # If write pump enabled for this socket, enqueue serialized text
        if self._use_write_pump and websocket in self._send_queues:
            try:
                import orjson

                text = orjson.dumps(payload).decode("utf-8")
            except Exception:
                text = json.dumps(payload, default=str)

            q = self._send_queues[websocket]
            if q.full():
                with contextlib.suppress(Exception):
                    _ = q.get_nowait()
            await q.put(text)
            return

        # Fallback: inline send
        try:
            import orjson

            await websocket.send_text(orjson.dumps(payload).decode("utf-8"))
        except Exception:
            await websocket.send_json(payload)

        # Periodic sweep to prune stale histories (global)
        now_ts = time.time()
        if self.history_ttl_sec > 0 and (now_ts - self._last_history_sweep_ts) >= 60:
            self._last_history_sweep_ts = now_ts
            cutoff_dt = datetime.now(timezone.utc) - timedelta(seconds=self.history_ttl_sec)
            to_delete: List[str] = []
            for rid, msgs in self.message_history.items():
                last_ts = msgs[-1].timestamp if msgs else None
                if not last_ts or last_ts < cutoff_dt:
                    to_delete.append(rid)
            for rid in to_delete:
                self.message_history.pop(rid, None)

    async def handle_client_message(
        self, websocket: WebSocket, message: Dict[str, Any]
    ):
        """Handle incoming message from client with per-message authorization"""
        message_type = message.get("type")

        if message_type == "ping":
            await self.send_to_websocket(
                websocket,
                WSMessage(
                    type=WSEventType.PONG,
                    data={"timestamp": datetime.now(timezone.utc).isoformat()},
                ),
            )
            return

        # Get authenticated user context from connection metadata
        meta = self.connection_metadata.get(websocket, {})
        user_id = meta.get("user_id")
        user_role = meta.get("client_info", {}).get("role") or meta.get("role")

        # Enforce ACL on subscribe/unsubscribe
        if message_type in ("subscribe", "unsubscribe"):
            research_id = message.get("research_id")
            if not research_id:
                await self.send_to_websocket(
                    websocket,
                    WSMessage(
                        type=WSEventType.ERROR,
                        data={"error": "invalid_request", "message": "Missing research_id"},
                    ),
                )
                return

            # Verify ownership or admin
            try:
                from services.research_store import research_store  # local import to avoid cycles
                research = await research_store.get(research_id)
            except Exception:
                research = None

            is_admin = _is_admin_role(user_role)
            if not research:
                await self.send_to_websocket(
                    websocket,
                    WSMessage(
                        type=WSEventType.ERROR,
                        data={"error": "not_found", "message": "Research not found"},
                    ),
                )
                return

            if (research.get("user_id") != str(user_id)) and not is_admin:
                await self.send_to_websocket(
                    websocket,
                    WSMessage(
                        type=WSEventType.ERROR,
                        data={"error": "access_denied", "message": "Access denied"},
                    ),
                )
                return

            if message_type == "subscribe":
                await self.subscribe_to_research(websocket, research_id)
            else:
                await self.unsubscribe_from_research(websocket, research_id)
            return

        # Forward other message types unchanged
        # (Extend here with explicit schema validation as needed)

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        total_connections = sum(
            len(conns) for conns in self.active_connections.values()
        )

        return {
            "total_connections": total_connections,
            "unique_users": len(self.active_connections),
            "active_researches": len(self.research_subscriptions),
            "connections_by_user": {
                user_id: len(conns)
                for user_id, conns in self.active_connections.items()
            },
            "subscriptions_by_research": {
                research_id: len(subs)
                for research_id, subs in self.research_subscriptions.items()
            },
        }

    async def disconnect_all(self):
        """Disconnect all active WebSocket connections"""
        logger.info("Disconnecting all WebSocket connections...")

        # Create a list of all connections to disconnect
        all_connections = []
        for user_connections in self.active_connections.values():
            all_connections.extend(user_connections)

        # Disconnect each connection
        for websocket in all_connections:
            try:
                await self.disconnect(websocket)
                # Try to close the websocket properly
                try:
                    await websocket.close(code=1001, reason="Server shutdown")
                except:
                    pass
            except Exception as e:
                logger.error(f"Error disconnecting websocket: {e}")

        # Clear all data structures
        self.active_connections.clear()
        self.connection_metadata.clear()
        self.research_subscriptions.clear()
        self.message_history.clear()

        logger.info("All WebSocket connections disconnected")

    # --- Message Transformation ---
    def _transform_for_frontend(self, ws_message: WSMessage) -> Dict[str, Any]:
        """Transform backend WSMessage to frontend-validated format.

        - Converts dotted research/source event names to underscored variants
        - Preserves already-accepted dotted names (e.g., search.started)
        - Ensures a top-level ISO timestamp string exists
        """
        type_mapping = {
            # Research events → underscored
            WSEventType.RESEARCH_STARTED: "research_started",
            WSEventType.RESEARCH_PROGRESS: "research_progress",
            WSEventType.RESEARCH_PHASE_CHANGE: "research_phase_change",
            WSEventType.RESEARCH_COMPLETED: "research_completed",
            WSEventType.RESEARCH_FAILED: "research_failed",
            # Cancellation is not explicitly typed on the frontend; treat as progress w/ status
            WSEventType.RESEARCH_CANCELLED: "research_progress",
            # Source events → underscored
            WSEventType.SOURCE_FOUND: "source_found",
            # Not explicitly allowed by FE schema; treat as a progress-style update
            WSEventType.SOURCE_ANALYZING: "research_progress",
            WSEventType.SOURCE_ANALYZED: "source_analyzed",
            # Connection/system events (already accepted as-is)
            WSEventType.CONNECTED: "connected",
            WSEventType.DISCONNECTED: "disconnected",
            WSEventType.ERROR: "error",
            WSEventType.PING: "ping",
            WSEventType.PONG: "pong",
            WSEventType.RATE_LIMIT_WARNING: "rate_limit.warning",
            WSEventType.SYSTEM_NOTIFICATION: "system.notification",
            # Search/analysis events
            # Frontend accepts search.started and search.completed, but not search.retry
            # Map retry → started to satisfy schema while conveying intent
            WSEventType.SEARCH_STARTED: "search.started",
            WSEventType.SEARCH_COMPLETED: "search.completed",
            WSEventType.SEARCH_RETRY: "search.started",
            # Frontend schema does not include synthesis.* events; map them to allowed types
            # so they render as progress/completion without schema errors.
            WSEventType.SYNTHESIS_STARTED: "research_progress",
            WSEventType.SYNTHESIS_PROGRESS: "research_progress",
            WSEventType.SYNTHESIS_COMPLETED: "research_completed",
            WSEventType.CREDIBILITY_CHECK: "credibility.check",
            WSEventType.DEDUPLICATION: "deduplication.progress",
            WSEventType.MCP_TOOL_EXECUTING: "system.notification",
            WSEventType.MCP_TOOL_COMPLETED: "system.notification",
            # Evidence events → explicit frontend hook
            WSEventType.EVIDENCE_BUILDER_SKIPPED: "evidence_builder_skipped",
        }

        # Clone data and, if cancelled event, include explicit status for UI hooks
        data = dict(ws_message.data or {})
        if ws_message.type == WSEventType.RESEARCH_CANCELLED and "status" not in data:
            data["status"] = "cancelled"

        # Ensure timestamp string
        ts = ws_message.timestamp
        ts_str = iso_or_none(ts) or str(ts)

        out: Dict[str, Any] = {
            "id": ws_message.id,
            "type": type_mapping.get(ws_message.type, str(ws_message.type)),
            "data": data,
            "timestamp": ts_str,
        }

        # Preserve retry intent so FE may act differently
        if ws_message.type == WSEventType.SEARCH_RETRY:
            out.setdefault("data", {})["retry"] = True

        return out


# --- Progress Tracker ---

# Phase ordering to prevent regressions
PHASE_ORDER = ["classification", "context_engineering", "search",
              "analysis", "agentic_loop", "synthesis", "complete"]


class ProgressTracker:
    """Tracks and broadcasts research progress"""

    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.research_progress: Dict[str, Dict[str, Any]] = {}
        # Add lock map to prevent race conditions
        self._progress_locks: Dict[str, asyncio.Lock] = {}

        # Keep running heart-beat tasks so the frontend never thinks the
        # connection is stale during very long phases (e.g. large LLM
        # synthesis).  A task is created for every `start_research` call and
        # cancelled automatically in `_cleanup`.
        self._heartbeat_tasks: Dict[str, asyncio.Task] = {}
        # Heartbeat interval derives from configured timeout to ensure
        # clients receive multiple keepalives before the UI grace period lapses.
        try:
            derived = max(5, min(30, int(PROGRESS_WS_TIMEOUT_MS / 3000)))
        except Exception:
            derived = 10
        self._heartbeat_interval_sec: int = derived or 10

        # Rolling average duration per phase (milliseconds)
        self._phase_stats: Dict[str, List[float]] = {}

        # Weighted progress model (sums to 1.0)
        # Align with frontend PhaseTracker which now exposes 8 phases:
        # classification → context_engineering → search → analysis → agentic_loop → synthesis → review → complete
        # We treat the terminal 'complete' phase with a small weight so the meter reaches 100% when the
        # orchestrator signals completion after final review.
        self._phase_weights: Dict[str, float] = {
            "classification": 0.08,
            "context_engineering": 0.12,
            "search": 0.38,
            "analysis": 0.16,   # dedup/credibility/filtering
            "agentic_loop": 0.08,  # iterative follow-ups
            "synthesis": 0.12,
            "review": 0.04,
            "complete": 0.02,
        }

        # Throttled persistence of progress snapshots to the research store
        self._persist_interval_sec: int = int(os.getenv("PROGRESS_PERSIST_INTERVAL_SEC", "2") or 2)
        self._last_persist: Dict[str, float] = {}

    def _canonical_phase(self, phase: Optional[str]) -> Optional[str]:
        if not phase:
            return None
        p = str(phase).lower()
        # Normalize to frontend-visible phase names
        if p in {"context_engineering", "contextualization", "context"}:
            return "context_engineering"
        if p in {"deduplication", "credibility", "filtering", "processing"}:
            return "analysis"
        if p in {"agentic_loop", "agentic", "followups", "followup_loop"}:
            return "agentic_loop"
        if p in {"classification", "search", "synthesis", "complete", "initialization"}:
            return p
        return p

    async def start_research(
        self,
        research_id: str,
        user_id: str,
        query: str,
        paradigm: str,
        depth: str,
        *,
        user_role: Optional[str] = None,
        triage_context: Optional[Dict[str, Any]] = None,
    ):
        """Track start of research"""
        self.research_progress[research_id] = {
            "user_id": user_id,
            "query": query,
            "paradigm": paradigm,
            "depth": depth,
            "started_at": datetime.now(timezone.utc),
            "last_update": datetime.now(timezone.utc),
            "phase": "initialization",
            "progress": 0,
            "sources_found": 0,
            "sources_analyzed": 0,
            "total_searches": 0,
            "searches_completed": 0,
            "high_quality_sources": 0,
            "duplicates_removed": 0,
            "mcp_tools_used": 0,
            "phase_start": datetime.now(timezone.utc),
            # Track units and completions for weighted progress
            "phase_units": {},
            "completed_phases": set(),
        }

        # Broadcast start event
        await self.connection_manager.broadcast_to_research(
            research_id,
            WSMessage(
                type=WSEventType.RESEARCH_STARTED,
                data={
                    "research_id": research_id,
                    "query": query,
                    "paradigm": paradigm,
                    "depth": depth,
                    "started_at": self.research_progress[research_id][
                        "started_at"
                    ].isoformat(),
                },
            ),
        )

        # Initialise triage board entry for Kanban consumers
        try:
            await triage_manager.initialize_entry(
                research_id=research_id,
                user_id=user_id,
                user_role=user_role,
                depth=depth,
                paradigm=paradigm,
                query=query,
                triage_context=triage_context,
            )
            await triage_manager.update_lane(research_id, phase="initialization")
        except Exception as exc:
            logger.debug("triage.init_failed", research_id=research_id, error=str(exc))

        # Launch background heartbeat so clients receive periodic updates even
        # when no phase events are emitted.
        self._heartbeat_tasks[research_id] = asyncio.create_task(
            self._heartbeat(research_id, interval=self._heartbeat_interval_sec)
        )

    async def update_progress(
        self,
        research_id: str,
        *positional: Any,
        message: Optional[str] = None,
        progress: Optional[int] = None,
        phase: Optional[str] = None,
        sources_found: Optional[int] = None,
        sources_analyzed: Optional[int] = None,
        total_searches: Optional[int] = None,
        searches_completed: Optional[int] = None,
        high_quality_sources: Optional[int] = None,
        duplicates_removed: Optional[int] = None,
        mcp_tools_used: Optional[int] = None,
        items_done: Optional[int] = None,
        items_total: Optional[int] = None,
        custom_data: Optional[Dict[str, Any]] = None,
        recompute: bool = True,
    ):
        """Update research progress with lock protection"""
        if research_id not in self.research_progress:
            return

        # Get or create lock for this research ID
        lock = self._progress_locks.setdefault(research_id, asyncio.Lock())
        try:
            # Add timeout to prevent deadlocks
            async with asyncio.timeout(5):
                async with lock:
                    await self._update_progress_inner(
                research_id, *positional, message=message, progress=progress,
                phase=phase, sources_found=sources_found,
                sources_analyzed=sources_analyzed, total_searches=total_searches,
                searches_completed=searches_completed,
                high_quality_sources=high_quality_sources,
                duplicates_removed=duplicates_removed,
                mcp_tools_used=mcp_tools_used,
                items_done=items_done, items_total=items_total,
                custom_data=custom_data, recompute=recompute
                    )
        except asyncio.TimeoutError:
            logger.warning(f"Progress update timeout for research {research_id}")
            return

    async def _update_progress_inner(
        self,
        research_id: str,
        *positional: Any,
        message: Optional[str] = None,
        progress: Optional[int] = None,
        phase: Optional[str] = None,
        sources_found: Optional[int] = None,
        sources_analyzed: Optional[int] = None,
        total_searches: Optional[int] = None,
        searches_completed: Optional[int] = None,
        high_quality_sources: Optional[int] = None,
        duplicates_removed: Optional[int] = None,
        mcp_tools_used: Optional[int] = None,
        items_done: Optional[int] = None,
        items_total: Optional[int] = None,
        custom_data: Optional[Dict[str, Any]] = None,
        recompute: bool = True,
    ):
        """Inner update method (must be called under lock)"""

        # Back-compat: allow old positional pattern
        phase_enum_values = {
            # Frontend-visible canonical phases
            "classification",
            "context_engineering",
            "search",
            "analysis",
            "agentic_loop",
            "synthesis",
            "complete",
            # Back-compat synonyms
            "initialization",
            "deduplication",
            "filtering",
            "credibility",
            "context",
            "processing",
        }

        phase_from_pos: Optional[str] = None
        message_from_pos: Optional[str] = None
        progress_from_pos: Optional[int] = None

        if positional:
            # First positional could historically be either a phase name or a
            # free-form message. Heuristic: if it matches known enum values,
            # treat as phase, else as message.
            first = positional[0]
            if isinstance(first, str):
                if first in phase_enum_values:
                    phase_from_pos = first
                else:
                    message_from_pos = first
            # Second positional (if int) is progress percentage
            if len(positional) > 1 and isinstance(positional[1], (int, float)):
                progress_from_pos = int(positional[1])

        # Resolve final values with priority: explicit kwarg > positional > None
        phase = phase if phase is not None else phase_from_pos
        message = message if message is not None else message_from_pos
        progress = progress if progress is not None else progress_from_pos

        progress_data = self.research_progress[research_id]

        # Canonicalize for consistency
        phase = self._canonical_phase(phase)

        # Prevent phase regressions
        if phase and phase in PHASE_ORDER:
            current_phase = progress_data.get("phase")
            if current_phase in PHASE_ORDER:
                try:
                    if PHASE_ORDER.index(phase) < PHASE_ORDER.index(current_phase):
                        return  # Ignore out-of-order updates
                except ValueError:
                    pass

        # Update fields
        if phase and phase != progress_data["phase"]:
            old_phase = progress_data["phase"]
            progress_data["phase"] = phase
            # Collect phase duration metric
            try:
                started = progress_data.get("phase_start")
                if started:
                    elapsed_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000
                    self._phase_stats.setdefault(old_phase, []).append(elapsed_ms)
            except Exception:
                pass

            # Reset phase_start for new phase
            progress_data["phase_start"] = datetime.now(timezone.utc)
            # Mark previous (non-initialization) phase as completed
            try:
                if old_phase and old_phase not in {None, "initialization"}:
                    cset = progress_data.get("completed_phases") or set()
                    # normalize old_phase key
                    cset.add(self._canonical_phase(old_phase))
                    progress_data["completed_phases"] = cset
            except Exception:
                pass
            await self.connection_manager.broadcast_to_research(
                research_id,
                WSMessage(
                    type=WSEventType.RESEARCH_PHASE_CHANGE,
                    data={
                        "research_id": research_id,
                        "old_phase": old_phase,
                        "new_phase": phase,
                    },
                ),
            )

        # Capture real units of work when provided (e.g., search queries)
        if items_total is not None:
            try:
                key = self._canonical_phase(progress_data.get("phase")) or ""
                if key:
                    units = progress_data.get("phase_units") or {}
                    # Prevent backwards counts - only allow increases
                    prev_done = units.get(key, {}).get("done", 0)
                    units[key] = {
                        "done": max(prev_done, int(items_done or 0)),
                        "total": max(1, int(items_total)),
                    }
                    progress_data["phase_units"] = units
            except Exception:
                pass

        # Compute weighted overall progress if recompute=True
        overall: Optional[int] = None
        if recompute:
            try:
                weights = self._phase_weights
                # Compute overall progress in the same order as the frontend PhaseTracker
                canonical_order = [
                    "classification",
                    "context_engineering",
                    "search",
                    "analysis",
                    "agentic_loop",
                    "synthesis",
                    "review",
                    "complete",
                ]
                cset = progress_data.get("completed_phases") or set()
                units = progress_data.get("phase_units") or {}
                current = self._canonical_phase(progress_data.get("phase"))
                total = 0.0
                for ph in canonical_order:
                    w = float(weights.get(ph, 0.0))
                    if ph in cset:
                        total += w
                    elif ph == current:
                        # progress within current phase
                        frac = 0.0
                        if ph in units and units[ph].get("total"):
                            frac = max(0.0, min(1.0, units[ph].get("done", 0) / float(units[ph]["total"])) )
                        elif isinstance(progress, (int, float)):
                            frac = max(0.0, min(1.0, float(progress) / 100.0))
                        total += w * frac
                    else:
                        total += 0.0
                overall = int(round(min(max(total, 0.0), 1.0) * 100))
            except Exception:
                overall = None

        if overall is not None:
            progress_data["progress"] = overall
        elif progress is not None:
            progress_data["progress"] = min(max(int(progress), 0), 100)

        # Update timestamp for ETA calculations and heart-beat idle detection
        progress_data["last_update"] = datetime.now(timezone.utc)

        # Calculate ETA from historical average if available
        eta_seconds: Optional[int] = None
        try:
            phase_key = progress_data.get("phase")
            if phase_key and self._phase_stats.get(phase_key):
                avg_ms = sum(self._phase_stats[phase_key]) / len(self._phase_stats[phase_key])
                elapsed_ms = (
                    datetime.now(timezone.utc) - progress_data.get("phase_start", datetime.now(timezone.utc))
                ).total_seconds() * 1000
                remaining_ms = max(0.0, avg_ms - elapsed_ms)
                eta_seconds = int(remaining_ms / 1000)
        except Exception:
            eta_seconds = None

        def _store_metric(key: str, value: Optional[int]):
            if value is None:
                return
            try:
                normalized = max(0, int(value))
            except Exception:
                return
            if progress_data.get(key) != normalized:
                progress_data[key] = normalized

        _store_metric("sources_found", sources_found)
        _store_metric("sources_analyzed", sources_analyzed)
        _store_metric("total_searches", total_searches)
        _store_metric("searches_completed", searches_completed)
        _store_metric("high_quality_sources", high_quality_sources)
        _store_metric("duplicates_removed", duplicates_removed)
        _store_metric("mcp_tools_used", mcp_tools_used)

        # Broadcast progress update
        update_data = {
            "research_id": research_id,
            "phase": progress_data["phase"],
            "progress": progress_data["progress"],
            "sources_found": progress_data["sources_found"],
            "sources_analyzed": progress_data["sources_analyzed"],
            # Help frontend status rendering during progress
            "status": "in_progress",
            "total_searches": progress_data.get("total_searches", 0),
            "searches_completed": progress_data.get("searches_completed", 0),
            "high_quality_sources": progress_data.get("high_quality_sources", 0),
            "duplicates_removed": progress_data.get("duplicates_removed", 0),
            "mcp_tools_used": progress_data.get("mcp_tools_used", 0),
            "progress_ws_timeout_ms": PROGRESS_WS_TIMEOUT_MS,
            "results_poll_timeout_ms": RESULTS_POLL_TIMEOUT_MS,
        }

        if message:
            update_data["message"] = message

        if items_done is not None:
            update_data["items_done"] = items_done
        if items_total is not None:
            update_data["items_total"] = items_total

        if custom_data:
            update_data.update(custom_data)

        if eta_seconds is not None:
            update_data["eta_seconds"] = eta_seconds

        try:
            await triage_manager.update_lane(
                research_id,
                phase=progress_data.get("phase"),
            )
        except Exception as exc:
            logger.debug("triage.update_failed", research_id=research_id, error=str(exc))

        await self.connection_manager.broadcast_to_research(
            research_id, WSMessage(type=WSEventType.RESEARCH_PROGRESS, data=update_data)
        )
        # Invalidate cached status for this research
        try:
            from services.cache import research_status_cache
            await research_status_cache().delete(research_id)
        except Exception as e:
            logger.debug(f"Cache invalidation (status) failed: {e}")

        # Persist a lightweight progress snapshot periodically so REST `/status`
        # stays in sync with live WS updates.
        try:
            import time as _t
            now = _t.time()
            last = float(self._last_persist.get(research_id, 0.0) or 0.0)
            if now - last >= max(1, self._persist_interval_sec):
                self._last_persist[research_id] = now
                snapshot = {
                    "phase": progress_data.get("phase"),
                    "progress": progress_data.get("progress"),
                    "sources_found": progress_data.get("sources_found", 0),
                    "sources_analyzed": progress_data.get("sources_analyzed", 0),
                    "total_searches": progress_data.get("total_searches", 0),
                    "searches_completed": progress_data.get("searches_completed", 0),
                    "high_quality_sources": progress_data.get("high_quality_sources", 0),
                    "duplicates_removed": progress_data.get("duplicates_removed", 0),
                    "mcp_tools_used": progress_data.get("mcp_tools_used", 0),
                    "last_update": progress_data.get("last_update").isoformat() if progress_data.get("last_update") else None,
                }
                if eta_seconds is not None:
                    snapshot["eta_seconds"] = eta_seconds
                # Local import to avoid cycles at module load
                from services.research_store import research_store as _rs
                await _rs.update_fields(research_id, {"progress": snapshot})
        except Exception as e:
            logger.debug(f"Progress persistence skipped: {e}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _heartbeat(self, research_id: str, interval: Optional[int] = None):
        """Send a heartbeat every *interval* seconds until research ends."""
        try:
            interval_sec = interval or self._heartbeat_interval_sec or 10
            while research_id in self.research_progress:
                await asyncio.sleep(interval_sec)

                if research_id not in self.research_progress:
                    break

                last = self.research_progress[research_id].get("last_update")
                # Only send a ping if nothing was sent for > interval seconds
                if last and (datetime.now(timezone.utc) - last).total_seconds() < interval_sec:
                    continue

                # Store previous progress to prevent regression
                prev = self.research_progress[research_id].get("progress", 0)
                await self.update_progress(
                    research_id,
                    custom_data={"heartbeat": True},
                    recompute=False  # Don't recompute weighted progress
                )
                # Ensure progress never goes backwards
                self.research_progress[research_id]["progress"] = max(prev, self.research_progress[research_id].get("progress", 0))
        except asyncio.CancelledError:
            pass

    async def _cleanup(self, research_id: str):
        """Cancel heartbeat and remove state (called on complete / fail)."""
        task = self._heartbeat_tasks.pop(research_id, None)
        if task and not task.done():
            task.cancel()

    async def report_error(self, research_id: str, exc: Exception):
        """Report error and ensure RESEARCH_FAILED is sent"""
        try:
            await self.connection_manager.broadcast_to_research(
                research_id, WSMessage(type=WSEventType.RESEARCH_FAILED,
                                      data={"research_id": research_id,
                                            "error": type(exc).__name__,
                                            "message": str(exc)})
            )
        except Exception as broadcast_error:
            logger.error(f"Failed to broadcast error for research {research_id}: {broadcast_error}")
        finally:
            # Always cleanup even if broadcast fails
            await self._cleanup(research_id)
            # Clean up the lock for this research
            self._progress_locks.pop(research_id, None)

    async def report_search_started(self, research_id: str, query: str, engine: str, index: int, total: int):
        """Broadcast the start of a search operation and sync counters."""
        await self.connection_manager.broadcast_to_research(
            research_id,
            WSMessage(
                type=WSEventType.SEARCH_STARTED,
                data={
                    "research_id": research_id,
                    "query": query[:50] + "..." if len(query) > 50 else query,
                    "engine": engine,
                    "index": index,
                    "total": total,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            ),
        )

        if research_id in self.research_progress:
            try:
                total_int = max(0, int(total))
            except Exception:
                total_int = None
            if total_int is not None:
                progress_state = self.research_progress[research_id]
                current_total = int(progress_state.get("total_searches", 0) or 0)
                if total_int > current_total:
                    progress_state["total_searches"] = total_int

    async def report_search_completed(self, research_id: str, query: str, results_count: int):
        """Broadcast completion of a search and update counters."""

        completed: Optional[int] = None
        total_value: Optional[int] = None

        if research_id in self.research_progress:
            try:
                progress_state = self.research_progress[research_id]
                completed = int(progress_state.get("searches_completed", 0) or 0) + 1
                total_recorded = int(progress_state.get("total_searches", 0) or 0)
                total_value = max(total_recorded, completed)
            except Exception:
                completed = None
                total_value = None

        await self.connection_manager.broadcast_to_research(
            research_id,
            WSMessage(
                type=WSEventType.SEARCH_COMPLETED,
                data={
                    "research_id": research_id,
                    "query": query[:50] + "..." if len(query) > 50 else query,
                    "results_count": results_count,
                    "searches_completed": completed,
                    "total_searches": total_value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            ),
        )

        if completed is not None:
            try:
                await self.update_progress(
                    research_id,
                    searches_completed=completed,
                    total_searches=total_value,
                    recompute=False,
                )
            except Exception:
                pass

    async def report_credibility_check(self, research_id: str, domain: str, score: float):
        """Broadcast an individual credibility evaluation."""
        await self.connection_manager.broadcast_to_research(
            research_id,
            WSMessage(
                type=WSEventType.CREDIBILITY_CHECK,
                data={
                    "research_id": research_id,
                    "domain": domain,
                    "score": score,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            ),
        )

    async def report_deduplication(self, research_id: str, before_count: int, after_count: int):
        """Broadcast deduplication statistics."""
        await self.connection_manager.broadcast_to_research(
            research_id,
            WSMessage(
                type=WSEventType.DEDUPLICATION,
                data={
                    "research_id": research_id,
                    "before_count": before_count,
                    "after_count": after_count,
                    "removed": before_count - after_count,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            ),
        )

        if research_id in self.research_progress:
            try:
                removed_total = max(0, int(before_count) - int(after_count))
            except Exception:
                removed_total = None
            if removed_total is not None:
                try:
                    await self.update_progress(
                        research_id,
                        duplicates_removed=removed_total,
                        recompute=False,
                    )
                except Exception:
                    pass

    async def report_evidence_builder_skipped(self, research_id: str, reason: str | None = None):
        """Broadcast that the evidence builder was skipped."""
        await self.connection_manager.broadcast_to_research(
            research_id,
            WSMessage(
                type=WSEventType.EVIDENCE_BUILDER_SKIPPED,
                data={
                    "research_id": research_id,
                    "reason": reason or "no_search_results",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            ),
        )

    async def report_source_found(self, research_id: str, source: Dict[str, Any]):
        """Report a new source found"""
        if research_id not in self.research_progress:
            # Initialize if not exists
            self.research_progress[research_id] = {
                "sources_found": 0,
                "sources_analyzed": 0,
                "total_searches": 0,
                "searches_completed": 0,
                "high_quality_sources": 0,
                "duplicates_removed": 0,
                "mcp_tools_used": 0
            }

        self.research_progress[research_id]["sources_found"] += 1

        await self.connection_manager.broadcast_to_research(
            research_id,
            WSMessage(
                type=WSEventType.SOURCE_FOUND,
                data={
                    "research_id": research_id,
                    "source": source,
                    "total_sources": self.research_progress[research_id][
                        "sources_found"
                    ],
                },
            ),
        )

    async def report_synthesis_started(self, research_id: str, strategy: str = "default"):
        """Report that synthesis has started"""
        await self.connection_manager.broadcast_to_research(
            research_id,
            WSMessage(
                type=WSEventType.SYNTHESIS_STARTED,
                data={
                    "research_id": research_id,
                    "strategy": strategy,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            ),
        )
        # Update weighted progress using the tracker
        await self.update_progress(
            research_id,
            phase="synthesis",
            message="Starting answer synthesis...",
        )

    async def report_synthesis_progress(self, research_id: str, completed: int, total: int):
        """Incremental synthesis progress (sections generated)."""
        if research_id not in self.research_progress:
            return

        pct_base = 80  # synthesis starts at 80
        pct_range = 15  # up to 95 before completion
        progress_pct = pct_base + int((completed / max(1, total)) * pct_range)

        await self.update_progress(
            research_id,
            phase="synthesis",
            message=f"Synthesis {completed}/{total} sections",
            progress=progress_pct,
            items_done=completed,
            items_total=total,
        )

    async def report_synthesis_completed(self, research_id: str, sections: int, citations: int):
        """Report that synthesis has completed"""
        await self.connection_manager.broadcast_to_research(
            research_id,
            WSMessage(
                type=WSEventType.SYNTHESIS_COMPLETED,
                data={
                    "research_id": research_id,
                    "sections": sections,
                    "citations": citations,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            ),
        )
        await self.update_progress(
            research_id,
            phase="synthesis",
            message=f"Answer synthesis completed ({sections} sections, {citations} citations)",
        )

    async def report_source_analyzed(
        self, research_id: str, source_id: str, analysis: Dict[str, Any]
    ):
        """Report source analysis completion"""
        if research_id not in self.research_progress:
            return

        self.research_progress[research_id]["sources_analyzed"] += 1

        await self.connection_manager.broadcast_to_research(
            research_id,
            WSMessage(
                type=WSEventType.SOURCE_ANALYZED,
                data={
                    "research_id": research_id,
                    "source_id": source_id,
                    "analysis": analysis,
                    "analyzed_count": self.research_progress[research_id][
                        "sources_analyzed"
                    ],
                },
            ),
        )

    async def complete_research(self, research_id: str, result: Dict[str, Any]):
        """Mark research as completed"""
        if research_id not in self.research_progress:
            return

        progress_data = self.research_progress[research_id]
        # Ensure progress shows 100 and phase is complete
        try:
            # Mark all weighted (non-terminal) phases complete then force final state
            weighted = {k for k, v in self._phase_weights.items() if v > 0}
            progress_data["completed_phases"] = weighted
            progress_data["phase"] = "complete"
            progress_data["progress"] = 100
        except Exception:
            pass
        duration = (
            datetime.now(timezone.utc) - progress_data["started_at"]
        ).total_seconds()

        await self.connection_manager.broadcast_to_research(
            research_id,
            WSMessage(
                type=WSEventType.RESEARCH_COMPLETED,
                data={
                    "research_id": research_id,
                    "duration_seconds": duration,
                    "sources_analyzed": progress_data["sources_analyzed"],
                    "result_summary": result,
                },
            ),
        )
        try:
            await triage_manager.mark_complete(research_id)
        except Exception as exc:
            logger.debug("triage.complete_failed", research_id=research_id, error=str(exc))
        # Invalidate cached status and results for this research
        try:
            from services.cache import research_status_cache, research_results_cache
            await research_status_cache().delete(research_id)
            await research_results_cache().delete(research_id)
        except Exception as e:
            logger.debug(f"Cache invalidation (complete) failed: {e}")

        # Clean up after delay (configurable retention)
        await asyncio.sleep(max(0, int(self.connection_manager._progress_retention_sec)))
        if research_id in self.research_progress:
            del self.research_progress[research_id]

        await self._cleanup(research_id)
        # Clean up the lock for this research
        self._progress_locks.pop(research_id, None)

    async def fail_research(
        self,
        research_id: str,
        error: str,
        error_details: Optional[Dict[str, Any]] = None,
    ):
        """Mark research as failed"""
        if research_id not in self.research_progress:
            return

        await self.connection_manager.broadcast_to_research(
            research_id,
            WSMessage(
                type=WSEventType.RESEARCH_FAILED,
                data={
                    "research_id": research_id,
                    "error": error,
                    "details": error_details or {},
                },
            ),
        )
        try:
            await triage_manager.mark_failed(research_id)
        except Exception as exc:
            logger.debug("triage.fail_failed", research_id=research_id, error=str(exc))
        # Invalidate cached status for this research
        try:
            from services.cache import research_status_cache
            await research_status_cache().delete(research_id)
        except Exception as e:
            logger.debug(f"Cache invalidation (failed) failed: {e}")

        # Clean up
        if research_id in self.research_progress:
            del self.research_progress[research_id]

        # Stop heartbeat
        await self._cleanup(research_id)
        # Clean up the lock for this research
        self._progress_locks.pop(research_id, None)



# --- WebSocket Authentication ---
# Import enhanced authentication from websocket_auth module
from services.websocket_auth import (
    authenticate_websocket,
    verify_websocket_rate_limit,
    check_websocket_message_rate,
    check_websocket_subscription_limit,
    cleanup_websocket_connection,
    secure_websocket_endpoint,
)

# --- WebSocket Endpoint ---

from fastapi import APIRouter


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _is_admin_role(role: Any) -> bool:
    """Return True when the supplied role (enum or str) indicates admin."""
    val = getattr(role, "value", role)
    return str(val).lower() == "admin"


def create_websocket_router(
    connection_manager: ConnectionManager, progress_tracker: ProgressTracker
) -> APIRouter:
    """Create WebSocket router"""
    router = APIRouter()

    @router.websocket("/ws")
    async def websocket_endpoint(
        websocket: WebSocket,
        authorization: Optional[str] = Header(None),
        sec_websocket_protocol: Optional[str] = Header(None),
        origin: Optional[str] = Header(None),
        user_agent: Optional[str] = Header(None),
        x_real_ip: Optional[str] = Header(None),
        x_forwarded_for: Optional[str] = Header(None),
    ):
        """Enhanced WebSocket endpoint with security features"""
        # Use secure authentication
        auth_result = await secure_websocket_endpoint(
            websocket,
            authorization=authorization,
            sec_websocket_protocol=sec_websocket_protocol,
            origin=origin,
            user_agent=user_agent,
            x_real_ip=x_real_ip,
            x_forwarded_for=x_forwarded_for,
        )

        if not auth_result:
            return

        user_data = auth_result["user_data"]
        connection_id = auth_result["connection_id"]

        try:
            # Connect with enhanced metadata
            await connection_manager.connect(
                websocket,
                user_data.user_id,
                metadata={
                    "user_agent": auth_result["user_agent"],
                    "origin": auth_result["origin"],
                    "client_ip": auth_result["client_ip"],
                    "connection_id": connection_id,
                    "role": user_data.role.value,
                },
            )

            # Handle messages with rate limiting
            while True:
                # Receive message (tolerate non-JSON ping text)
                try:
                    data = await websocket.receive_json()
                except WebSocketDisconnect:
                    raise
                except Exception:
                    # Attempt to treat text 'ping' frames as heartbeat
                    with contextlib.suppress(Exception):
                        text = await websocket.receive_text()
                        if (text or "").strip().lower() == "ping":
                            await connection_manager.handle_client_message(
                                websocket, {"type": "ping"}
                            )
                            continue
                    # Unknown frame type → ignore and keep loop
                    continue

                # Check message rate limit
                if not await check_websocket_message_rate(
                    user_data.user_id, user_data.role
                ):
                    await websocket.send_json(
                        {
                            "type": "error",
                            "error": "rate_limit_exceeded",
                            "message": "Message rate limit exceeded. Please slow down.",
                        }
                    )
                    continue

                # Check subscription limits for subscribe messages
                if data.get("type") == "subscribe":
                    if not await check_websocket_subscription_limit(
                        connection_id, user_data.role
                    ):
                        await websocket.send_json(
                            {
                                "type": "error",
                                "error": "subscription_limit_exceeded",
                                "message": "Maximum subscriptions reached for this connection.",
                            }
                        )
                        continue

                # Handle message
                await connection_manager.handle_client_message(websocket, data)

        except WebSocketDisconnect:
            await connection_manager.disconnect(websocket)
            await cleanup_websocket_connection(user_data.user_id, connection_id)
        except Exception as e:
            logger.error(f"WebSocket error for user {user_data.user_id}: {e}")
            await connection_manager.disconnect(websocket)
            await cleanup_websocket_connection(user_data.user_id, connection_id)
            try:
                await websocket.close(code=1011, reason="Internal server error")
            except:
                pass

    @router.websocket("/ws/research/{research_id}")
    async def websocket_research_endpoint(
        websocket: WebSocket,
        research_id: str,
        authorization: Optional[str] = Header(None),
        sec_websocket_protocol: Optional[str] = Header(None),
        origin: Optional[str] = Header(None),
        user_agent: Optional[str] = Header(None),
        x_real_ip: Optional[str] = Header(None),
        x_forwarded_for: Optional[str] = Header(None),
    ):
        """
        Authenticate and automatically subscribe this connection to the
        provided research_id. Matches frontend expectation of connecting to
        `/ws/research/{id}` without sending an explicit subscribe message.
        """
        # Authenticate the WebSocket handshake
        auth_result = await secure_websocket_endpoint(
            websocket,
            authorization=authorization,
            sec_websocket_protocol=sec_websocket_protocol,
            origin=origin,
            user_agent=user_agent,
            x_real_ip=x_real_ip,
            x_forwarded_for=x_forwarded_for,
        )

        if not auth_result:
            return

        user_data = auth_result["user_data"]
        connection_id = auth_result["connection_id"]

        try:
            # Register connection with metadata
            await connection_manager.connect(
                websocket,
                user_data.user_id,
                metadata={
                    "user_agent": auth_result["user_agent"],
                    "origin": auth_result["origin"],
                    "client_ip": auth_result["client_ip"],
                    "connection_id": connection_id,
                    "role": user_data.role.value,
                },
            )

            allowed_triage_roles = {UserRole.ADMIN, UserRole.ENTERPRISE}
            is_admin = _is_admin_role(user_data.role)

            if research_id == "triage-board":
                if getattr(user_data, "role", None) not in allowed_triage_roles:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "error": "access_denied",
                            "message": "Triage board access restricted",
                        }
                    )
                    await websocket.close(code=1008, reason="Access denied")
                    return
            else:
                # Verify access to research before subscribing
                try:
                    from services.research_store import research_store  # local import
                    research = await research_store.get(research_id)
                except Exception:
                    research = None

                if (not research) or (
                    (str(research.get("user_id")) != str(user_data.user_id)) and not is_admin
                ):
                    await websocket.send_json(
                        {
                            "type": "error",
                            "error": "access_denied",
                            "message": "Access denied or research not found",
                        }
                    )
                    await websocket.close(code=1008, reason="Access denied")
                    return

            # Auto-subscribe to updates for this research
            await connection_manager.subscribe_to_research(websocket, research_id)

            # Pump incoming messages (e.g., ping) through handler
            while True:
                try:
                    data = await websocket.receive_json()
                except WebSocketDisconnect:
                    raise
                except Exception:
                    # Non-JSON: try text ping
                    try:
                        text = await websocket.receive_text()
                        if (text or "").strip().lower() == "ping":
                            await connection_manager.handle_client_message(
                                websocket, {"type": "ping"}
                            )
                    except Exception:
                        pass
                    continue

                if not await check_websocket_message_rate(
                    user_data.user_id, user_data.role
                ):
                    await websocket.send_json(
                        {
                            "type": "error",
                            "error": "rate_limit_exceeded",
                            "message": "Message rate limit exceeded. Please slow down.",
                        }
                    )
                    continue

                await connection_manager.handle_client_message(websocket, data)

        except WebSocketDisconnect:
            await connection_manager.disconnect(websocket)
            await cleanup_websocket_connection(user_data.user_id, connection_id)
        except Exception as e:
            logger.error(
                f"WebSocket (auto-subscribe) error for user {user_data.user_id}: {e}"
            )
            await connection_manager.disconnect(websocket)
            await cleanup_websocket_connection(user_data.user_id, connection_id)
            try:
                await websocket.close(code=1011, reason="Internal server error")
            except:
                pass

    return router


# --- Integration with Research Service ---


class WebSocketIntegration:
    """Integration helpers for WebSocket updates"""

    def __init__(
        self, connection_manager: ConnectionManager, progress_tracker: ProgressTracker
    ):
        self.connection_manager = connection_manager
        self.progress_tracker = progress_tracker

    async def notify_rate_limit_warning(
        self, user_id: str, limit_type: str, remaining: int, reset_time: datetime
    ):
        """Notify user of approaching rate limit"""
        await self.connection_manager.send_to_user(
            user_id,
            WSMessage(
                type=WSEventType.RATE_LIMIT_WARNING,
                data={
                    "limit_type": limit_type,
                    "remaining": remaining,
                    "reset_time": reset_time.isoformat(),
                    "message": f"Approaching {limit_type} rate limit. {remaining} requests remaining.",
                },
            ),
        )

    async def send_system_notification(
        self,
        user_id: str,
        message: str,
        level: str = "info",
        data: Optional[Dict[str, Any]] = None,
    ):
        """Send system notification to user"""
        await self.connection_manager.send_to_user(
            user_id,
            WSMessage(
                type=WSEventType.SYSTEM_NOTIFICATION,
                data={"message": message, "level": level, "data": data or {}},
            ),
        )


# --- Create global instances ---

connection_manager = ConnectionManager()
progress_tracker = ProgressTracker(connection_manager)


async def _broadcast_triage_board(payload: Dict[str, Any]) -> None:
    await connection_manager.broadcast_to_research(
        "triage-board",
        WSMessage(type=WSEventType.TRIAGE_BOARD_UPDATE, data=payload),
    )


triage_manager.register_broadcaster(_broadcast_triage_board)
ws_integration = WebSocketIntegration(connection_manager, progress_tracker)
