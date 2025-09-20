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
from typing import Any
import uuid

from utils.date_utils import iso_or_none

from fastapi import WebSocket, WebSocketDisconnect, Depends, Query, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from pydantic import BaseModel, Field

from services.auth_service import decode_token, TokenData, UserRole

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
        # Connection metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        # Research subscriptions (research_id -> set of websockets)
        self.research_subscriptions: Dict[str, Set[WebSocket]] = {}
        # Message history for reconnection
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

        # Cleanup metadata
        del self.connection_metadata[websocket]

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
            await websocket.send_json(self._transform_for_frontend(message))
        except Exception as e:
            logger.error(f"Error sending to WebSocket: {e}")
            await self.disconnect(websocket)

    async def send_to_user(self, user_id: str, message: WSMessage):
        """Send a message to all connections for a user"""
        if user_id in self.active_connections:
            disconnected = []
            for websocket in self.active_connections[user_id]:
                try:
                    await websocket.send_json(self._transform_for_frontend(message))
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
        if research_id in self.research_subscriptions:
            disconnected = []
            for websocket in self.research_subscriptions[research_id]:
                try:
                    await websocket.send_json(self._transform_for_frontend(message))
                except Exception as e:
                    logger.error(f"Error broadcasting to research {research_id}: {e}")
                    disconnected.append(websocket)

            # Clean up disconnected websockets
            for ws in disconnected:
                await self.disconnect(ws)

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

            is_admin = str(user_role).lower() == "admin"
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
        }

        # Clone data and, if cancelled event, include explicit status for UI hooks
        data = dict(ws_message.data or {})
        if ws_message.type == WSEventType.RESEARCH_CANCELLED and "status" not in data:
            data["status"] = "cancelled"

        # Ensure timestamp string
        ts = ws_message.timestamp
        ts_str = iso_or_none(ts) or str(ts)

        return {
            "type": type_mapping.get(ws_message.type, str(ws_message.type)),
            "data": data,
            "timestamp": ts_str,
        }


# --- Progress Tracker ---



class ProgressTracker:
    """Tracks and broadcasts research progress"""

    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.research_progress: Dict[str, Dict[str, Any]] = {}

        # Keep running heart-beat tasks so the frontend never thinks the
        # connection is stale during very long phases (e.g. large LLM
        # synthesis).  A task is created for every `start_research` call and
        # cancelled automatically in `_cleanup`.
        self._heartbeat_tasks: Dict[str, asyncio.Task] = {}

        # Rolling average duration per phase (milliseconds)
        self._phase_stats: Dict[str, List[float]] = {}

        # Weighted progress model (sums to 1.0)
        self._phase_weights: Dict[str, float] = {
            "classification": 0.10,
            "context": 0.15,
            "search": 0.45,
            "processing": 0.10,  # dedup/credibility/filtering
            "synthesis": 0.20,
        }

        # Throttled persistence of progress snapshots to the research store
        self._persist_interval_sec: int = int(os.getenv("PROGRESS_PERSIST_INTERVAL_SEC", "2") or 2)
        self._last_persist: Dict[str, float] = {}

    def _canonical_phase(self, phase: Optional[str]) -> Optional[str]:
        if not phase:
            return None
        p = str(phase).lower()
        if p in {"context_engineering", "contextualization", "context"}:
            return "context"
        if p in {"deduplication", "credibility", "filtering", "agentic_loop", "processing"}:
            return "processing"
        if p in {"classification", "search", "synthesis", "complete", "initialization"}:
            return p
        return p

    async def start_research(
        self, research_id: str, user_id: str, query: str, paradigm: str, depth: str
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

        # Launch background heartbeat so clients receive periodic updates even
        # when no phase events are emitted.
        self._heartbeat_tasks[research_id] = asyncio.create_task(
            self._heartbeat(research_id)
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
    ):
        """Update research progress"""
        if research_id not in self.research_progress:
            return

        # Back-compat: allow old positional pattern
        phase_enum_values = {
            "initialization",
            "search",
            "deduplication",
            "filtering",
            "credibility",
            "agentic_loop",
            "synthesis",
            "complete",
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
                    units[key] = {
                        "done": int(items_done or 0),
                        "total": max(1, int(items_total)),
                    }
                    progress_data["phase_units"] = units
            except Exception:
                pass

        # Compute weighted overall progress if possible; otherwise respect explicit numeric value
        overall: Optional[int] = None
        try:
            weights = self._phase_weights
            canonical_order = ["classification", "context", "search", "processing", "synthesis"]
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

    async def _heartbeat(self, research_id: str, interval: int = 10):
        """Send a heartbeat every *interval* seconds until research ends."""
        try:
            while research_id in self.research_progress:
                await asyncio.sleep(interval)

                if research_id not in self.research_progress:
                    break

                last = self.research_progress[research_id].get("last_update")
                # Only send a ping if nothing was sent for > interval seconds
                if last and (datetime.now(timezone.utc) - last).total_seconds() < interval:
                    continue

                await self.update_progress(
                    research_id,
                    custom_data={"heartbeat": True},
                )
        except asyncio.CancelledError:
            pass

    async def _cleanup(self, research_id: str):
        """Cancel heartbeat and remove state (called on complete / fail)."""
        task = self._heartbeat_tasks.pop(research_id, None)
        if task and not task.done():
            task.cancel()

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
            progress_data["completed_phases"] = set(self._phase_weights.keys())
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
        # Invalidate cached status and results for this research
        try:
            from services.cache import research_status_cache, research_results_cache
            await research_status_cache().delete(research_id)
            await research_results_cache().delete(research_id)
        except Exception as e:
            logger.debug(f"Cache invalidation (complete) failed: {e}")

        # Clean up after delay
        await asyncio.sleep(300)  # Keep for 5 minutes
        if research_id in self.research_progress:
            del self.research_progress[research_id]

        await self._cleanup(research_id)

        # Stop heartbeat
        await self._cleanup(research_id)

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


# --- Research Progress Tracker (alias for compatibility) ---


class ResearchProgressTracker(ProgressTracker):
    """Back-compat wrapper exposing the same flexible signature as ProgressTracker.

    Legacy code may still import this alias and call `update_progress` with the
    original (research_id, message, progress) positional arguments **or** the
    new keyword-rich form.  We therefore forward *args/**kwargs to the parent
    implementation so both styles work transparently.
    """

    async def update_progress(self, research_id: str, *positional: Any, **kwargs):  # type: ignore[override]
        # Detect the classic 3-positional call and translate it into the new
        # keyword structure so downstream consumers remain consistent.
        if positional and len(positional) == 2 and not kwargs:
            message, progress = positional  # type: ignore[assignment]
            kwargs = {
                "message": message,
                "progress": progress,
            }
            positional = ()

        await super().update_progress(research_id, *positional, **kwargs)

    async def report_search_started(self, research_id: str, query: str, engine: str, index: int, total: int):
        """Report that a search has started"""
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
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
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
        """Report that a search has completed"""
        # Calculate up-to-date search counters before broadcasting so the
        # frontend receives a self-contained payload with the latest metrics.
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

        # Broadcast search completion event including derived counters so the
        # frontend can update its UI without inferring state.
        await self.connection_manager.broadcast_to_research(
            research_id,
            WSMessage(
                type=WSEventType.SEARCH_COMPLETED,
                data={
                    "research_id": research_id,
                    "query": query[:50] + "..." if len(query) > 50 else query,
                    "results_count": results_count,
                    # Provide running counters for UI synchronisation.
                    "searches_completed": completed,
                    "total_searches": total_value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            ),
        )

        # Persist counters via a regular progress update so downstream
        # consumers (e.g., other WebSocket clients, database snapshots) stay in
        # sync with the event we just emitted.
        if completed is not None:
            try:
                await self.update_progress(
                    research_id,
                    searches_completed=completed,
                    total_searches=total_value,
                )
            except Exception:
                # Intentionally ignore to avoid breaking the primary event flow.
                pass

    async def report_credibility_check(self, research_id: str, domain: str, score: float):
        """Report credibility check progress"""
        await self.connection_manager.broadcast_to_research(
            research_id,
            WSMessage(
                type=WSEventType.CREDIBILITY_CHECK,
                data={
                    "research_id": research_id,
                    "domain": domain,
                    "score": score,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        )

    async def report_deduplication(self, research_id: str, before_count: int, after_count: int):
        """Report deduplication progress"""
        await self.connection_manager.broadcast_to_research(
            research_id,
            WSMessage(
                type=WSEventType.DEDUPLICATION,
                data={
                    "research_id": research_id,
                    "before_count": before_count,
                    "after_count": after_count,
                    "removed": before_count - after_count,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
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
                    )
                except Exception:
                    pass


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
                # Receive message
                data = await websocket.receive_json()

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

            # Verify access to research before subscribing
            try:
                from services.research_store import research_store  # local import
                research = await research_store.get(research_id)
            except Exception:
                research = None

            is_admin = str(user_data.role).lower() == "admin"
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
# Use the compatibility alias so legacy callers invoking update_progress(message, progress)
# continue to work while newer code can use the richer ProgressTracker API.
progress_tracker = ResearchProgressTracker(connection_manager)
ws_integration = WebSocketIntegration(connection_manager, progress_tracker)
