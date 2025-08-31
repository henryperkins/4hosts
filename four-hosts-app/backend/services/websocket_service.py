"""
WebSocket Support for Real-time Progress Tracking
Phase 5: Production-Ready Features
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Set, Any, Optional, List
from enum import Enum
import uuid

from fastapi import WebSocket, WebSocketDisconnect, Depends, Query, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from pydantic import BaseModel, Field

from services.auth import decode_token, TokenData, UserRole

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            await websocket.send_json(message.model_dump(mode="json"))
        except Exception as e:
            logger.error(f"Error sending to WebSocket: {e}")
            await self.disconnect(websocket)

    async def send_to_user(self, user_id: str, message: WSMessage):
        """Send a message to all connections for a user"""
        if user_id in self.active_connections:
            disconnected = []
            for websocket in self.active_connections[user_id]:
                try:
                    await websocket.send_json(message.model_dump(mode="json"))
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

        # Send to subscribers
        if research_id in self.research_subscriptions:
            disconnected = []
            for websocket in self.research_subscriptions[research_id]:
                try:
                    await websocket.send_json(message.model_dump(mode="json"))
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


# --- Progress Tracker ---


class ProgressTracker:
    """Tracks and broadcasts research progress"""

    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.research_progress: Dict[str, Dict[str, Any]] = {}

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
            "phase": "initialization",
            "progress": 0,
            "sources_found": 0,
            "sources_analyzed": 0,
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

    async def update_progress(
        self,
        research_id: str,
        phase: Optional[str] = None,
        progress: Optional[int] = None,
        sources_found: Optional[int] = None,
        sources_analyzed: Optional[int] = None,
        custom_data: Optional[Dict[str, Any]] = None,
    ):
        """Update research progress"""
        if research_id not in self.research_progress:
            return

        progress_data = self.research_progress[research_id]

        # Update fields
        if phase and phase != progress_data["phase"]:
            progress_data["phase"] = phase
            await self.connection_manager.broadcast_to_research(
                research_id,
                WSMessage(
                    type=WSEventType.RESEARCH_PHASE_CHANGE,
                    data={
                        "research_id": research_id,
                        "old_phase": progress_data["phase"],
                        "new_phase": phase,
                    },
                ),
            )

        if progress is not None:
            progress_data["progress"] = progress

        if sources_found is not None:
            progress_data["sources_found"] = sources_found

        if sources_analyzed is not None:
            progress_data["sources_analyzed"] = sources_analyzed

        # Broadcast progress update
        update_data = {
            "research_id": research_id,
            "phase": progress_data["phase"],
            "progress": progress_data["progress"],
            "sources_found": progress_data["sources_found"],
            "sources_analyzed": progress_data["sources_analyzed"],
        }

        if custom_data:
            update_data.update(custom_data)

        await self.connection_manager.broadcast_to_research(
            research_id, WSMessage(type=WSEventType.RESEARCH_PROGRESS, data=update_data)
        )
        # Invalidate cached status for this research
        try:
            from services.cache import research_status_cache
            await research_status_cache().delete(research_id)
        except Exception as e:
            logger.debug(f"Cache invalidation (status) failed: {e}")

    async def report_source_found(self, research_id: str, source: Dict[str, Any]):
        """Report a new source found"""
        if research_id not in self.research_progress:
            # Initialize if not exists
            self.research_progress[research_id] = {
                "sources_found": 0,
                "sources_analyzed": 0
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


# --- Research Progress Tracker (alias for compatibility) ---


class ResearchProgressTracker(ProgressTracker):
    """Alias for ProgressTracker to maintain compatibility with imports"""

    async def update_progress(self, research_id: str, message: str, progress: int):
        """Simple update method for compatibility with main.py"""
        await self.connection_manager.broadcast_to_research(
            research_id,
            WSMessage(
                type=WSEventType.RESEARCH_PROGRESS,
                data={
                    "research_id": research_id,
                    "message": message,
                    "progress": progress,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        )
    
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
    
    async def report_search_completed(self, research_id: str, query: str, results_count: int):
        """Report that a search has completed"""
        await self.connection_manager.broadcast_to_research(
            research_id,
            WSMessage(
                type=WSEventType.SEARCH_COMPLETED,
                data={
                    "research_id": research_id,
                    "query": query[:50] + "..." if len(query) > 50 else query,
                    "results_count": results_count,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        )
    
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
                        if text.strip().lower() == "ping":
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
