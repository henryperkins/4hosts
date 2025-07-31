"""
Enhanced WebSocket Service V2 with Proper Context Handling
Includes sequence tracking, proper serialization, and message ordering
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Set, Any, Optional, List, Deque
from collections import deque, defaultdict
from enum import Enum
import uuid
import time

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from models.context_models import WebSocketMessageSchema, HostParadigm
from services.auth import UserRole
from services.memory_management import memory_manager
from services.text_compression import text_compressor

logger = logging.getLogger(__name__)


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


class MessageQueue:
    """Thread-safe message queue with ordering guarantees"""

    def __init__(self, max_size: int = 1000):
        self._queue: Deque[WebSocketMessageSchema] = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
        self._sequence_counter = 0

    async def add(self, message: WebSocketMessageSchema) -> int:
        """Add message and return sequence number"""
        async with self._lock:
            self._sequence_counter += 1
            message.sequence_number = self._sequence_counter
            self._queue.append(message)
            return self._sequence_counter

    async def get_since(self, sequence: int) -> List[WebSocketMessageSchema]:
        """Get all messages since a given sequence number"""
        async with self._lock:
            return [
                msg for msg in self._queue
                if msg.sequence_number > sequence
            ]

    async def get_last_n(self, n: int) -> List[WebSocketMessageSchema]:
        """Get last n messages"""
        async with self._lock:
            return list(self._queue)[-n:]

    async def clear_old(self, age_seconds: int = 3600):
        """Remove messages older than specified age"""
        async with self._lock:
            cutoff = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
            while self._queue and self._queue[0].timestamp < cutoff:
                self._queue.popleft()


class ConnectionManagerV2:
    """Enhanced connection manager with proper context handling"""

    def __init__(self):
        # Active connections by user ID
        self.active_connections: Dict[str, Set[WebSocket]] = defaultdict(set)

        # Connection metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}

        # Research subscriptions (research_id -> set of websockets)
        self.research_subscriptions: Dict[str, Set[WebSocket]] = defaultdict(set)

        # Message queues per research
        self.message_queues: Dict[str, MessageQueue] = {}

        # Global message queue for system messages
        self.system_queue = MessageQueue()

        # Rate limiting
        self.rate_limits: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "reset_time": time.time() + 60}
        )

        # Memory management integration
        self.memory_manager = memory_manager

        # Metrics
        self._metrics = {
            "total_connections": 0,
            "messages_sent": 0,
            "messages_failed": 0,
            "reconnections": 0
        }

    async def connect(
        self,
        websocket: WebSocket,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Accept and register a new WebSocket connection"""
        await websocket.accept()

        # Check if reconnection
        if user_id in self.active_connections:
            self._metrics["reconnections"] += 1

        # Add to active connections
        self.active_connections[user_id].add(websocket)
        self._metrics["total_connections"] += 1

        # Store enhanced metadata
        self.connection_metadata[websocket] = {
            "user_id": user_id,
            "connected_at": datetime.now(timezone.utc),
            "last_activity": datetime.now(timezone.utc),
            "client_info": metadata or {},
            "subscriptions": set(),
            "last_sequence_received": 0,
            "message_count": 0
        }

        # Register connection with memory manager
        await self.memory_manager.register_connection(
            connection_id=str(id(websocket)),
            user_id=user_id,
            metadata=metadata or {}
        )

        # Send connection confirmation with sequence info
        welcome_msg = WebSocketMessageSchema(
            sequence_number=0,
            type=WSEventType.CONNECTED.value,
            data={
                "user_id": user_id,
                "connection_id": id(websocket),
                "server_time": datetime.now(timezone.utc).isoformat(),
                "protocol_version": "2.0",
                "features": {
                    "sequence_tracking": True,
                    "message_replay": True,
                    "compression": True
                }
            }
        )

        await self._send_message(websocket, welcome_msg)
        logger.info(f"WebSocket connected for user {user_id}")

    async def disconnect(self, websocket: WebSocket):
        """Disconnect and cleanup WebSocket connection"""
        metadata = self.connection_metadata.get(websocket)
        if not metadata:
            return

        user_id = metadata["user_id"]

        # Remove from active connections
        self.active_connections[user_id].discard(websocket)
        if not self.active_connections[user_id]:
            del self.active_connections[user_id]

        # Remove from all research subscriptions
        for research_id in list(metadata["subscriptions"]):
            self.research_subscriptions[research_id].discard(websocket)
            if not self.research_subscriptions[research_id]:
                # Clean up empty subscription
                del self.research_subscriptions[research_id]
                # Clean up message queue after delay
                asyncio.create_task(self._cleanup_message_queue(research_id))

        # Cleanup metadata
        del self.connection_metadata[websocket]

        # Unregister from memory manager
        await self.memory_manager.unregister_connection(
            connection_id=str(id(websocket))
        )

        logger.info(f"WebSocket disconnected for user {user_id}")

    async def _cleanup_message_queue(self, research_id: str, delay: int = 300):
        """Clean up message queue after delay if no subscribers"""
        await asyncio.sleep(delay)
        if research_id not in self.research_subscriptions:
            self.message_queues.pop(research_id, None)

    async def subscribe_to_research(self, websocket: WebSocket, research_id: str):
        """Subscribe a WebSocket to research updates with replay"""
        # Add to subscriptions
        self.research_subscriptions[research_id].add(websocket)

        # Update metadata
        if websocket in self.connection_metadata:
            self.connection_metadata[websocket]["subscriptions"].add(research_id)

        # Create message queue if needed
        if research_id not in self.message_queues:
            self.message_queues[research_id] = MessageQueue()

        # Send subscription confirmation
        confirm_msg = WebSocketMessageSchema(
            sequence_number=0,
            type=WSEventType.SYSTEM_NOTIFICATION.value,
            research_id=research_id,
            data={
                "message": f"Subscribed to research {research_id}",
                "research_id": research_id,
                "subscription_time": datetime.now(timezone.utc).isoformat()
            }
        )
        await self._send_message(websocket, confirm_msg)

        # Replay missed messages
        metadata = self.connection_metadata.get(websocket, {})
        last_seq = metadata.get("last_sequence_received", 0)

        if last_seq > 0:
            missed_messages = await self.message_queues[research_id].get_since(last_seq)
            for msg in missed_messages:
                await self._send_message(websocket, msg)

    async def broadcast_to_research(
        self,
        research_id: str,
        event_type: WSEventType,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Broadcast message to all subscribers with guaranteed ordering"""
        # Create message
        message = WebSocketMessageSchema(
            sequence_number=0,  # Will be set by queue
            type=event_type.value,
            research_id=research_id,
            data=self._truncate_data(data),
            metadata=metadata or {}
        )

        # Add to queue and get sequence number
        if research_id not in self.message_queues:
            self.message_queues[research_id] = MessageQueue()

        seq_num = await self.message_queues[research_id].add(message)

        # Broadcast to all subscribers
        subscribers = self.research_subscriptions.get(research_id, set())
        failed_websockets = set()

        for websocket in subscribers:
            try:
                await self._send_message(websocket, message)
                # Update last sequence
                if websocket in self.connection_metadata:
                    self.connection_metadata[websocket]["last_sequence_received"] = seq_num
            except Exception as e:
                logger.error(f"Failed to send to websocket: {e}")
                failed_websockets.add(websocket)

        # Clean up failed connections
        for ws in failed_websockets:
            await self.disconnect(ws)

        return seq_num

    async def broadcast_to_user(
        self,
        user_id: str,
        event_type: WSEventType,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Broadcast message to all connections for a user"""
        message = WebSocketMessageSchema(
            sequence_number=0,
            type=event_type.value,
            data=self._truncate_data(data),
            metadata=metadata or {}
        )

        # Add to system queue
        await self.system_queue.add(message)

        # Send to all user connections
        connections = self.active_connections.get(user_id, set())
        failed_websockets = set()

        for websocket in connections:
            try:
                await self._send_message(websocket, message)
            except Exception as e:
                logger.error(f"Failed to send to websocket: {e}")
                failed_websockets.add(websocket)

        # Clean up failed connections
        for ws in failed_websockets:
            await self.disconnect(ws)

    async def _send_message(self, websocket: WebSocket, message: WebSocketMessageSchema):
        """Send a message with proper error handling and compression"""
        try:
            # Update activity timestamp
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]["last_activity"] = datetime.now(timezone.utc)
                self.connection_metadata[websocket]["message_count"] += 1

            # Get message JSON
            message_json = message.to_json()

            # Apply text compression for large messages
            if len(message_json) > 1000:  # Compress messages > 1KB
                try:
                    compressed_data = await text_compressor.compress_text(
                        message_json,
                        max_length=5000  # Limit to 5KB after compression
                    )
                    message_json = compressed_data
                except Exception as e:
                    logger.warning(f"Text compression failed: {e}")
                    # Fall back to truncation
                    message_json = message_json[:5000] + "..." if len(message_json) > 5000 else message_json

            # Send message
            await websocket.send_text(message_json)
            self._metrics["messages_sent"] += 1

        except WebSocketDisconnect:
            raise
        except Exception as e:
            self._metrics["messages_failed"] += 1
            logger.error(f"WebSocket send error: {e}")
            raise

    def _truncate_data(self, data: Dict[str, Any], max_length: int = 10000) -> Dict[str, Any]:
        """Intelligently truncate data to prevent oversized messages"""
        # This will be replaced with dynamic truncation
        return data

    async def handle_ping(self, websocket: WebSocket):
        """Handle ping/pong for keepalive"""
        pong_msg = WebSocketMessageSchema(
            sequence_number=0,
            type=WSEventType.PONG.value,
            data={"timestamp": datetime.now(timezone.utc).isoformat()}
        )
        await self._send_message(websocket, pong_msg)

    def get_connection_info(self, user_id: str) -> Dict[str, Any]:
        """Get information about user's connections"""
        connections = self.active_connections.get(user_id, set())
        return {
            "user_id": user_id,
            "connection_count": len(connections),
            "connections": [
                {
                    "connection_id": id(ws),
                    "metadata": self.connection_metadata.get(ws, {})
                }
                for ws in connections
            ]
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get connection manager metrics"""
        return {
            **self._metrics,
            "active_users": len(self.active_connections),
            "total_websockets": sum(len(conns) for conns in self.active_connections.values()),
            "active_research": len(self.research_subscriptions),
            "message_queues": len(self.message_queues)
        }

    async def check_rate_limit(self, user_id: str, action: str = "message") -> bool:
        """Check if user is within rate limits"""
        now = time.time()
        user_limits = self.rate_limits[user_id]

        # Reset if window expired
        if now > user_limits["reset_time"]:
            user_limits["count"] = 0
            user_limits["reset_time"] = now + 60

        # Check limit (100 messages per minute)
        if user_limits["count"] >= 100:
            # Send rate limit warning
            await self.broadcast_to_user(
                user_id,
                WSEventType.RATE_LIMIT_WARNING,
                {
                    "message": "Rate limit exceeded. Please slow down.",
                    "reset_time": user_limits["reset_time"]
                }
            )
            return False

        user_limits["count"] += 1
        return True


class ResearchProgressTrackerV2:
    """Enhanced progress tracker with V2 features"""

    def __init__(self, connection_manager: ConnectionManagerV2):
        self.connection_manager = connection_manager

    async def start_research(self, research_id: str, user_id: str, query: str, paradigm: str, research_type: str = "standard"):
        """Start tracking research progress"""
        message = WebSocketMessageSchema(
            sequence_number=0,
            type=WSEventType.RESEARCH_STARTED.value,
            data={
                "research_id": research_id,
                "user_id": user_id,
                "query": query,
                "paradigm": paradigm,
                "research_type": research_type,
                "started_at": datetime.now(timezone.utc).isoformat()
            }
        )
        await self.connection_manager.broadcast_to_research(research_id, message)

    async def update_progress(self, research_id: str, progress: float, status: str, details: dict = None):
        """Update research progress"""
        message = WebSocketMessageSchema(
            sequence_number=0,
            type=WSEventType.RESEARCH_PROGRESS.value,
            data={
                "research_id": research_id,
                "progress": progress,
                "status": status,
                "details": details or {},
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
        )
        await self.connection_manager.broadcast_to_research(research_id, message)


def create_websocket_router_v2(connection_manager: ConnectionManagerV2, progress_tracker: ResearchProgressTrackerV2):
    """Create WebSocket router with V2 features"""
    from fastapi import APIRouter
    router = APIRouter()

    # Add any additional router endpoints here
    return router


# Create global instances
connection_manager_v2 = ConnectionManagerV2()
progress_tracker_v2 = ResearchProgressTrackerV2(connection_manager_v2)
