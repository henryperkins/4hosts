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

from fastapi import WebSocket, WebSocketDisconnect, Depends, Query
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
    
    # Source events
    SOURCE_FOUND = "source.found"
    SOURCE_ANALYZING = "source.analyzing"
    SOURCE_ANALYZED = "source.analyzed"
    
    # Synthesis events
    SYNTHESIS_STARTED = "synthesis.started"
    SYNTHESIS_PROGRESS = "synthesis.progress"
    SYNTHESIS_COMPLETED = "synthesis.completed"
    
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
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Accept and register a new WebSocket connection"""
        await websocket.accept()
        
        # Add to active connections
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        self.active_connections[user_id].add(websocket)
        
        # Store metadata
        self.connection_metadata[websocket] = {
            "user_id": user_id,
            "connected_at": datetime.now(timezone.utc),
            "client_info": metadata or {},
            "subscriptions": set()
        }
        
        # Send connection confirmation
        await self.send_to_websocket(
            websocket,
            WSMessage(
                type=WSEventType.CONNECTED,
                data={
                    "user_id": user_id,
                    "connection_id": id(websocket),
                    "server_time": datetime.now(timezone.utc).isoformat()
                }
            )
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
    
    async def subscribe_to_research(
        self,
        websocket: WebSocket,
        research_id: str
    ):
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
                    "research_id": research_id
                }
            )
        )
        
        # Send any missed messages
        if research_id in self.message_history:
            for message in self.message_history[research_id][-10:]:  # Last 10 messages
                await self.send_to_websocket(websocket, message)
    
    async def unsubscribe_from_research(
        self,
        websocket: WebSocket,
        research_id: str
    ):
        """Unsubscribe a WebSocket from research updates"""
        if research_id in self.research_subscriptions:
            self.research_subscriptions[research_id].discard(websocket)
            if not self.research_subscriptions[research_id]:
                del self.research_subscriptions[research_id]
        
        # Update connection metadata
        if websocket in self.connection_metadata:
            self.connection_metadata[websocket]["subscriptions"].discard(research_id)
    
    async def send_to_websocket(
        self,
        websocket: WebSocket,
        message: WSMessage
    ):
        """Send a message to a specific WebSocket"""
        try:
            await websocket.send_json(message.model_dump(mode="json"))
        except Exception as e:
            logger.error(f"Error sending to WebSocket: {e}")
            await self.disconnect(websocket)
    
    async def send_to_user(
        self,
        user_id: str,
        message: WSMessage
    ):
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
    
    async def broadcast_to_research(
        self,
        research_id: str,
        message: WSMessage
    ):
        """Broadcast a message to all subscribers of a research"""
        # Store in history
        if research_id not in self.message_history:
            self.message_history[research_id] = []
        
        self.message_history[research_id].append(message)
        if len(self.message_history[research_id]) > self.history_limit:
            self.message_history[research_id] = self.message_history[research_id][-self.history_limit:]
        
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
        self,
        websocket: WebSocket,
        message: Dict[str, Any]
    ):
        """Handle incoming message from client"""
        message_type = message.get("type")
        
        if message_type == "ping":
            await self.send_to_websocket(
                websocket,
                WSMessage(
                    type=WSEventType.PONG,
                    data={"timestamp": datetime.now(timezone.utc).isoformat()}
                )
            )
        
        elif message_type == "subscribe":
            research_id = message.get("research_id")
            if research_id:
                await self.subscribe_to_research(websocket, research_id)
        
        elif message_type == "unsubscribe":
            research_id = message.get("research_id")
            if research_id:
                await self.unsubscribe_from_research(websocket, research_id)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        total_connections = sum(len(conns) for conns in self.active_connections.values())
        
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
            }
        }

# --- Progress Tracker ---

class ProgressTracker:
    """Tracks and broadcasts research progress"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.research_progress: Dict[str, Dict[str, Any]] = {}
    
    async def start_research(
        self,
        research_id: str,
        user_id: str,
        query: str,
        paradigm: str,
        depth: str
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
            "sources_analyzed": 0
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
                    "started_at": self.research_progress[research_id]["started_at"].isoformat()
                }
            )
        )
    
    async def update_progress(
        self,
        research_id: str,
        phase: Optional[str] = None,
        progress: Optional[int] = None,
        sources_found: Optional[int] = None,
        sources_analyzed: Optional[int] = None,
        custom_data: Optional[Dict[str, Any]] = None
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
                        "new_phase": phase
                    }
                )
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
            "sources_analyzed": progress_data["sources_analyzed"]
        }
        
        if custom_data:
            update_data.update(custom_data)
        
        await self.connection_manager.broadcast_to_research(
            research_id,
            WSMessage(
                type=WSEventType.RESEARCH_PROGRESS,
                data=update_data
            )
        )
    
    async def report_source_found(
        self,
        research_id: str,
        source: Dict[str, Any]
    ):
        """Report a new source found"""
        if research_id not in self.research_progress:
            return
        
        self.research_progress[research_id]["sources_found"] += 1
        
        await self.connection_manager.broadcast_to_research(
            research_id,
            WSMessage(
                type=WSEventType.SOURCE_FOUND,
                data={
                    "research_id": research_id,
                    "source": source,
                    "total_sources": self.research_progress[research_id]["sources_found"]
                }
            )
        )
    
    async def report_source_analyzed(
        self,
        research_id: str,
        source_id: str,
        analysis: Dict[str, Any]
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
                    "analyzed_count": self.research_progress[research_id]["sources_analyzed"]
                }
            )
        )
    
    async def complete_research(
        self,
        research_id: str,
        result: Dict[str, Any]
    ):
        """Mark research as completed"""
        if research_id not in self.research_progress:
            return
        
        progress_data = self.research_progress[research_id]
        duration = (datetime.now(timezone.utc) - progress_data["started_at"]).total_seconds()
        
        await self.connection_manager.broadcast_to_research(
            research_id,
            WSMessage(
                type=WSEventType.RESEARCH_COMPLETED,
                data={
                    "research_id": research_id,
                    "duration_seconds": duration,
                    "sources_analyzed": progress_data["sources_analyzed"],
                    "result_summary": result
                }
            )
        )
        
        # Clean up after delay
        await asyncio.sleep(300)  # Keep for 5 minutes
        if research_id in self.research_progress:
            del self.research_progress[research_id]
    
    async def fail_research(
        self,
        research_id: str,
        error: str,
        error_details: Optional[Dict[str, Any]] = None
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
                    "details": error_details or {}
                }
            )
        )
        
        # Clean up
        if research_id in self.research_progress:
            del self.research_progress[research_id]

# --- Research Progress Tracker (alias for compatibility) ---

class ResearchProgressTracker(ProgressTracker):
    """Alias for ProgressTracker to maintain compatibility with imports"""
    pass

# --- WebSocket Authentication ---

async def get_websocket_user(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
) -> Optional[TokenData]:
    """Authenticate WebSocket connection"""
    if not token:
        await websocket.close(code=1008, reason="Missing authentication token")
        return None
    
    try:
        # Decode token
        user_data = decode_token(token)
        return TokenData(**user_data)
    except Exception as e:
        await websocket.close(code=1008, reason="Invalid authentication token")
        return None

# --- WebSocket Endpoint ---

from fastapi import APIRouter

def create_websocket_router(
    connection_manager: ConnectionManager,
    progress_tracker: ProgressTracker
) -> APIRouter:
    """Create WebSocket router"""
    router = APIRouter()
    
    @router.websocket("/ws")
    async def websocket_endpoint(
        websocket: WebSocket,
        token: Optional[str] = Query(None)
    ):
        """WebSocket endpoint for real-time updates"""
        # Authenticate
        user_data = await get_websocket_user(websocket, token)
        if not user_data:
            return
        
        try:
            # Connect
            await connection_manager.connect(
                websocket,
                user_data.user_id,
                metadata={
                    "user_agent": websocket.headers.get("user-agent"),
                    "origin": websocket.headers.get("origin")
                }
            )
            
            # Handle messages
            while True:
                # Receive message
                data = await websocket.receive_json()
                
                # Handle message
                await connection_manager.handle_client_message(websocket, data)
                
        except WebSocketDisconnect:
            await connection_manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await connection_manager.disconnect(websocket)
            await websocket.close(code=1011, reason="Internal server error")
    
    return router

# --- Integration with Research Service ---

class WebSocketIntegration:
    """Integration helpers for WebSocket updates"""
    
    def __init__(
        self,
        connection_manager: ConnectionManager,
        progress_tracker: ProgressTracker
    ):
        self.connection_manager = connection_manager
        self.progress_tracker = progress_tracker
    
    async def notify_rate_limit_warning(
        self,
        user_id: str,
        limit_type: str,
        remaining: int,
        reset_time: datetime
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
                    "message": f"Approaching {limit_type} rate limit. {remaining} requests remaining."
                }
            )
        )
    
    async def send_system_notification(
        self,
        user_id: str,
        message: str,
        level: str = "info",
        data: Optional[Dict[str, Any]] = None
    ):
        """Send system notification to user"""
        await self.connection_manager.send_to_user(
            user_id,
            WSMessage(
                type=WSEventType.SYSTEM_NOTIFICATION,
                data={
                    "message": message,
                    "level": level,
                    "data": data or {}
                }
            )
        )

# --- Create global instances ---

connection_manager = ConnectionManager()
progress_tracker = ProgressTracker(connection_manager)
ws_integration = WebSocketIntegration(connection_manager, progress_tracker)