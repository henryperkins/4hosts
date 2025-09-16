"""
Enhanced WebSocket Authentication and Security
Phase 5: Production-Ready Features
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Set
from collections import defaultdict
import os

from fastapi import WebSocket
from fastapi import Header
from fastapi import HTTPException
from fastapi.security.utils import get_authorization_scheme_param
import jwt

from services.auth_service import decode_token, TokenData, UserRole
from core.limits import WS_RATE_LIMITS
from core.config import is_production
from services.token_manager import token_manager

logger = logging.getLogger(__name__)

# WebSocket rate limiting configuration comes from core.limits

# Allowed origins for WebSocket connections
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
    "https://fourhosts.com",
    "https://www.fourhosts.com",
    "https://app.fourhosts.com",
]

# Dynamically extend allowed origins via env (comma-separated) to avoid
# code edits per host/IP.
_extra = [
    o.strip() for o in os.getenv(
        "ADDITIONAL_ALLOWED_ORIGINS", ""
    ).split(",") if o.strip()
]
if _extra:
    seen = set()
    merged = []
    for o in ALLOWED_ORIGINS + _extra:  # ALLOWED_ORIGINS defined earlier
        if o not in seen:
            merged.append(o)
            seen.add(o)
    ALLOWED_ORIGINS = merged


class WebSocketRateLimiter:
    """Rate limiter for WebSocket connections"""

    def __init__(self):
        # Track connections per user
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)
        # Track messages per user (timestamp of messages)
        self.user_messages: Dict[str, list] = defaultdict(list)
        # Track subscriptions per connection
        self.connection_subscriptions: Dict[str, int] = defaultdict(int)

    async def check_connection_limit(
        self, user_id: str, connection_id: str, role: UserRole
    ) -> bool:
        """Check if user can create new WebSocket connection"""
        limits = WS_RATE_LIMITS.get(role, WS_RATE_LIMITS[UserRole.FREE])
        max_connections = limits["connections_per_user"]

        current_connections = len(self.user_connections[user_id])
        if current_connections >= max_connections:
            logger.warning(
                "User %s exceeded max WebSocket connections: %s/%s",
                user_id,
                current_connections,
                max_connections,
            )
            return False

        self.user_connections[user_id].add(connection_id)
        return True

    async def check_message_rate(self, user_id: str, role: UserRole) -> bool:
        """Check if user can send a message"""
        limits = WS_RATE_LIMITS.get(role, WS_RATE_LIMITS[UserRole.FREE])
        max_messages = limits["messages_per_minute"]

        # Clean old messages
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=1)
        self.user_messages[user_id] = [
            ts for ts in self.user_messages[user_id] if ts > cutoff
        ]

        # Check limit
        if len(self.user_messages[user_id]) >= max_messages:
            logger.warning(f"User {user_id} exceeded message rate limit")
            return False

        self.user_messages[user_id].append(now)
        return True

    async def check_subscription_limit(
        self, connection_id: str, role: UserRole
    ) -> bool:
        """Check if connection can add more subscriptions"""
        limits = WS_RATE_LIMITS.get(role, WS_RATE_LIMITS[UserRole.FREE])
        max_subscriptions = limits["subscriptions_per_connection"]

        current_subscriptions = self.connection_subscriptions[connection_id]
        if current_subscriptions >= max_subscriptions:
            logger.warning(
                "Connection %s exceeded subscription limit", connection_id
            )
            return False

        self.connection_subscriptions[connection_id] += 1
        return True

    async def remove_connection(self, user_id: str, connection_id: str):
        """Remove connection from tracking"""
        self.user_connections[user_id].discard(connection_id)
        if not self.user_connections[user_id]:
            del self.user_connections[user_id]

        if connection_id in self.connection_subscriptions:
            del self.connection_subscriptions[connection_id]


# Helper to safely obtain limits regardless of enum origin mismatch
def _get_limits(role):  # type: ignore[no-untyped-def]
    try:
        return WS_RATE_LIMITS[role]  # type: ignore[index]
    except Exception:  # noqa: BLE001
        name = getattr(role, "name", None)
        if name and name in WS_RATE_LIMITS:  # type: ignore[operator]
            return WS_RATE_LIMITS[name]  # type: ignore[index]
        # Fallback: first value
        return list(WS_RATE_LIMITS.values())[0]


# Global rate limiter instance
ws_rate_limiter = WebSocketRateLimiter()


async def authenticate_websocket(
    websocket: WebSocket,
    authorization: Optional[str] = Header(None),
    sec_websocket_protocol: Optional[str] = Header(None),
    origin: Optional[str] = Header(None),
) -> Optional[TokenData]:
    """
    Authenticate WebSocket connection with enhanced security

    1. Check Origin header against allowed origins
    2. Extract token from Authorization header or Sec-WebSocket-Protocol
    3. Validate token and check revocation
    """

    # 1. Verify Origin (relaxed in non-production)
    if origin:
        if is_production():
            allowed = (
                origin in ALLOWED_ORIGINS
                or origin.startswith("http://localhost")
                or origin.startswith("https://localhost")
                or origin.startswith("http://127.0.0.1")
                or origin.startswith("https://127.0.0.1")
            )
            if not allowed:
                logger.warning(
                    "WebSocket connection rejected from unauthorized origin: %s",
                    origin,
                )
                await websocket.close(code=1008, reason="Unauthorized origin")
                return None
        else:
            # In dev, accept any localhost or container-based origins
            pass

    # 2. Extract token (priority order)
    token: Optional[str] = None

    # Try Authorization header first
    if authorization:
        scheme, credentials = get_authorization_scheme_param(authorization)
        if scheme.lower() == "bearer":
            token = credentials

    # Try Sec-WebSocket-Protocol header (for browsers that don't support
    # custom headers)
    if not token and sec_websocket_protocol:
        # Format: "access_token, <actual_token>"
        protocols = sec_websocket_protocol.split(",")
        for protocol in protocols:
            protocol = protocol.strip()
            if protocol.startswith("access_token."):
                token = protocol[13:]  # Remove "access_token." prefix
                break

    # Fallback: try to read JWT from cookies (set by /auth/login)
    if not token:
        try:
            cookie_header = (
                websocket.headers.get("cookie")
                or websocket.headers.get("Cookie")
            )
            if cookie_header:
                # Minimal cookie parsing to avoid extra deps
                cookies = {}
                for part in cookie_header.split(";"):
                    if not part:
                        continue
                    if "=" in part:
                        k, v = part.split("=", 1)
                        cookies[k.strip()] = v.strip()
                token = cookies.get("access_token")
        except Exception:
            token = None

    if not token:
        await websocket.close(code=1008, reason="Missing authentication token")
        return None

    try:
        # Decode and validate token
        payload = await decode_token(token)

        # Token payload validation (avoid None fields)
        required = ["user_id", "email", "exp", "iat", "jti"]
        if not all(k in payload and payload[k] for k in required):  # type: ignore[index]
            raise HTTPException(status_code=400, detail="Invalid token payload")
        user_data = TokenData(
            user_id=str(payload.get("user_id")),
            email=str(payload.get("email")),
            role=payload.get("role", UserRole.FREE),  # type: ignore[index]
            exp=datetime.fromtimestamp(
                float(payload.get("exp")), tz=timezone.utc
            ),
            iat=datetime.fromtimestamp(
                float(payload.get("iat")), tz=timezone.utc
            ),
            jti=str(payload.get("jti")),
        )

        return user_data

    except HTTPException as e:
        await websocket.close(code=1008, reason=e.detail)
        return None
    except Exception as e:
        logger.error(f"WebSocket authentication error: {e}")
        await websocket.close(code=1008, reason="Authentication failed")
        return None


async def verify_websocket_rate_limit(
    user_id: str, connection_id: str, role: UserRole, websocket: WebSocket
) -> bool:
    """Verify WebSocket connection is within rate limits"""
    limits = _get_limits(role)
    if not await ws_rate_limiter.check_connection_limit(
        user_id, connection_id, role
    ):
        max_conn = limits.get("connections_per_user")
        await websocket.close(
            code=4001,
            reason=(
                "Connection limit exceeded. Maximum "
                f"{max_conn} connections allowed."
            ),
        )
        return False
    return True


async def check_websocket_message_rate(user_id: str, role: UserRole) -> bool:
    """Check if user can send a WebSocket message"""
    return await ws_rate_limiter.check_message_rate(user_id, role)


async def check_websocket_subscription_limit(
    connection_id: str, role: UserRole
) -> bool:
    """Check if connection can add more subscriptions"""
    limits = _get_limits(role)
    if not await ws_rate_limiter.check_subscription_limit(
        user_id, connection_id, role
    ):
        logger.warning(
            "Subscription limit exceeded for user=%s conn=%s", user_id, connection_id
        )
        return False
    return True


async def cleanup_websocket_connection(user_id: str, connection_id: str):
    """Clean up WebSocket connection resources"""
    await ws_rate_limiter.remove_connection(user_id, connection_id)


# Enhanced WebSocket endpoint with better security
async def secure_websocket_endpoint(
    websocket: WebSocket,
    authorization: Optional[str] = Header(None),
    sec_websocket_protocol: Optional[str] = Header(None),
    origin: Optional[str] = Header(None),
    user_agent: Optional[str] = Header(None),
    x_real_ip: Optional[str] = Header(None),
    x_forwarded_for: Optional[str] = Header(None),
):
    """
    Secure WebSocket endpoint with enhanced authentication and rate limiting
    """
    # Get client IP
    client_ip = (
        x_real_ip
        or (x_forwarded_for.split(",")[0] if x_forwarded_for else None)
        or "unknown"
    )

    # Authenticate
    user_data = await authenticate_websocket(
        websocket,
        authorization=authorization,
        sec_websocket_protocol=sec_websocket_protocol,
        origin=origin,
    )

    if not user_data:
        return

    connection_id = (
        f"ws_{user_data.user_id}_{datetime.now(timezone.utc).timestamp()}"
    )

    # Check rate limits
    if not await verify_websocket_rate_limit(
        user_data.user_id, connection_id, user_data.role, websocket
    ):
        return

    try:
        # Accept connection with subprotocol if requested
        if sec_websocket_protocol and "access_token" in sec_websocket_protocol:
            await websocket.accept(subprotocol="access_token")
        else:
            await websocket.accept()

        logger.info(
            "WebSocket connected: user=%s, ip=%s, origin=%s",
            user_data.user_id,
            client_ip,
            origin,
        )

        # Return connection info for further processing
        return {
            "websocket": websocket,
            "user_data": user_data,
            "connection_id": connection_id,
            "client_ip": client_ip,
            "origin": origin,
            "user_agent": user_agent,
        }

    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        await cleanup_websocket_connection(user_data.user_id, connection_id)
        raise
