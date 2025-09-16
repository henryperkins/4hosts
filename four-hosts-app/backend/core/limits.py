"""
Centralized rate-limit definitions for API and WebSocket channels.
"""

from typing import Dict, Any
from models import UserRole


# API rate limits by role
API_RATE_LIMITS: Dict[UserRole, Dict[str, Any]] = {
    UserRole.FREE: {
        "requests_per_minute": 10,
        "requests_per_hour": 100,
        "requests_per_day": 500,
        "concurrent_requests": 1,
        "max_query_length": 200,
        "max_sources": 50,
    },
    UserRole.BASIC: {
        "requests_per_minute": 30,
        "requests_per_hour": 500,
        "requests_per_day": 5000,
        "concurrent_requests": 3,
        "max_query_length": 500,
        "max_sources": 100,
    },
    UserRole.PRO: {
        "requests_per_minute": 60,
        "requests_per_hour": 1000,
        "requests_per_day": 20000,
        "concurrent_requests": 10,
        "max_query_length": 1000,
        "max_sources": 200,
    },
    UserRole.ENTERPRISE: {
        "requests_per_minute": 200,
        "requests_per_hour": 10000,
        "requests_per_day": 100000,
        "concurrent_requests": 50,
        "max_query_length": 2000,
        "max_sources": 500,
    },
    UserRole.ADMIN: {
        "requests_per_minute": 1000,
        "requests_per_hour": 100000,
        "requests_per_day": 1000000,
        "concurrent_requests": 100,
        "max_query_length": 5000,
        "max_sources": 1000,
    },
}


# WebSocket rate limits by role
WS_RATE_LIMITS: Dict[UserRole, Dict[str, int]] = {
    UserRole.FREE: {
        "connections_per_user": 2,
        "messages_per_minute": 10,
        "subscriptions_per_connection": 5,
    },
    UserRole.BASIC: {
        "connections_per_user": 3,
        "messages_per_minute": 30,
        "subscriptions_per_connection": 10,
    },
    UserRole.PRO: {
        "connections_per_user": 5,
        "messages_per_minute": 60,
        "subscriptions_per_connection": 20,
    },
    UserRole.ENTERPRISE: {
        "connections_per_user": 10,
        "messages_per_minute": 200,
        "subscriptions_per_connection": 50,
    },
    UserRole.ADMIN: {
        "connections_per_user": 100,
        "messages_per_minute": 1000,
        "subscriptions_per_connection": 100,
    },
}


def get_api_limits(role: UserRole) -> Dict[str, Any]:
    return API_RATE_LIMITS.get(role, API_RATE_LIMITS[UserRole.FREE])


def get_ws_limits(role: UserRole) -> Dict[str, int]:
    return WS_RATE_LIMITS.get(role, WS_RATE_LIMITS[UserRole.FREE])

