"""
Security middleware for the Four Hosts Research API
"""

import secrets
import logging

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from core.config import is_production

logger = logging.getLogger(__name__)


async def csrf_protection_middleware(request: Request, call_next):
    """CSRF protection middleware"""
    # CSRF exempt routes
    csrf_exempt_routes = [
        "/api/csrf-token",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/health",
        "/ws",
        "/api/health",
        "/auth/refresh",
        "/auth/user",
        "/research/history",
        "/research/status",
        "/system/public-stats",
    ]

    # Skip CSRF check for safe methods and exempt routes
    if (request.method in ["POST", "PUT", "DELETE", "PATCH"] and
        not any(request.url.path.startswith(route)
                for route in csrf_exempt_routes)):

        csrf_token_from_cookie = request.cookies.get("csrf_token")
        csrf_token_from_header = request.headers.get("X-CSRF-Token")

        logger.debug(f"CSRF check for {request.url.path}")
        logger.debug(f"Cookie token: {csrf_token_from_cookie}")
        logger.debug(f"Header token: {csrf_token_from_header}")

        if (not csrf_token_from_cookie or
            not csrf_token_from_header or
            csrf_token_from_cookie != csrf_token_from_header):

            logger.warning(
                f"CSRF token mismatch on {request.url.path}: "
                f"cookie={csrf_token_from_cookie}, "
                f"header={csrf_token_from_header}"
            )
            return JSONResponse(
                status_code=403,
                content={
                    "error": "HTTP Error",
                    "detail": "CSRF token mismatch",
                    "status_code": 403,
                    "request_id": getattr(
                        request.state, "request_id", "unknown"
                    ),
                },
            )

    response = await call_next(request)
    return response


async def security_middleware(request: Request, call_next):
    """Block malicious requests early"""
    path = request.url.path.lower()

    # Block PHP file requests - return 404 to reduce info leakage
    if path.endswith(('.php', '.asp', '.jsp')):
        return Response(status_code=404, content="Not Found")

    # Block common attack patterns - return 404 for admin paths
    attack_patterns = ['/admin', '/wp-admin', '/.env', '/config']
    if any(pattern in path for pattern in attack_patterns):
        return Response(status_code=404, content="Not Found")

    return await call_next(request)


def get_csrf_token(request: Request, response: Response) -> dict:
    """Generate or return existing CSRF token"""
    logger.info(f"CSRF token endpoint accessed from {request.client.host}")

    # Check if a valid CSRF token already exists
    existing_token = request.cookies.get("csrf_token")

    if existing_token:
        logger.debug(f"Returning existing CSRF token: {existing_token}")
        return {"csrf_token": existing_token}

    # Generate a new token only if none exists
    token = secrets.token_urlsafe(16)
    logger.debug(f"Generating new CSRF token: {token}")

    # Only use secure cookies in production
    production = is_production()
    same_site = "none" if production else "lax"
    secure_flag = True if production else False

    response.set_cookie(
        key="csrf_token",
        value=token,
        httponly=False,
        secure=secure_flag,
        samesite=same_site,
        path="/",
    )
    return {"csrf_token": token}
