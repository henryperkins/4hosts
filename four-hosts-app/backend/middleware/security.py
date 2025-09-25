"""
Security middleware for the Four Hosts Research API
"""

import secrets
import structlog
import time

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse

from core.config import is_production

logger = structlog.get_logger(__name__)


async def csrf_protection_middleware(request: Request, call_next):
    """CSRF protection middleware"""
    # CSRF exempt routes
    csrf_exempt_routes = [
        "/api/csrf-token",
        "/api/session/create",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/health",
        "/ws",
        "/api/health",
        # Unversioned (legacy) paths
        "/auth/refresh",
        "/auth/user",
        "/research/history",
        "/research/status",
        "/system/public-stats",
        # Versioned API paths (current)
        "/v1/auth/refresh",
        "/v1/auth/user",
        "/v1/research/history",
        "/v1/research/status",
        "/v1/system/public-stats",
        # Authentication endpoints should be exempt because the client may
        # not yet possess a CSRF cookie when attempting to log in or
        # register a brand-new account. Protecting these routes with a
        # per-request token therefore blocks legitimate first-factor
        # authentication attempts.
        "/auth/login",
        "/auth/register",
        "/v1/auth/login",
        "/v1/auth/register",
    ]

    # Skip CSRF check for safe methods and exempt routes
    if (request.method in ["POST", "PUT", "DELETE", "PATCH"] and
        not any(request.url.path.startswith(route)
                for route in csrf_exempt_routes)):

        csrf_token_from_cookie = request.cookies.get("csrf_token")
        csrf_token_from_header = request.headers.get("X-CSRF-Token")

        logger.debug("CSRF check", path=request.url.path,
                    cookie_token_present=bool(csrf_token_from_cookie),
                    header_token_present=bool(csrf_token_from_header))

        if (not csrf_token_from_cookie or
            not csrf_token_from_header or
            csrf_token_from_cookie != csrf_token_from_header):

            logger.warning("CSRF token mismatch",
                         path=request.url.path,
                         has_cookie=bool(csrf_token_from_cookie),
                         has_header=bool(csrf_token_from_header),
                         tokens_match=(csrf_token_from_cookie == csrf_token_from_header
                                      if csrf_token_from_cookie and csrf_token_from_header
                                      else False))
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
    """Generate or return existing CSRF token with enhanced error handling"""

    try:
        logger.info("CSRF token endpoint accessed",
                   client=request.client.host if request.client else "unknown",
                   user_agent=request.headers.get("user-agent", "unknown"))

        # Check if a valid CSRF token already exists
        existing_token = request.cookies.get("csrf_token")

        if existing_token:
            logger.debug("Returning existing CSRF token")
            return {"csrf_token": existing_token}

        # Generate a new token with retry logic for entropy
        max_retries = 3
        for attempt in range(max_retries):
            try:
                token = secrets.token_urlsafe(16)
                logger.debug("Generated new CSRF token", attempt=attempt + 1)

                # Determine cookie attributes from actual scheme to avoid setting
                # Secure cookies over HTTP during local development or behind proxies.
                forwarded_proto = request.headers.get("x-forwarded-proto", "").lower()
                is_https = forwarded_proto == "https" or request.url.scheme == "https"
                same_site = "none" if is_https else "lax"
                secure_flag = True if is_https else False

                # Set cookie with proper security flags
                response.set_cookie(
                    key="csrf_token",
                    value=token,
                    httponly=False,  # Allow JavaScript access for CSRF tokens
                    secure=secure_flag,
                    samesite=same_site,
                    path="/",
                    max_age=3600,  # 1 hour expiry
                )

                logger.info("CSRF token generated successfully",
                           token_length=len(token),
                           is_https=is_https,
                           same_site=same_site)

                return {"csrf_token": token}

            except Exception as cookie_error:
                logger.warning("Cookie setting failed",
                              attempt=attempt + 1,
                              error=str(cookie_error))
                if attempt == max_retries - 1:
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to set CSRF cookie after multiple attempts"
                    )
                time.sleep(0.1)  # Small delay before retry

        # This should never be reached, but just in case
        raise HTTPException(status_code=500, detail="CSRF token generation failed")

    except Exception as e:
        logger.error("CSRF token generation failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    client=request.client.host if request.client else "unknown")
        raise HTTPException(status_code=500, detail="CSRF token unavailable") from e
