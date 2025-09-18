"""
JWT token utilities for authentication
"""

import os
import secrets
import jwt
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Final

from utils.async_utils import run_in_thread

# Configuration
SECRET_KEY_ENV = os.getenv("JWT_SECRET_KEY")
if SECRET_KEY_ENV is None or SECRET_KEY_ENV.strip() == "":
    raise ValueError(
        "JWT_SECRET_KEY environment variable must be set. "
        "Generate one with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
    )
SECRET_KEY: Final[str] = SECRET_KEY_ENV

# Avoid leaking secrets in logs. Provide an optional, safe debug path that
# logs only a short hash and length when explicitly enabled.
if os.getenv("DEBUG_SECURE_LOGS", "0") == "1":
    try:
        import hashlib
        import structlog  # preferred logger per repository guidelines

        logger = structlog.get_logger(__name__)
        digest = hashlib.sha256(SECRET_KEY.encode()).hexdigest()[:8]
        logger.debug(
            "JWT secret loaded",
            sha256_prefix=digest,
            length=len(SECRET_KEY),
        )
    except Exception:
        # As a last resort, never print the secret; fail silently if logging stack isn't ready
        pass

# Define constants directly to avoid circular imports
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))


def create_access_token(
    data: Dict[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)

    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    # Encode temporal claims as integer epoch seconds for robust decoding
    to_encode.update(
        {
            "exp": int(expire.timestamp()),
            "iat": int(now.timestamp()),
            "jti": secrets.token_urlsafe(16),  # JWT ID for revocation
        }
    )

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def create_refresh_token(
    user_id: str,
    device_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> str:
    """Create a refresh token using TokenManager"""
    from services.token_manager import token_manager

    result = await token_manager.create_refresh_token(
        user_id=user_id,
        device_id=device_id,
        ip_address=ip_address,
        user_agent=user_agent,
    )

    return result["token"]


async def decode_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT token"""
    try:
        payload = await run_in_thread(
            jwt.decode, token, SECRET_KEY, algorithms=[ALGORITHM]
        )

        # Check if JTI is revoked
        jti = payload.get("jti")
        if jti:
            from services.token_manager import token_manager

            if await token_manager.is_jti_revoked(jti):
                from fastapi import HTTPException, status
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        return payload
    except jwt.ExpiredSignatureError:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError as e:
        import logging
        logging.getLogger(__name__).warning("JWT validation error: %s", e)
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
