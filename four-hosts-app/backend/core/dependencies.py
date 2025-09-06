"""
Common dependencies for the Four Hosts Research API
"""

import uuid
from typing import Optional, Any
from types import SimpleNamespace

from fastapi import Request, HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from services.auth_service import get_current_user as auth_get_current_user
from models.base import UserRole


# HTTPBearer security scheme
security = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> Any:
    """
    Resolve the current authenticated user.

    Token lookup order:
    1. access_token cookie
    2. Authorization header via HTTPBearer
    3. Manual Authorization header inspection
    """
    token: Optional[str] = request.cookies.get("access_token")

    # Token provided via HTTPBearer dependency
    if not token and credentials and credentials.credentials:
        token = credentials.credentials

    # Manual inspection of Authorization header
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.lower().startswith("bearer "):
            token = auth_header[7:]

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication token"
        )

    # Use auth service's validator
    bearer_credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials=token
    )
    token_data = await auth_get_current_user(bearer_credentials)

    # Map to lightweight user object for backward compatibility
    user = SimpleNamespace(
        id=token_data.user_id,
        user_id=token_data.user_id,
        email=token_data.email,
        role=(
            UserRole(token_data.role)
            if isinstance(token_data.role, str)
            else token_data.role
        ),
    )
    return user


async def get_current_user_optional(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> Optional[Any]:
    """Get current user if authenticated, None otherwise"""
    try:
        return await get_current_user(request, credentials)
    except HTTPException:
        return None


def require_role(required_role: UserRole):
    """Dependency to require specific role"""
    async def role_checker(current_user: Any = Depends(get_current_user)):
        if current_user.role not in [
            UserRole.PRO, UserRole.ENTERPRISE, UserRole.ADMIN
        ]:
            if required_role in [
                UserRole.PRO, UserRole.ENTERPRISE, UserRole.ADMIN
            ]:
                raise HTTPException(
                    status_code=403,
                    detail=f"{required_role.value} subscription required"
                )
        return current_user
    return role_checker


def get_request_id() -> str:
    """Generate a unique request ID"""
    return str(uuid.uuid4())
