"""
Authentication routes for the Four Hosts Research API
"""

import os
import uuid
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, Response, Request, Depends
from fastapi.security import HTTPAuthorizationCredentials

from models.auth import (
    UserCreate,
    UserLogin,
    Token,
    RefreshTokenRequest,
    LogoutRequest,
    PreferencesPayload
)
from services.auth import (
    auth_service as real_auth_service,
    create_access_token,
    create_refresh_token,
    get_current_user as auth_get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from services.token_manager import token_manager
from services.user_management import user_profile_service
from database.connection import get_db_context, get_db
from database.models import User as DBUser
from core.dependencies import get_current_user, get_current_user_optional
from core.config import is_production
from middleware.security import get_csrf_token
from sqlalchemy import select

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/auth", tags=["authentication"])


@router.get("/csrf-token")
def csrf_token_endpoint(request: Request, response: Response):
    """Get CSRF token for authentication"""
    return get_csrf_token(request, response)


@router.post("/register", response_model=Token)
async def register(user_data: UserCreate):
    """Register a new user"""
    # Convert to auth module's UserCreate model
    from services.auth import UserCreate as AuthUserCreate

    auth_user_data = AuthUserCreate(
        email=user_data.email,
        username=user_data.username,
        password=user_data.password,
        role=user_data.role,
    )

    # Use async context manager for database session
    async with get_db_context() as db:
        user = await real_auth_service.create_user(auth_user_data, db)

    access_token = create_access_token(
        {"user_id": str(user.id), "email": user.email, "role": user.role.value}
    )

    refresh_token = await create_refresh_token(
        user_id=str(user.id), device_id=None, ip_address=None, user_agent=None
    )

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/login")
async def login(login_data: UserLogin, response: Response):
    """Login with email and password"""
    # Convert to auth module's UserLogin model
    from services.auth import UserLogin as AuthUserLogin

    auth_login_data = AuthUserLogin(
        email=login_data.email, password=login_data.password
    )

    user = await real_auth_service.authenticate_user(auth_login_data)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token(
        {"user_id": str(user.id), "email": user.email, "role": user.role.value}
    )

    refresh_token = await create_refresh_token(
        user_id=str(user.id), device_id=None, ip_address=None, user_agent=None
    )

    # Set cookie attributes for cross-site compatibility
    production = is_production()
    same_site = "none" if production else "lax"
    secure_flag = True if production else False

    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=secure_flag,
        samesite=same_site,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=secure_flag,
        samesite=same_site,
        max_age=60 * 60 * 24 * 7,  # 7 days
        path="/",
    )

    # Return user data
    return {
        "success": True,
        "user": {
            "id": str(user.id),
            "email": user.email,
            "name": user.full_name or user.username,
            "role": user.role.value,
            "created_at": (
                user.created_at.isoformat()
                if user.created_at else datetime.utcnow().isoformat()
            ),
            "is_active": user.is_active
        },
        "message": "Login successful"
    }


@router.post("/refresh")
async def refresh_token(request: Request, response: Response):
    """Refresh access token using secure token rotation"""
    refresh_token = request.cookies.get("refresh_token")

    # Validate the token first to get user info
    token_info = await token_manager.validate_refresh_token(refresh_token)
    if not token_info:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    # Then rotate the refresh token
    token_result = await token_manager.rotate_refresh_token(refresh_token)
    if not token_result:
        raise HTTPException(
            status_code=401, detail="Invalid or expired refresh token"
        )

    # Get user from database
    db_gen = get_db()
    db = await anext(db_gen)
    try:
        result = await db.execute(
            select(DBUser).filter(DBUser.id == token_info["user_id"])
        )
        user = result.scalars().first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Create new access token
        access_token = create_access_token(
            {"user_id": str(user.id), "email": user.email, "role": user.role.value}
        )

        # Set cookie attributes for cross-site compatibility
        production = is_production()
        same_site = "none" if production else "lax"
        secure_flag = True if production else False

        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            secure=secure_flag,
            samesite=same_site,
            max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )
        response.set_cookie(
            key="refresh_token",
            value=token_result["token"],
            httponly=True,
            secure=secure_flag,
            samesite=same_site,
            max_age=60 * 60 * 24 * 7,  # 7 days
            path="/",
        )
        return {"message": "Token refreshed"}
    finally:
        await db_gen.aclose()


@router.get("/user")
async def get_current_user_info(current_user=Depends(get_current_user)):
    """Get current user information"""
    # Fetch canonical username and optional profile details
    username_value = None
    full_name = None

    try:
        # Prefer user_profile_service if available
        profile = await user_profile_service.get_user_profile(
            uuid.UUID(str(current_user.user_id))
        )
        if profile:
            username_value = profile.get("username") or None
            full_name = profile.get("full_name")

        # Fallback to DB username if not in profile
        if not username_value:
            db_gen = get_db()
            db = await anext(db_gen)
            try:
                result = await db.execute(
                    select(DBUser).filter(DBUser.id == current_user.user_id)
                )
                db_user = result.scalars().first()
                if db_user:
                    username_value = getattr(db_user, "username", None)
            finally:
                await db_gen.aclose()
    except Exception:
        # Silent fallback to synthesized username for resilience
        pass

    # Final fallback if still missing
    if not username_value:
        username_value = current_user.email.split("@")[0]

    # Get user from database to get created_at and is_active
    db_gen = get_db()
    db = await anext(db_gen)
    try:
        result = await db.execute(
            select(DBUser).filter(DBUser.id == current_user.user_id)
        )
        db_user = result.scalars().first()
        created_at = (
            db_user.created_at.isoformat()
            if db_user and db_user.created_at
            else datetime.utcnow().isoformat()
        )
        is_active = db_user.is_active if db_user else True
    finally:
        await db_gen.aclose()

    # Return format matching frontend expectations
    return {
        "id": str(current_user.user_id),
        "email": current_user.email,
        "name": full_name or username_value,
        "role": str(
            current_user.role.value
            if hasattr(current_user, "role")
            else current_user.role
        ),
        "created_at": created_at,
        "is_active": is_active
    }


@router.post("/logout")
async def logout(
    response: Response,
    request: Request,
    logout_data: LogoutRequest = None,
    current_user=Depends(get_current_user_optional),
):
    """Logout user by revoking tokens and clearing cookies"""
    # Handle case where user is already logged out
    if not current_user:
        return {"message": "Successfully logged out"}

    refresh_token = (
        logout_data.refresh_token if logout_data else None
    )

    # Revoke the access token using its JTI
    if current_user and hasattr(current_user, 'jti'):
        await real_auth_service.revoke_token(
            jti=current_user.jti,
            token_type="access",
            user_id=current_user.user_id,
            expires_at=current_user.exp,
        )

    # Revoke the refresh token if provided
    if refresh_token:
        # Revoke all tokens in the refresh token family
        token_info = await token_manager.validate_refresh_token(refresh_token)
        if token_info and "family_id" in token_info:
            await token_manager.revoke_token_family(
                family_id=token_info["family_id"], reason="user_logout"
            )

    response.delete_cookie("access_token", path="/")
    response.delete_cookie("refresh_token", path="/")
    response.delete_cookie("csrf_token", path="/")
    return {"message": "Successfully logged out"}


@router.put("/preferences")
async def update_user_preferences(
    payload: PreferencesPayload, current_user=Depends(get_current_user)
):
    """Update the current user's preferences"""
    success = await user_profile_service.update_user_preferences(
        uuid.UUID(str(current_user.user_id)), payload.preferences
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update preferences")

    # Return updated profile
    profile = await user_profile_service.get_user_profile(
        uuid.UUID(str(current_user.user_id))
    )
    return profile


@router.get("/preferences")
async def get_user_preferences(current_user=Depends(get_current_user)):
    """Retrieve the current user's preferences"""
    profile = await user_profile_service.get_user_profile(
        uuid.UUID(str(current_user.user_id))
    )
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")
    return {"preferences": profile.get("preferences", {})}
