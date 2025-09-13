"""
Authentication and Authorization Service for Four Hosts Research API
Phase 5: Production-Ready Features
"""

import os
import bcrypt
import secrets
import json
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Final, cast
from pydantic import BaseModel, EmailStr, Field
from fastapi import HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
import logging
from enum import Enum

# Re-export canonical helpers from submodules to avoid duplication
from .auth.password import (
    hash_password,
    verify_password,
    validate_password_strength,
)
from .auth.tokens import (
    create_access_token,
    create_refresh_token,
    decode_token,
)
# Define directly to avoid circular imports
_CFG_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
_CFG_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Import database models and connection
from database.models import (
    User as DBUser,
    APIKey as DBAPIKey,
    UserRole as _DBUserRole,
    AuthProvider as DBAuthProvider,
)
from sqlalchemy import select
from sqlalchemy.sql import ColumnElement
from database.connection import get_db

# Re-export to preserve "services.auth.UserRole"
UserRole = _DBUserRole

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration (compat re-exports)
ALGORITHM = _CFG_ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = _CFG_ACCESS_TOKEN_EXPIRE_MINUTES
API_KEY_LENGTH = 32

# Security
security = HTTPBearer()

# --- Data Models ---


class AuthProvider(str, Enum):
    """Authentication providers"""

    LOCAL = "local"
    GOOGLE = "google"
    GITHUB = "github"
    SAML = "saml"


class User(BaseModel):
    """User model"""

    id: str
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    role: UserRole = UserRole.FREE
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None
    auth_provider: AuthProvider = AuthProvider.LOCAL
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UserCreate(BaseModel):
    """User creation model"""

    email: EmailStr
    username: str
    password: str
    role: UserRole = UserRole.FREE
    auth_provider: AuthProvider = AuthProvider.LOCAL


class UserLogin(BaseModel):
    """User login model"""

    email: EmailStr
    password: str


class APIKeyInfo(BaseModel):
    """API Key info model returned by service layer"""

    id: str
    key_hash: str
    user_id: str
    name: str
    role: UserRole
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True
    allowed_origins: List[str] = Field(default_factory=list)
    rate_limit_tier: str = "standard"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Token(BaseModel):
    """JWT Token model"""

    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token payload data"""

    user_id: str
    email: str
    role: UserRole
    exp: datetime
    iat: datetime
    jti: str  # JWT ID for revocation


# --- Rate Limit Configurations ---
from core.limits import API_RATE_LIMITS as RATE_LIMITS

# --- Authentication Functions (canonical helpers re-exported from submodules) ---


def generate_api_key() -> str:
    """Generate a secure API key"""
    return f"fh_{secrets.token_urlsafe(API_KEY_LENGTH)}"


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage"""
    return bcrypt.hashpw(api_key.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


# Token helpers are imported from services.auth.tokens


# --- Authorization Functions ---


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> TokenData:
    """Get current user from JWT token"""
    token = credentials.credentials
    payload = await decode_token(token)

    # Coerce/validate payload fields for type safety
    user_id_val = str(payload.get("user_id"))
    email_val = str(payload.get("email"))
    role_val = UserRole(payload.get("role"))
    exp_raw = payload.get("exp")
    iat_raw = payload.get("iat")
    jti_val = str(payload.get("jti"))

    # Validate required time fields robustly for type checker
    if exp_raw is None or iat_raw is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        exp_dt = datetime.fromtimestamp(float(cast(float, exp_raw)), tz=timezone.utc)
        iat_dt = datetime.fromtimestamp(float(cast(float, iat_raw)), tz=timezone.utc)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token timestamps",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return TokenData(
        user_id=user_id_val,
        email=email_val,
        role=role_val,
        exp=exp_dt,
        iat=iat_dt,
        jti=jti_val,
    )


async def get_api_key_info(api_key: str, db: Optional[AsyncSession] = None) -> Optional[APIKeyInfo]:
    """Validate API key and return key info"""
    if not api_key.startswith("fh_"):
        return None

    # Get database session
    if db is None:
        db_gen = get_db()
        db = await anext(db_gen)
        should_close_gen = db_gen
    else:
        should_close_gen = None

    try:
        # Query all active API keys and check each one
        from sqlalchemy import select

        result = await db.execute(select(DBAPIKey).filter(DBAPIKey.is_active == True))
        db_api_keys = result.scalars().all()

        # Check each stored hash against the provided API key
        matching_key = None
        for db_api_key in db_api_keys:
            if bcrypt.checkpw(
                api_key.encode("utf-8"), db_api_key.key_hash.encode("utf-8")
            ):
                matching_key = db_api_key
                break

        if not matching_key:
            return None

        # Check expiration (handle SQLAlchemy Column types safely)
        expires_at_val = getattr(matching_key, "expires_at", None)
        now_dt = datetime.now(timezone.utc)
        if expires_at_val is not None:
            try:
                # If it's a plain datetime, this will work; otherwise skip strict check
                if isinstance(expires_at_val, datetime) and expires_at_val < now_dt:
                    return None
            except Exception:
                # Fallback: if cannot compare, don't treat as expired
                pass

        # Update last used timestamp on ORM instance and persist
        setattr(matching_key, "last_used", now_dt)
        await db.commit()

        # Convert to response model (use plain python values)
        # Normalize rate_limit_tier to a plain string for Pydantic/APIKey model
        _tier = getattr(matching_key, "rate_limit_tier", None)
        if not isinstance(_tier, str):
            _tier = getattr(_tier, "value", None) or getattr(_tier, "name", None)
        rate_limit_tier_str = _tier or "standard"

        # Build APIKeyInfo (Pydantic model defined above)
        api_key_info = APIKeyInfo(
            id=str(getattr(matching_key, "id")),
            key_hash=str(getattr(matching_key, "key_hash")),
            user_id=str(getattr(matching_key, "user_id")),
            name=str(getattr(matching_key, "name")),
            role=UserRole(getattr(matching_key, "role")),
            created_at=getattr(matching_key, "created_at"),
            last_used=getattr(matching_key, "last_used"),
            expires_at=getattr(matching_key, "expires_at"),
            is_active=bool(getattr(matching_key, "is_active")),
            allowed_origins=getattr(matching_key, "allowed_origins") or [],
            rate_limit_tier=rate_limit_tier_str,
            metadata=getattr(matching_key, "metadata") or {},
        )

        return api_key_info

    finally:
        if should_close_gen is not None:
            await should_close_gen.aclose()


def check_permissions(required_role: UserRole, user_role: UserRole) -> bool:
    """Check if user has required role permissions"""
    role_hierarchy = {
        UserRole.FREE: 0,
        UserRole.BASIC: 1,
        UserRole.PRO: 2,
        UserRole.ENTERPRISE: 3,
        UserRole.ADMIN: 4,
    }

    return role_hierarchy.get(user_role, 0) >= role_hierarchy.get(required_role, 0)


def require_role(required_role: UserRole):
    """Dependency to require specific role"""

    async def role_checker(current_user: TokenData = Depends(get_current_user)):
        if not check_permissions(required_role, current_user.role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role}",
            )
        return current_user

    return role_checker


# --- User Management Service ---


class AuthService:
    """Authentication and authorization service"""

    def __init__(self):
        pass

    async def create_user(self, user_data: UserCreate, db: AsyncSession) -> User:
        """Create a new user"""
        # Validate password strength
        if not validate_password_strength(user_data.password):
            raise HTTPException(
                status_code=400,
                detail="Password must be at least 8 characters with uppercase, lowercase, number, and special character",
            )

        try:
            # Check if user exists
            from sqlalchemy import select

            result = await db.execute(
                select(DBUser).filter(
                    (DBUser.email == user_data.email)
                    | (DBUser.username == user_data.username)
                )
            )
            # SQLAlchemy 2.x async returns ChunkedIteratorResult; use scalars().first()
            existing_user = result.scalars().first()

            if existing_user:
                raise HTTPException(
                    status_code=409,
                    detail="User with this email or username already exists",
                )

            # Create user in database
            # Let PostgreSQL UUID default generate the ID
            # Normalize enum fields to database values (lowercase strings)
            role_value = (
                user_data.role.value
                if hasattr(user_data.role, "value")
                else str(user_data.role or UserRole.FREE.value)
            )

            auth_provider_value = (
                user_data.auth_provider.value
                if hasattr(user_data, "auth_provider") and hasattr(user_data.auth_provider, "value")
                else DBAuthProvider.LOCAL.value
            )

            db_user = DBUser(
                email=user_data.email.lower(),
                username=user_data.username,
                password_hash=await hash_password(user_data.password),
                role=role_value,
                is_active=True,
                auth_provider=auth_provider_value,
            )

            db.add(db_user)
            await db.commit()
            await db.refresh(db_user)

            # Convert to Pydantic model
            user = User(
                id=str(db_user.id),
                email=str(db_user.email),
                username=str(db_user.username),
                full_name=str(db_user.full_name) if db_user.full_name else None,
                role=UserRole(db_user.role),
                auth_provider=AuthProvider(db_user.auth_provider),
                is_active=bool(db_user.is_active),
                is_verified=bool(db_user.is_verified),
                created_at=db_user.created_at,
                last_login=db_user.last_login,
            )

            logger.info(f"Created user: {user.email}")
            return user

        except IntegrityError:
            await db.rollback()
            raise HTTPException(status_code=409, detail="User already exists")

    async def authenticate_user(
        self, login_data: UserLogin, db: Optional[AsyncSession] = None
    ) -> Optional[User]:
        """Authenticate user with email and password"""
        # Obtain DB session correctly from async generator
        if db is None:
            db_gen = get_db()  # async generator
            db = await anext(db_gen)  # first yielded session
            should_close_gen = db_gen  # remember to close later
        else:
            should_close_gen = None

        try:
            from sqlalchemy import select, func

            # Normalize email for case-insensitive lookup
            email_normalized = login_data.email.strip().lower()

            # Async query (case-insensitive email match)
            result = await db.execute(
                select(DBUser).filter(func.lower(DBUser.email) == email_normalized)
            )
            db_user = result.scalars().first()

            if not db_user:
                return None

            # Verify password
            if not await verify_password(login_data.password, str(db_user.password_hash)):
                return None

            # Check if user is active
            if not bool(db_user.is_active):
                raise HTTPException(status_code=403, detail="Account is disabled")

            # Update last login
            setattr(db_user, "last_login", datetime.now(timezone.utc))
            await db.commit()
            await db.refresh(db_user)

            # Convert to Pydantic model
            user = User(
                id=str(db_user.id),
                email=str(db_user.email),
                username=str(db_user.username),
                full_name=str(db_user.full_name) if db_user.full_name else None,
                role=UserRole(db_user.role),
                auth_provider=AuthProvider(db_user.auth_provider),
                is_active=bool(db_user.is_active),
                is_verified=bool(db_user.is_verified),
                created_at=db_user.created_at,
                last_login=db_user.last_login,
            )

            return user

        finally:
            if (
                should_close_gen is not None
            ):  # close async generator – commits / rollbacks
                await should_close_gen.aclose()

    async def create_api_key(
        self, user_id: str, key_name: str, db: Optional[AsyncSession] = None
    ) -> str:
        """Create a new API key for user"""
        # Get database session
        if db is None:
            db_gen = get_db()
            db = await anext(db_gen)
            should_close_gen = db_gen
        else:
            should_close_gen = None

        try:
            # Get user
            from sqlalchemy import select

            result = await db.execute(select(DBUser).filter(DBUser.id == user_id))
            db_user = result.scalars().first()
            if not db_user:
                raise HTTPException(status_code=404, detail="User not found")

            # Generate key
            api_key = generate_api_key()

            # Create key record in database (let DB assign UUID primary key)
            db_api_key = DBAPIKey(
                key_hash=hash_api_key(api_key),
                user_id=user_id,
                name=key_name,
                role=db_user.role,
                is_active=True,
                created_at=datetime.now(timezone.utc),
            )

            db.add(db_api_key)
            await db.commit()

            logger.info(f"Created API key '{key_name}' for user {user_id}")
            return api_key  # Return unhashed key only once

        finally:
            if should_close_gen is not None:
                await should_close_gen.aclose()

    async def revoke_token(
        self,
        jti: str,
        token_type: str = "access",
        user_id: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ):
        """Revoke a JWT token"""
        from services.token_manager import token_manager

        if not expires_at:
            expires_at = datetime.now(timezone.utc) + timedelta(days=1)

        await token_manager.add_revoked_jti(
            jti=jti,
            token_type=token_type,
            user_id=user_id or "unknown",
            expires_at=expires_at,
            reason="manual_revocation",
        )

        logger.info(f"Revoked token: {jti}")

    async def is_token_revoked(self, jti: str) -> bool:
        """Check if token is revoked"""
        from services.token_manager import token_manager

        return await token_manager.is_jti_revoked(jti)

    def get_user_rate_limits(self, role: UserRole) -> Dict[str, Any]:
        """Get rate limits for user role"""
        return RATE_LIMITS.get(role, RATE_LIMITS[UserRole.FREE])


# Create global auth service instance
auth_service = AuthService()
