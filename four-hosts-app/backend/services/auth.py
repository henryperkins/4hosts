"""
Authentication and Authorization Service for Four Hosts Research API
Phase 5: Production-Ready Features
"""

import os
import jwt
import bcrypt
import secrets
import json
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, EmailStr, Field
from fastapi import HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
import logging
from enum import Enum
import re

# Import database models and connection
from database.models import User as DBUser, APIKey as DBAPIKey, UserRole as _DBUserRole
from database.connection import get_db

# Re-export to preserve "services.auth.UserRole"
UserRole = _DBUserRole

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY environment variable must be set. Generate one with: python -c 'import secrets; print(secrets.token_urlsafe(32))'")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7
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

class APIKey(BaseModel):
    """API Key model"""
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

RATE_LIMITS = {
    UserRole.FREE: {
        "requests_per_minute": 10,
        "requests_per_hour": 100,
        "requests_per_day": 500,
        "concurrent_requests": 1,
        "max_query_length": 200,
        "max_sources": 50
    },
    UserRole.BASIC: {
        "requests_per_minute": 30,
        "requests_per_hour": 500,
        "requests_per_day": 5000,
        "concurrent_requests": 3,
        "max_query_length": 500,
        "max_sources": 100
    },
    UserRole.PRO: {
        "requests_per_minute": 60,
        "requests_per_hour": 1000,
        "requests_per_day": 20000,
        "concurrent_requests": 10,
        "max_query_length": 1000,
        "max_sources": 200
    },
    UserRole.ENTERPRISE: {
        "requests_per_minute": 200,
        "requests_per_hour": 10000,
        "requests_per_day": 100000,
        "concurrent_requests": 50,
        "max_query_length": 2000,
        "max_sources": 500
    },
    UserRole.ADMIN: {
        "requests_per_minute": 1000,
        "requests_per_hour": 100000,
        "requests_per_day": 1000000,
        "concurrent_requests": 100,
        "max_query_length": 5000,
        "max_sources": 1000
    }
}

# --- Authentication Functions ---

def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def validate_password_strength(password: str) -> bool:
    """Validate password meets security requirements"""
    if len(password) < 8:
        return False
    if not re.search(r'[A-Z]', password):
        return False
    if not re.search(r'[a-z]', password):
        return False
    if not re.search(r'[0-9]', password):
        return False
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False
    return True

def generate_api_key() -> str:
    """Generate a secure API key"""
    return f"fh_{secrets.token_urlsafe(API_KEY_LENGTH)}"

def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage"""
    return bcrypt.hashpw(api_key.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)

    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({
        "exp": expire,
        "iat": now,
        "jti": secrets.token_urlsafe(16)  # JWT ID for revocation
    })

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def create_refresh_token(
    user_id: str,
    device_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None
) -> str:
    """Create a refresh token using TokenManager"""
    from services.token_manager import token_manager

    result = await token_manager.create_refresh_token(
        user_id=user_id,
        device_id=device_id,
        ip_address=ip_address,
        user_agent=user_agent
    )

    return result["token"]

async def decode_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        # Check if JTI is revoked
        jti = payload.get("jti")
        if jti:
            from services.token_manager import token_manager
            if await token_manager.is_jti_revoked(jti):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# --- Authorization Functions ---

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> TokenData:
    """Get current user from JWT token"""
    token = credentials.credentials
    payload = await decode_token(token)

    return TokenData(
        user_id=payload.get("user_id"),
        email=payload.get("email"),
        role=UserRole(payload.get("role")),
        exp=datetime.fromtimestamp(payload.get("exp"), tz=timezone.utc),
        iat=datetime.fromtimestamp(payload.get("iat"), tz=timezone.utc),
        jti=payload.get("jti")
    )

async def get_api_key_info(api_key: str, db: AsyncSession = None) -> Optional[APIKey]:
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
        result = await db.execute(
            select(DBAPIKey).filter(
                DBAPIKey.is_active == True
            )
        )
        db_api_keys = result.scalars().all()

        # Check each stored hash against the provided API key
        matching_key = None
        for db_api_key in db_api_keys:
            if bcrypt.checkpw(api_key.encode('utf-8'), db_api_key.key_hash.encode('utf-8')):
                matching_key = db_api_key
                break

        if not matching_key:
            return None

        # Check expiration
        if matching_key.expires_at and matching_key.expires_at < datetime.now(timezone.utc):
            return None

        # Update last used timestamp
        matching_key.last_used = datetime.now(timezone.utc)
        await db.commit()

        # Convert to Pydantic model
        api_key_info = APIKey(
            id=matching_key.id,
            key_hash=matching_key.key_hash,
            user_id=matching_key.user_id,
            name=matching_key.name,
            role=UserRole(matching_key.role),
            created_at=matching_key.created_at,
            last_used=matching_key.last_used,
            expires_at=matching_key.expires_at,
            is_active=matching_key.is_active,
            allowed_origins=matching_key.allowed_origins or [],
            rate_limit_tier=matching_key.rate_limit_tier or "standard",
            metadata=matching_key.metadata or {}
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
        UserRole.ADMIN: 4
    }

    return role_hierarchy.get(user_role, 0) >= role_hierarchy.get(required_role, 0)

def require_role(required_role: UserRole):
    """Dependency to require specific role"""
    async def role_checker(current_user: TokenData = Depends(get_current_user)):
        if not check_permissions(required_role, current_user.role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role}"
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
                detail="Password must be at least 8 characters with uppercase, lowercase, number, and special character"
            )

        try:
            # Check if user exists
            from sqlalchemy import select
            result = await db.execute(
                select(DBUser).filter(
                    (DBUser.email == user_data.email) |
                    (DBUser.username == user_data.username)
                )
            )
            # SQLAlchemy 2.x async returns ChunkedIteratorResult; use scalars().first()
            existing_user = result.scalars().first()

            if existing_user:
                raise HTTPException(
                    status_code=409,
                    detail="User with this email or username already exists"
                )

            # Create user in database
            # Let PostgreSQL UUID default generate the ID
            db_user = DBUser(
                email=user_data.email,
                username=user_data.username,
                password_hash=hash_password(user_data.password),
                role=user_data.role or UserRole.FREE,
                is_active=True,
                auth_provider=user_data.auth_provider or AuthProvider.LOCAL
            )

            db.add(db_user)
            await db.commit()
            await db.refresh(db_user)

            # Convert to Pydantic model
            user = User(
                id=str(db_user.id),  # Cast UUID to string for Pydantic
                email=db_user.email,
                username=db_user.username,
                role=UserRole(db_user.role),
                auth_provider=AuthProvider(db_user.auth_provider)
            )

            logger.info(f"Created user: {user.email}")
            return user

        except IntegrityError:
            await db.rollback()
            raise HTTPException(
                status_code=409,
                detail="User already exists"
            )

    async def authenticate_user(self, login_data: UserLogin, db: AsyncSession = None) -> Optional[User]:
        """Authenticate user with email and password"""
        # Obtain DB session correctly from async generator
        if db is None:
            db_gen = get_db()                    # async generator
            db = await anext(db_gen)             # first yielded session
            should_close_gen = db_gen            # remember to close later
        else:
            should_close_gen = None

        try:
            from sqlalchemy import select

            # Async query
            result = await db.execute(
                select(DBUser).filter(DBUser.email == login_data.email)
            )
            db_user = result.scalars().first()

            if not db_user:
                return None

            # Verify password
            if not verify_password(login_data.password, db_user.password_hash):
                return None

            # Check if user is active
            if not db_user.is_active:
                raise HTTPException(
                    status_code=403,
                    detail="Account is disabled"
                )

            # Update last login
            db_user.last_login = datetime.now(timezone.utc)
            await db.commit()
            await db.refresh(db_user)

            # Convert to Pydantic model
            user = User(
                id=str(db_user.id),  # Cast UUID to string for Pydantic
                email=db_user.email,
                username=db_user.username,
                role=UserRole(db_user.role),
                auth_provider=AuthProvider(db_user.auth_provider)
            )

            return user

        finally:
            if should_close_gen is not None:     # close async generator – commits / rollbacks
                await should_close_gen.aclose()

    async def create_api_key(self, user_id: str, key_name: str, db: AsyncSession = None) -> str:
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
            result = await db.execute(
                select(DBUser).filter(DBUser.id == user_id)
            )
            db_user = result.scalars().first()
            if not db_user:
                raise HTTPException(status_code=404, detail="User not found")

            # Generate key
            api_key = generate_api_key()
            key_id = f"key_{secrets.token_hex(8)}"

            # Create key record in database
            db_api_key = DBAPIKey(
                id=key_id,
                key_hash=hash_api_key(api_key),
                user_id=user_id,
                name=key_name,
                role=db_user.role,
                is_active=True,
                created_at=datetime.now(timezone.utc)
            )

            db.add(db_api_key)
            await db.commit()

            logger.info(f"Created API key '{key_name}' for user {user_id}")
            return api_key  # Return unhashed key only once

        finally:
            if should_close_gen is not None:
                await should_close_gen.aclose()

    async def revoke_token(self, jti: str, token_type: str = "access", user_id: str = None, expires_at: datetime = None):
        """Revoke a JWT token"""
        from services.token_manager import token_manager

        if not expires_at:
            expires_at = datetime.now(timezone.utc) + timedelta(days=1)

        await token_manager.add_revoked_jti(
            jti=jti,
            token_type=token_type,
            user_id=user_id or "unknown",
            expires_at=expires_at,
            reason="manual_revocation"
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

# --- OAuth2 Integration ---

class OAuth2Service:
    """OAuth2 integration service"""

    def __init__(self):
        self.providers = {
            "google": {
                "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                "redirect_uri": os.getenv("GOOGLE_REDIRECT_URI"),
                "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
                "token_url": "https://oauth2.googleapis.com/token"
            },
            "github": {
                "client_id": os.getenv("GITHUB_CLIENT_ID"),
                "client_secret": os.getenv("GITHUB_CLIENT_SECRET"),
                "redirect_uri": os.getenv("GITHUB_REDIRECT_URI"),
                "auth_url": "https://github.com/login/oauth/authorize",
                "token_url": "https://github.com/login/oauth/access_token"
            }
        }

    def get_authorization_url(self, provider: str, state: str) -> str:
        """Get OAuth2 authorization URL"""
        config = self.providers.get(provider)
        if not config:
            raise ValueError(f"Unknown OAuth provider: {provider}")

        params = {
            "client_id": config["client_id"],
            "redirect_uri": config["redirect_uri"],
            "response_type": "code",
            "scope": "email profile",
            "state": state
        }

        # Build URL with params
        from urllib.parse import urlencode
        return f"{config['auth_url']}?{urlencode(params)}"

    async def exchange_code_for_token(self, provider: str, code: str) -> Dict[str, Any]:
        """Exchange authorization code for access token"""
        # Implementation would make HTTP request to provider
        # Return mock data for now
        return {
            "access_token": "mock_oauth_token",
            "token_type": "Bearer",
            "expires_in": 3600
        }

# Create global OAuth service
oauth_service = OAuth2Service()

# --- Session Management ---

class SessionManager:
    """Manage user sessions with database backing"""

    def __init__(self, redis_url: Optional[str] = None):
        self.session_timeout = timedelta(hours=24)
        self.redis_client = None

        if redis_url:
            try:
                import redis
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("Connected to Redis for session management")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")

    async def create_session(
        self,
        user_id: str,
        token_data: TokenData,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        device_id: Optional[str] = None,
        db: AsyncSession = None
    ) -> str:
        """Create a new session"""
        if db is None:
            db_gen = get_db()
            db = await anext(db_gen)
            should_close_gen = db_gen
        else:
            should_close_gen = None

        try:
            from database.models import UserSession as DBUserSession

            session_token = f"sess_{secrets.token_urlsafe(32)}"
            expires_at = datetime.now(timezone.utc) + self.session_timeout

            # Create database record
            db_session = DBUserSession(
                user_id=user_id,
                session_token=session_token,
                ip_address=ip_address,
                user_agent=user_agent,
                device_id=device_id,
                is_active=True,
                expires_at=expires_at
            )

            db.add(db_session)
            await db.commit()
            await db.refresh(db_session)

            # Cache in Redis if available
            if self.redis_client:
                cache_key = f"session:{session_token}"
                cache_data = json.dumps({
                    "user_id": str(user_id),
                    "session_id": str(db_session.id),
                    "expires_at": expires_at.isoformat()
                })
                self.redis_client.setex(
                    cache_key,
                    int(self.session_timeout.total_seconds()),
                    cache_data
                )

            logger.info(f"Created session for user {user_id}")
            return session_token

        finally:
            if should_close_gen is not None:
                await should_close_gen.aclose()

    async def validate_session(self, session_token: str, db: AsyncSession = None) -> Optional[Dict[str, Any]]:
        """Validate if session is still active"""
        # Check Redis cache first
        if self.redis_client:
            cache_key = f"session:{session_token}"
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                session_data = json.loads(cached_data)
                return {
                    "user_id": session_data["user_id"],
                    "session_id": session_data["session_id"]
                }

        # Check database
        if db is None:
            db_gen = get_db()
            db = await anext(db_gen)
            should_close_gen = db_gen
        else:
            should_close_gen = None

        try:
            from database.models import UserSession as DBUserSession

            from sqlalchemy import select
            result = await db.execute(
                select(DBUserSession).filter(
                    DBUserSession.session_token == session_token,
                    DBUserSession.is_active == True,
                    DBUserSession.expires_at > datetime.now(timezone.utc)
                )
            )
            db_session = result.scalars().first()

            if not db_session:
                return None

            # Update last activity
            db_session.last_activity = datetime.now(timezone.utc)
            await db.commit()

            # Update cache
            if self.redis_client:
                cache_key = f"session:{session_token}"
                cache_data = json.dumps({
                    "user_id": str(db_session.user_id),
                    "session_id": str(db_session.id),
                    "expires_at": db_session.expires_at.isoformat()
                })
                self.redis_client.setex(
                    cache_key,
                    int(self.session_timeout.total_seconds()),
                    cache_data
                )

            return {
                "user_id": str(db_session.user_id),
                "session_id": str(db_session.id)
            }

        finally:
            if should_close_gen is not None:
                await should_close_gen.aclose()

    async def end_session(self, session_token: str, db: AsyncSession = None):
        """End a user session"""
        # Remove from cache
        if self.redis_client:
            self.redis_client.delete(f"session:{session_token}")

        # Update database
        if db is None:
            db_gen = get_db()
            db = await anext(db_gen)
            should_close_gen = db_gen
        else:
            should_close_gen = None

        try:
            from database.models import UserSession as DBUserSession

            from sqlalchemy import select
            result = await db.execute(
                select(DBUserSession).filter(
                    DBUserSession.session_token == session_token
                )
            )
            db_session = result.scalars().first()

            if db_session:
                db_session.is_active = False
                await db.commit()
                logger.info(f"Ended session for user {db_session.user_id}")

        finally:
            if should_close_gen is not None:
                await should_close_gen.aclose()

    async def end_all_user_sessions(self, user_id: str, db: AsyncSession = None):
        """End all sessions for a user"""
        if db is None:
            db_gen = get_db()
            db = await anext(db_gen)
            should_close_gen = db_gen
        else:
            should_close_gen = None

        try:
            from database.models import UserSession as DBUserSession

            from sqlalchemy import select
            result = await db.execute(
                select(DBUserSession).filter(
                    DBUserSession.user_id == user_id,
                    DBUserSession.is_active == True
                )
            )
            db_sessions = result.scalars().all()

            for session in db_sessions:
                session.is_active = False
                # Remove from cache
                if self.redis_client:
                    self.redis_client.delete(f"session:{session.session_token}")

            await db.commit()
            logger.info(f"Ended all sessions for user {user_id}")

        finally:
            if should_close_gen is not None:
                await should_close_gen.aclose()

# Create global session manager
session_manager = SessionManager(redis_url=os.getenv("REDIS_URL"))
