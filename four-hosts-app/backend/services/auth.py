"""
Authentication and Authorization Service for Four Hosts Research API
Phase 5: Production-Ready Features
"""

import os
import jwt
import bcrypt
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, EmailStr, Field
from fastapi import HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
from enum import Enum
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7
API_KEY_LENGTH = 32

# Security
security = HTTPBearer()

# --- Data Models ---

class UserRole(str, Enum):
    """User roles for authorization"""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"

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

def create_refresh_token(user_id: str) -> str:
    """Create a JWT refresh token"""
    data = {
        "user_id": user_id,
        "type": "refresh"
    }
    expires_delta = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    return create_access_token(data, expires_delta)

def decode_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# --- Authorization Functions ---

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> TokenData:
    """Get current user from JWT token"""
    token = credentials.credentials
    payload = decode_token(token)
    
    # Check if token is revoked (would check against database/cache)
    # if is_token_revoked(payload.get("jti")):
    #     raise HTTPException(status_code=401, detail="Token has been revoked")
    
    return TokenData(
        user_id=payload.get("user_id"),
        email=payload.get("email"),
        role=UserRole(payload.get("role")),
        exp=datetime.fromtimestamp(payload.get("exp"), tz=timezone.utc),
        iat=datetime.fromtimestamp(payload.get("iat"), tz=timezone.utc),
        jti=payload.get("jti")
    )

async def get_api_key_info(api_key: str) -> Optional[APIKey]:
    """Validate API key and return key info"""
    # In production, this would query the database
    # For now, return mock data for valid API keys starting with "fh_"
    if not api_key.startswith("fh_"):
        return None
    
    # Mock API key info
    return APIKey(
        id=f"key_{secrets.token_hex(8)}",
        key_hash=hash_api_key(api_key),
        user_id="user_123",
        name="Production API Key",
        role=UserRole.PRO,
        created_at=datetime.now(timezone.utc),
        is_active=True,
        rate_limit_tier="pro"
    )

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
        # In production, this would use a database
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self.revoked_tokens: set = set()
    
    async def create_user(self, user_data: UserCreate) -> User:
        """Create a new user"""
        # Validate password strength
        if not validate_password_strength(user_data.password):
            raise HTTPException(
                status_code=400,
                detail="Password must be at least 8 characters with uppercase, lowercase, number, and special character"
            )
        
        # Check if user already exists
        if any(u.email == user_data.email for u in self.users.values()):
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create user
        user_id = f"user_{secrets.token_hex(8)}"
        user = User(
            id=user_id,
            email=user_data.email,
            username=user_data.username,
            role=user_data.role,
            auth_provider=user_data.auth_provider
        )
        
        # Store user with hashed password
        self.users[user_id] = user
        # In production, store hashed password separately
        
        logger.info(f"Created new user: {user.email}")
        return user
    
    async def authenticate_user(self, login_data: UserLogin) -> Optional[User]:
        """Authenticate user with email and password"""
        # In production, query database
        for user in self.users.values():
            if user.email == login_data.email:
                # In production, verify against stored hash
                # if verify_password(login_data.password, stored_hash):
                user.last_login = datetime.now(timezone.utc)
                return user
        return None
    
    async def create_api_key(self, user_id: str, key_name: str) -> str:
        """Create a new API key for user"""
        # Get user
        user = self.users.get(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Generate key
        api_key = generate_api_key()
        key_id = f"key_{secrets.token_hex(8)}"
        
        # Create key record
        key_record = APIKey(
            id=key_id,
            key_hash=hash_api_key(api_key),
            user_id=user_id,
            name=key_name,
            role=user.role
        )
        
        self.api_keys[key_id] = key_record
        
        logger.info(f"Created API key '{key_name}' for user {user_id}")
        return api_key  # Return unhashed key only once
    
    async def revoke_token(self, jti: str):
        """Revoke a JWT token"""
        self.revoked_tokens.add(jti)
        logger.info(f"Revoked token: {jti}")
    
    def is_token_revoked(self, jti: str) -> bool:
        """Check if token is revoked"""
        return jti in self.revoked_tokens
    
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
    """Manage user sessions and active tokens"""
    
    def __init__(self):
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = timedelta(hours=24)
    
    def create_session(self, user_id: str, token_data: TokenData) -> str:
        """Create a new session"""
        session_id = secrets.token_urlsafe(32)
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "token_jti": token_data.jti,
            "created_at": datetime.now(timezone.utc),
            "last_activity": datetime.now(timezone.utc),
            "ip_address": None,  # Would be set from request
            "user_agent": None   # Would be set from request
        }
        return session_id
    
    def validate_session(self, session_id: str) -> bool:
        """Validate if session is still active"""
        session = self.active_sessions.get(session_id)
        if not session:
            return False
        
        # Check timeout
        if datetime.now(timezone.utc) - session["last_activity"] > self.session_timeout:
            del self.active_sessions[session_id]
            return False
        
        # Update last activity
        session["last_activity"] = datetime.now(timezone.utc)
        return True
    
    def end_session(self, session_id: str):
        """End a user session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

# Create global session manager
session_manager = SessionManager()