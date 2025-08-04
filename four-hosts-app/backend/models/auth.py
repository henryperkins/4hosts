"""
Authentication-related Pydantic models
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, EmailStr

from models.base import UserRole


class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    role: UserRole = UserRole.FREE


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: Optional[str] = None


class PreferencesPayload(BaseModel):
    """Payload for updating user preferences"""
    preferences: Dict[str, Any]
