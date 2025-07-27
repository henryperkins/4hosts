"""
Basic authentication service for Four Hosts API
"""
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import jwt
import os
from pydantic import BaseModel

class TokenData(BaseModel):
    username: Optional[str] = None
    scopes: list[str] = []

class AuthService:
    """Basic authentication service"""
    
    def __init__(self):
        self.secret_key = os.getenv("SECRET_KEY", "your-secret-key-here")
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Create a JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            if username is None:
                return None
            token_data = TokenData(username=username, scopes=payload.get("scopes", []))
            return token_data
        except jwt.JWTError:
            return None

# Create singleton instance
auth_service = AuthService()
