"""
Token Management Service for Four Hosts Research API
Handles refresh tokens, token revocation, and session management
"""

import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import Column, String, DateTime, Boolean, JSON, Integer
from sqlalchemy.ext.declarative import declarative_base
import redis
import logging

from utils.async_utils import run_in_thread

from database.connection import get_db
from database.models import Base

# Configure logging
logger = logging.getLogger(__name__)

# Token configuration
REFRESH_TOKEN_LENGTH = 32
REFRESH_TOKEN_EXPIRE_DAYS = 30
ACCESS_TOKEN_EXPIRE_MINUTES = 30


class RefreshToken(Base):
    """Database model for refresh tokens"""

    __tablename__ = "refresh_tokens"

    id = Column(String(255), primary_key=True)
    token_hash = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    device_id = Column(String(255))
    ip_address = Column(String(45))
    user_agent = Column(String(500))

    # Token metadata
    family_id = Column(String(255), index=True)  # For refresh token rotation
    generation = Column(Integer, default=0)

    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    is_revoked = Column(Boolean, default=False, nullable=False)
    revoked_at = Column(DateTime(timezone=True))
    revoked_reason = Column(String(255))

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    expires_at = Column(DateTime(timezone=True), nullable=False)
    last_used_at = Column(DateTime(timezone=True))

    # Additional security
    scope = Column(JSON, default=list)
    token_metadata = Column(JSON, default=dict)


class RevokedToken(Base):
    """Database model for revoked JWT tokens (JTI blacklist)"""

    __tablename__ = "revoked_tokens"

    jti = Column(String(255), primary_key=True)
    token_type = Column(String(50))  # access, refresh
    user_id = Column(String(255), index=True)
    revoked_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    revoked_reason = Column(String(255))
    expires_at = Column(DateTime(timezone=True), nullable=False)  # Original expiration


class TokenManager:
    """Manages refresh tokens and token revocation"""

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("Connected to Redis for token management")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")

    def generate_refresh_token(self) -> str:
        """Generate a secure refresh token"""
        return f"fhrt_{secrets.token_urlsafe(REFRESH_TOKEN_LENGTH)}"

    def hash_token(self, token: str) -> str:
        """Hash a token for storage using SHA-256 for constant-time lookup"""
        import hashlib

        return hashlib.sha256(token.encode()).hexdigest()

    def verify_token_hash(self, token: str, hashed: str) -> bool:
        """Verify a token against its hash using SHA-256"""
        import hashlib

        return hashlib.sha256(token.encode()).hexdigest() == hashed

    async def create_refresh_token(
        self,
        user_id: str,
        device_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        scope: Optional[list] = None,
        db: AsyncSession = None,
    ) -> Dict[str, Any]:
        """Create a new refresh token (pure async version)"""
        db_gen = get_db()  # async generator
        db = await anext(db_gen)  # first yielded session
        try:
            # Generate tokens
            refresh_token = self.generate_refresh_token()
            token_id = f"rt_{secrets.token_hex(8)}"
            family_id = f"rtf_{secrets.token_hex(8)}"

            # Calculate expiration
            expires_at = datetime.now(timezone.utc) + timedelta(
                days=REFRESH_TOKEN_EXPIRE_DAYS
            )

            # Create database record
            db_token = RefreshToken(
                id=token_id,
                token_hash=self.hash_token(refresh_token),
                user_id=user_id,
                device_id=device_id,
                ip_address=ip_address,
                user_agent=user_agent,
                family_id=family_id,
                generation=0,
                expires_at=expires_at,
                scope=scope or [],
                is_active=True,
            )

            db.add(db_token)
            await db.commit()

            # Cache in Redis if available
            if self.redis_client:
                cache_key = f"refresh_token:{token_id}"
                cache_data = {
                    "user_id": user_id,
                    "family_id": family_id,
                    "generation": 0,
                    "expires_at": expires_at.isoformat(),
                }

                await run_in_thread(
                    self.redis_client.setex,
                    cache_key,
                    int(timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS).total_seconds()),
                    str(cache_data),
                )

            logger.info(f"Created refresh token for user {user_id}")

            return {
                "token": refresh_token,
                "token_id": token_id,
                "expires_at": expires_at,
            }

        finally:
            await db_gen.aclose()

    async def validate_refresh_token(
        self, refresh_token: str, db: AsyncSession = None
    ) -> Optional[Dict[str, Any]]:
        """Validate a refresh token and return token info"""
        if not refresh_token or not refresh_token.startswith("fhrt_"):
            return None

        if db is None:
            db_gen = get_db()
            db = await anext(db_gen)
            should_close_gen = db_gen
        else:
            should_close_gen = None

        try:
            # Query all active refresh tokens
            from sqlalchemy import select

            result = await db.execute(
                select(RefreshToken).filter(
                    RefreshToken.is_active == True,
                    RefreshToken.is_revoked == False,
                    RefreshToken.expires_at > datetime.now(timezone.utc),
                )
            )
            db_tokens = result.scalars().all()

            # Find matching token
            matching_token = None
            for db_token in db_tokens:
                if self.verify_token_hash(refresh_token, db_token.token_hash):
                    matching_token = db_token
                    break

            if not matching_token:
                return None

            # Update last used
            matching_token.last_used_at = datetime.now(timezone.utc)
            await db.commit()

            return {
                "token_id": matching_token.id,
                "user_id": matching_token.user_id,
                "family_id": matching_token.family_id,
                "generation": matching_token.generation,
                "device_id": matching_token.device_id,
                "scope": matching_token.scope,
            }

        finally:
            if should_close_gen is not None:
                await should_close_gen.aclose()

    async def rotate_refresh_token(
        self, old_token: str, db: AsyncSession = None
    ) -> Optional[Dict[str, Any]]:
        """Rotate a refresh token (create new, invalidate old)"""
        # Validate old token
        token_info = await self.validate_refresh_token(old_token, db)
        if not token_info:
            return None

        if db is None:
            db_gen = get_db()
            db = await anext(db_gen)
            should_close_gen = db_gen
        else:
            should_close_gen = None

        try:
            # Revoke old token
            from sqlalchemy import select

            result = await db.execute(
                select(RefreshToken).filter(RefreshToken.id == token_info["token_id"])
            )
            old_db_token = result.scalars().first()

            if old_db_token:
                old_db_token.is_active = False
                old_db_token.is_revoked = True
                old_db_token.revoked_at = datetime.now(timezone.utc)
                old_db_token.revoked_reason = "rotated"

            # Create new token in same family
            new_token = self.generate_refresh_token()
            new_token_id = f"rt_{secrets.token_hex(8)}"
            expires_at = datetime.now(timezone.utc) + timedelta(
                days=REFRESH_TOKEN_EXPIRE_DAYS
            )

            new_db_token = RefreshToken(
                id=new_token_id,
                token_hash=self.hash_token(new_token),
                user_id=token_info["user_id"],
                device_id=token_info.get("device_id"),
                family_id=token_info["family_id"],
                generation=token_info["generation"] + 1,
                expires_at=expires_at,
                scope=token_info.get("scope", []),
                is_active=True,
            )

            db.add(new_db_token)
            await db.commit()

            logger.info(f"Rotated refresh token for user {token_info['user_id']}")

            return {
                "token": new_token,
                "token_id": new_token_id,
                "expires_at": expires_at,
            }

        finally:
            if should_close_gen is not None:
                await should_close_gen.aclose()

    async def revoke_token_family(
        self, family_id: str, reason: str = "security", db: AsyncSession = None
    ):
        """Revoke all tokens in a family (potential token reuse detected)"""
        if db is None:
            db_gen = get_db()
            db = await anext(db_gen)
            should_close_gen = db_gen
        else:
            should_close_gen = None

        try:
            from sqlalchemy import select

            result = await db.execute(
                select(RefreshToken).filter(
                    RefreshToken.family_id == family_id, RefreshToken.is_active == True
                )
            )
            tokens = result.scalars().all()

            for token in tokens:
                token.is_active = False
                token.is_revoked = True
                token.revoked_at = datetime.now(timezone.utc)
                token.revoked_reason = reason

            await db.commit()
            logger.warning(f"Revoked token family {family_id} for reason: {reason}")

        finally:
            if should_close_gen is not None:
                await should_close_gen.aclose()

    async def revoke_user_tokens(
        self, user_id: str, reason: str = "user_request", db: AsyncSession = None
    ):
        """Revoke all refresh tokens for a user"""
        if db is None:
            db_gen = get_db()
            db = await anext(db_gen)
            should_close_gen = db_gen
        else:
            should_close_gen = None

        try:
            from sqlalchemy import select

            result = await db.execute(
                select(RefreshToken).filter(
                    RefreshToken.user_id == user_id, RefreshToken.is_active == True
                )
            )
            tokens = result.scalars().all()

            for token in tokens:
                token.is_active = False
                token.is_revoked = True
                token.revoked_at = datetime.now(timezone.utc)
                token.revoked_reason = reason

            await db.commit()
            logger.info(f"Revoked all tokens for user {user_id}")

        finally:
            if should_close_gen is not None:
                await should_close_gen.aclose()

    async def add_revoked_jti(
        self,
        jti: str,
        token_type: str,
        user_id: str,
        expires_at: datetime,
        reason: str = "manual",
        db: AsyncSession = None,
    ):
        """Add a JTI to the revocation list"""
        if db is None:
            db_gen = get_db()
            db = await anext(db_gen)
            should_close_gen = db_gen
        else:
            should_close_gen = None

        try:
            revoked = RevokedToken(
                jti=jti,
                token_type=token_type,
                user_id=user_id,
                expires_at=expires_at,
                revoked_reason=reason,
            )

            db.add(revoked)
            await db.commit()

            # Cache in Redis if available
            if self.redis_client:
                cache_key = f"revoked_jti:{jti}"
                ttl = int((expires_at - datetime.now(timezone.utc)).total_seconds())
                if ttl > 0:
                    self.redis_client.setex(cache_key, ttl, "1")

            logger.info(f"Revoked JTI {jti} for user {user_id}")

        finally:
            if should_close_gen is not None:
                await should_close_gen.aclose()

    async def is_jti_revoked(self, jti: str, db: AsyncSession = None) -> bool:
        """Check if a JTI is revoked"""
        # Check Redis cache first
        if self.redis_client:
            if self.redis_client.exists(f"revoked_jti:{jti}"):
                return True

        # Check database
        if db is None:
            db_gen = get_db()
            db = await anext(db_gen)
            should_close_gen = db_gen
        else:
            should_close_gen = None

        try:
            from sqlalchemy import select

            result = await db.execute(
                select(RevokedToken).filter(RevokedToken.jti == jti)
            )
            revoked = result.scalars().first()

            return revoked is not None

        finally:
            if should_close_gen is not None:
                await should_close_gen.aclose()

    async def cleanup_expired_tokens(self, db: AsyncSession = None):
        """Clean up expired tokens from database"""
        if db is None:
            db_gen = get_db()
            db = await anext(db_gen)
            should_close_gen = db_gen
        else:
            should_close_gen = None

        try:
            now = datetime.now(timezone.utc)

            # Delete expired refresh tokens
            from sqlalchemy import delete

            await db.execute(delete(RefreshToken).filter(RefreshToken.expires_at < now))

            # Delete expired revoked JTIs
            await db.execute(delete(RevokedToken).filter(RevokedToken.expires_at < now))

            await db.commit()
            logger.info("Cleaned up expired tokens")

        finally:
            if should_close_gen is not None:
                await should_close_gen.aclose()


# Create global token manager instance (single authoritative one)
token_manager = TokenManager(redis_url=os.getenv("REDIS_URL"))
