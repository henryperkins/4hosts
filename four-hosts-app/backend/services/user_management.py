"""
User Management and Saved Searches Service
Phase 5: Production-Ready Features
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from uuid import UUID
import secrets
import logging

from sqlalchemy import select, update, delete, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from database.models import (
    User,
    ResearchQuery,
    APIKey,
    UserSession,
    user_saved_searches,
    UserRole,
    ResearchStatus,
    ParadigmType,
)
from database.connection import get_db_context
from services.auth import hash_password, verify_password, create_access_token

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- User Profile Management ---


class UserProfileService:
    """Service for managing user profiles"""

    async def get_user_profile(self, user_id: UUID) -> Optional[Dict[str, Any]]:
        """Get complete user profile"""
        async with get_db_context() as session:
            stmt = (
                select(User)
                .where(User.id == user_id)
                .options(
                    selectinload(User.api_keys),
                    selectinload(User.research_queries),
                    selectinload(User.saved_searches),
                )
            )

            result = await session.execute(stmt)
            user = result.scalar_one_or_none()

            if not user:
                return None

            # Calculate usage statistics
            research_count = len(user.research_queries)
            completed_research = sum(
                1 for r in user.research_queries if r.status == ResearchStatus.COMPLETED
            )

            return {
                "id": str(user.id),
                "email": user.email,
                "username": user.username,
                "full_name": user.full_name,
                "avatar_url": user.avatar_url,
                "bio": user.bio,
                "role": user.role,
                "is_verified": user.is_verified,
                "created_at": user.created_at.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "preferences": user.preferences or {},
                "statistics": {
                    "total_research": research_count,
                    "completed_research": completed_research,
                    "saved_searches": len(user.saved_searches),
                    "active_api_keys": sum(1 for k in user.api_keys if k.is_active),
                },
            }

    async def update_user_profile(self, user_id: UUID, updates: Dict[str, Any]) -> bool:
        """Update user profile"""
        allowed_fields = {"full_name", "avatar_url", "bio", "preferences"}

        # Filter to allowed fields
        filtered_updates = {k: v for k, v in updates.items() if k in allowed_fields}

        if not filtered_updates:
            return False

        async with get_db_context() as session:
            stmt = (
                update(User)
                .where(User.id == user_id)
                .values(**filtered_updates, updated_at=datetime.now(timezone.utc))
            )

            result = await session.execute(stmt)
            await session.commit()

            return result.rowcount > 0

    async def update_user_preferences(
        self, user_id: UUID, preferences: Dict[str, Any]
    ) -> bool:
        """Update user preferences"""
        async with get_db_context() as session:
            # Get current preferences
            stmt = select(User.preferences).where(User.id == user_id)
            result = await session.execute(stmt)
            current_prefs = result.scalar_one_or_none() or {}

            # Merge preferences
            updated_prefs = {**current_prefs, **preferences}

            # Update
            stmt = (
                update(User).where(User.id == user_id).values(preferences=updated_prefs)
            )

            result = await session.execute(stmt)
            await session.commit()

            return result.rowcount > 0

    async def change_password(
        self, user_id: UUID, current_password: str, new_password: str
    ) -> bool:
        """Change user password"""
        async with get_db_context() as session:
            # Get user
            stmt = select(User).where(User.id == user_id)
            result = await session.execute(stmt)
            user = result.scalar_one_or_none()

            if not user or not user.password_hash:
                return False

            # Verify current password
            if not await verify_password(current_password, user.password_hash):
                return False

            # Update password
            new_hash = await hash_password(new_password)
            stmt = update(User).where(User.id == user_id).values(password_hash=new_hash)

            await session.execute(stmt)
            await session.commit()

            logger.info(f"Password changed for user {user_id}")
            return True

    async def delete_user_account(self, user_id: UUID, password: str) -> bool:
        """Delete user account (soft delete)"""
        async with get_db_context() as session:
            # Verify password
            stmt = select(User).where(User.id == user_id)
            result = await session.execute(stmt)
            user = result.scalar_one_or_none()

            if not user or not await verify_password(password, user.password_hash):
                return False

            # Soft delete - anonymize data
            stmt = (
                update(User)
                .where(User.id == user_id)
                .values(
                    email=f"deleted_{user.id}@deleted.com",
                    username=f"deleted_{user.id}",
                    full_name="Deleted User",
                    password_hash=None,
                    is_active=False,
                    avatar_url=None,
                    bio=None,
                    preferences={},
                    updated_at=datetime.now(timezone.utc),
                )
            )

            await session.execute(stmt)
            await session.commit()

            logger.info(f"User account {user_id} deleted")
            return True


# --- Saved Searches Management ---


class SavedSearchesService:
    """Service for managing saved searches"""

    async def save_search(
        self,
        user_id: UUID,
        research_id: UUID,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Save a research query for later reference"""
        async with get_db_context() as session:
            # Check if research exists and belongs to user or is public
            stmt = select(ResearchQuery).where(
                and_(
                    ResearchQuery.id == research_id,
                    or_(
                        ResearchQuery.user_id == user_id,
                        ResearchQuery.user_id == None,  # Public research
                    ),
                )
            )

            result = await session.execute(stmt)
            research = result.scalar_one_or_none()

            if not research:
                return False

            # Check if already saved
            stmt = select(user_saved_searches).where(
                and_(
                    user_saved_searches.c.user_id == user_id,
                    user_saved_searches.c.research_id == research_id,
                )
            )

            result = await session.execute(stmt)
            if result.first():
                # Update existing save
                stmt = (
                    update(user_saved_searches)
                    .where(
                        and_(
                            user_saved_searches.c.user_id == user_id,
                            user_saved_searches.c.research_id == research_id,
                        )
                    )
                    .values(
                        tags=tags or [],
                        notes=notes,
                        saved_at=datetime.now(timezone.utc),
                    )
                )
            else:
                # Create new save
                stmt = user_saved_searches.insert().values(
                    user_id=user_id,
                    research_id=research_id,
                    tags=tags or [],
                    notes=notes,
                    saved_at=datetime.now(timezone.utc),
                )

            await session.execute(stmt)
            await session.commit()

            return True

    async def unsave_search(self, user_id: UUID, research_id: UUID) -> bool:
        """Remove a saved search"""
        async with get_db_context() as session:
            stmt = delete(user_saved_searches).where(
                and_(
                    user_saved_searches.c.user_id == user_id,
                    user_saved_searches.c.research_id == research_id,
                )
            )

            result = await session.execute(stmt)
            await session.commit()

            return result.rowcount > 0

    async def get_saved_searches(
        self,
        user_id: UUID,
        tags: Optional[List[str]] = None,
        paradigm: Optional[ParadigmType] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get user's saved searches with filters"""
        async with get_db_context() as session:
            # Build query
            stmt = (
                select(
                    ResearchQuery,
                    user_saved_searches.c.tags,
                    user_saved_searches.c.notes,
                    user_saved_searches.c.saved_at,
                )
                .join(
                    user_saved_searches,
                    ResearchQuery.id == user_saved_searches.c.research_id,
                )
                .where(user_saved_searches.c.user_id == user_id)
            )

            # Apply filters
            if tags:
                # Filter by any matching tag
                tag_conditions = []
                for tag in tags:
                    tag_conditions.append(user_saved_searches.c.tags.contains([tag]))
                stmt = stmt.where(or_(*tag_conditions))

            if paradigm:
                stmt = stmt.where(ResearchQuery.primary_paradigm == paradigm)

            # Order by saved date
            stmt = (
                stmt.order_by(user_saved_searches.c.saved_at.desc())
                .limit(limit)
                .offset(offset)
            )

            result = await session.execute(stmt)
            saved_searches = []

            for row in result:
                research = row[0]
                saved_searches.append(
                    {
                        "research_id": str(research.id),
                        "query": research.query_text,
                        "paradigm": research.primary_paradigm,
                        "status": research.status,
                        "created_at": research.created_at.isoformat(),
                        "completed_at": (
                            research.completed_at.isoformat()
                            if research.completed_at
                            else None
                        ),
                        "confidence_score": research.confidence_score,
                        "saved_at": row.saved_at.isoformat(),
                        "tags": row.tags or [],
                        "notes": row.notes,
                    }
                )

            return saved_searches

    async def search_saved_queries(
        self, user_id: UUID, search_term: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search through saved queries"""
        async with get_db_context() as session:
            # Search in query text and notes
            stmt = (
                select(
                    ResearchQuery,
                    user_saved_searches.c.tags,
                    user_saved_searches.c.notes,
                    user_saved_searches.c.saved_at,
                )
                .join(
                    user_saved_searches,
                    ResearchQuery.id == user_saved_searches.c.research_id,
                )
                .where(
                    and_(
                        user_saved_searches.c.user_id == user_id,
                        or_(
                            ResearchQuery.query_text.ilike(f"%{search_term}%"),
                            user_saved_searches.c.notes.ilike(f"%{search_term}%"),
                        ),
                    )
                )
                .order_by(user_saved_searches.c.saved_at.desc())
                .limit(limit)
            )

            result = await session.execute(stmt)
            results = []

            for row in result:
                research = row[0]
                results.append(
                    {
                        "research_id": str(research.id),
                        "query": research.query_text,
                        "paradigm": research.primary_paradigm,
                        "saved_at": row.saved_at.isoformat(),
                        "tags": row.tags or [],
                        "notes": row.notes,
                        "match_context": self._get_match_context(
                            research.query_text, row.notes, search_term
                        ),
                    }
                )

            return results

    def _get_match_context(
        self, query: str, notes: Optional[str], search_term: str
    ) -> str:
        """Get context around search match"""
        search_lower = search_term.lower()

        # Check query
        if search_lower in query.lower():
            idx = query.lower().index(search_lower)
            start = max(0, idx - 30)
            end = min(len(query), idx + len(search_term) + 30)
            return f"...{query[start:end]}..."

        # Check notes
        if notes and search_lower in notes.lower():
            idx = notes.lower().index(search_lower)
            start = max(0, idx - 30)
            end = min(len(notes), idx + len(search_term) + 30)
            return f"...{notes[start:end]}..."

        return query[:100] + "..."


# --- API Key Management ---


class APIKeyService:
    """Service for managing API keys"""

    async def list_api_keys(self, user_id: UUID) -> List[Dict[str, Any]]:
        """List all API keys for user"""
        async with get_db_context() as session:
            stmt = (
                select(APIKey)
                .where(APIKey.user_id == user_id)
                .order_by(APIKey.created_at.desc())
            )

            result = await session.execute(stmt)
            keys = []

            for key in result.scalars():
                keys.append(
                    {
                        "id": str(key.id),
                        "name": key.name,
                        "role": key.role,
                        "is_active": key.is_active,
                        "created_at": key.created_at.isoformat(),
                        "last_used": (
                            key.last_used.isoformat() if key.last_used else None
                        ),
                        "usage_count": key.usage_count,
                        "expires_at": (
                            key.expires_at.isoformat() if key.expires_at else None
                        ),
                        "allowed_origins": key.allowed_origins,
                        "rate_limit_tier": key.rate_limit_tier,
                    }
                )

            return keys

    async def create_api_key(
        self,
        user_id: UUID,
        name: str,
        role: Optional[UserRole] = None,
        expires_in_days: Optional[int] = None,
        allowed_origins: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create new API key"""
        async with get_db_context() as session:
            # Get user's role if not specified
            if not role:
                stmt = select(User.role).where(User.id == user_id)
                result = await session.execute(stmt)
                user_role = result.scalar_one_or_none()
                role = user_role or UserRole.FREE

            # Generate key
            raw_key = f"fh_{secrets.token_urlsafe(32)}"
            key_hash = await hash_password(raw_key)  # Reuse password hashing

            # Calculate expiry
            expires_at = None
            if expires_in_days:
                expires_at = datetime.now(timezone.utc) + timedelta(
                    days=expires_in_days
                )

            # Create key record
            api_key = APIKey(
                user_id=user_id,
                key_hash=key_hash,
                name=name,
                role=role,
                allowed_origins=allowed_origins or [],
                expires_at=expires_at,
                rate_limit_tier=self._get_rate_limit_tier(role),
            )

            session.add(api_key)
            await session.commit()

            return {
                "id": str(api_key.id),
                "key": raw_key,  # Only returned once!
                "name": name,
                "role": role,
                "expires_at": expires_at.isoformat() if expires_at else None,
                "message": "Store this key securely. It cannot be retrieved again.",
            }

    async def revoke_api_key(self, user_id: UUID, key_id: UUID) -> bool:
        """Revoke an API key"""
        async with get_db_context() as session:
            stmt = (
                update(APIKey)
                .where(and_(APIKey.id == key_id, APIKey.user_id == user_id))
                .values(is_active=False, revoked_at=datetime.now(timezone.utc))
            )

            result = await session.execute(stmt)
            await session.commit()

            return result.rowcount > 0

    async def update_api_key(
        self, user_id: UUID, key_id: UUID, updates: Dict[str, Any]
    ) -> bool:
        """Update API key settings"""
        allowed_fields = {"name", "allowed_origins", "allowed_ips", "rate_limit_tier"}

        filtered_updates = {k: v for k, v in updates.items() if k in allowed_fields}

        if not filtered_updates:
            return False

        async with get_db_context() as session:
            stmt = (
                update(APIKey)
                .where(
                    and_(
                        APIKey.id == key_id,
                        APIKey.user_id == user_id,
                        APIKey.is_active == True,
                    )
                )
                .values(**filtered_updates)
            )

            result = await session.execute(stmt)
            await session.commit()

            return result.rowcount > 0

    def _get_rate_limit_tier(self, role: UserRole) -> str:
        """Get rate limit tier based on role"""
        tier_map = {
            UserRole.FREE: "basic",
            UserRole.BASIC: "standard",
            UserRole.PRO: "professional",
            UserRole.ENTERPRISE: "enterprise",
            UserRole.ADMIN: "unlimited",
        }
        return tier_map.get(role, "basic")


# --- Research History Service ---


class ResearchHistoryService:
    """Service for managing research history"""

    async def get_research_history(
        self,
        user_id: UUID,
        status: Optional[ResearchStatus] = None,
        paradigm: Optional[ParadigmType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get user's research history with filters"""
        async with get_db_context() as session:
            stmt = select(ResearchQuery).where(ResearchQuery.user_id == user_id)

            # Apply filters
            if status:
                stmt = stmt.where(ResearchQuery.status == status)

            if paradigm:
                stmt = stmt.where(ResearchQuery.primary_paradigm == paradigm)

            if start_date:
                stmt = stmt.where(ResearchQuery.created_at >= start_date)

            if end_date:
                stmt = stmt.where(ResearchQuery.created_at <= end_date)

            # Order and limit
            stmt = (
                stmt.order_by(ResearchQuery.created_at.desc())
                .limit(limit)
                .offset(offset)
            )

            result = await session.execute(stmt)
            history = []

            for research in result.scalars():
                history.append(
                    {
                        "id": str(research.id),
                        "query": research.query_text,
                        "paradigm": research.primary_paradigm,
                        "secondary_paradigm": research.secondary_paradigm,
                        "status": research.status,
                        "depth": research.depth,
                        "created_at": research.created_at.isoformat(),
                        "completed_at": (
                            research.completed_at.isoformat()
                            if research.completed_at
                            else None
                        ),
                        "duration_seconds": research.duration_seconds,
                        "sources_analyzed": research.sources_analyzed,
                        "confidence_score": research.confidence_score,
                        "error_message": research.error_message,
                    }
                )

            return history

    async def get_research_statistics(
        self, user_id: UUID, period_days: int = 30
    ) -> Dict[str, Any]:
        """Get research usage statistics"""
        async with get_db_context() as session:
            since_date = datetime.now(timezone.utc) - timedelta(days=period_days)

            # Get aggregated stats
            stmt = select(
                func.count(ResearchQuery.id).label("total_queries"),
                func.count(ResearchQuery.id)
                .filter(ResearchQuery.status == ResearchStatus.COMPLETED)
                .label("completed_queries"),
                func.avg(ResearchQuery.duration_seconds).label("avg_duration"),
                func.avg(ResearchQuery.confidence_score).label("avg_confidence"),
                func.sum(ResearchQuery.sources_analyzed).label("total_sources"),
            ).where(
                and_(
                    ResearchQuery.user_id == user_id,
                    ResearchQuery.created_at >= since_date,
                )
            )

            result = await session.execute(stmt)
            stats = result.one()

            # Get paradigm distribution
            paradigm_stmt = (
                select(
                    ResearchQuery.primary_paradigm,
                    func.count(ResearchQuery.id).label("count"),
                )
                .where(
                    and_(
                        ResearchQuery.user_id == user_id,
                        ResearchQuery.created_at >= since_date,
                    )
                )
                .group_by(ResearchQuery.primary_paradigm)
            )

            paradigm_result = await session.execute(paradigm_stmt)
            paradigm_dist = {row.primary_paradigm: row.count for row in paradigm_result}

            return {
                "period_days": period_days,
                "total_queries": stats.total_queries or 0,
                "completed_queries": stats.completed_queries or 0,
                "completion_rate": (
                    (stats.completed_queries / stats.total_queries * 100)
                    if stats.total_queries > 0
                    else 0
                ),
                "avg_duration_seconds": float(stats.avg_duration or 0),
                "avg_confidence_score": float(stats.avg_confidence or 0),
                "total_sources_analyzed": stats.total_sources or 0,
                "paradigm_distribution": paradigm_dist,
            }


# --- Session Management ---


class SessionService:
    """Service for managing user sessions"""

    async def create_session(
        self,
        user_id: UUID,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        device_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create new user session"""
        async with get_db_context() as session:
            # Generate tokens
            session_token = secrets.token_urlsafe(32)
            refresh_token = secrets.token_urlsafe(32)

            # Create session
            user_session = UserSession(
                user_id=user_id,
                session_token=session_token,
                refresh_token=refresh_token,
                ip_address=ip_address,
                user_agent=user_agent,
                device_id=device_id,
                expires_at=datetime.now(timezone.utc) + timedelta(days=30),
            )

            session.add(user_session)

            # Update user last login
            stmt = (
                update(User)
                .where(User.id == user_id)
                .values(last_login=datetime.now(timezone.utc))
            )
            await session.execute(stmt)

            await session.commit()

            return {
                "session_id": str(user_session.id),
                "session_token": session_token,
                "refresh_token": refresh_token,
                "expires_at": user_session.expires_at.isoformat(),
            }

    async def validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Validate and get session info"""
        async with get_db_context() as session:
            stmt = select(UserSession).where(
                and_(
                    UserSession.session_token == session_token,
                    UserSession.is_active == True,
                    UserSession.expires_at > datetime.now(timezone.utc),
                )
            )

            result = await session.execute(stmt)
            user_session = result.scalar_one_or_none()

            if not user_session:
                return None

            # Update last activity
            stmt = (
                update(UserSession)
                .where(UserSession.id == user_session.id)
                .values(last_activity=datetime.now(timezone.utc))
            )
            await session.execute(stmt)
            await session.commit()

            return {
                "session_id": str(user_session.id),
                "user_id": str(user_session.user_id),
                "created_at": user_session.created_at.isoformat(),
                "expires_at": user_session.expires_at.isoformat(),
            }

    async def end_session(self, session_token: str) -> bool:
        """End a user session"""
        async with get_db_context() as session:
            stmt = (
                update(UserSession)
                .where(UserSession.session_token == session_token)
                .values(is_active=False)
            )

            result = await session.execute(stmt)
            await session.commit()

            return result.rowcount > 0

    async def end_all_sessions(
        self, user_id: UUID, except_token: Optional[str] = None
    ) -> int:
        """End all sessions for user except specified one"""
        async with get_db_context() as session:
            stmt = update(UserSession).where(
                and_(UserSession.user_id == user_id, UserSession.is_active == True)
            )

            if except_token:
                stmt = stmt.where(UserSession.session_token != except_token)

            stmt = stmt.values(is_active=False)

            result = await session.execute(stmt)
            await session.commit()

            return result.rowcount


# --- Create service instances ---

user_profile_service = UserProfileService()
saved_searches_service = SavedSearchesService()
api_key_service = APIKeyService()
research_history_service = ResearchHistoryService()
session_service = SessionService()
