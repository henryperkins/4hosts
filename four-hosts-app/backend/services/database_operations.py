"""Utility mixin that provides a light wrapper around the existing
`database.connection.get_db_context` helper so service classes don’t have to
repeat the `async with get_db_context() as session:` pattern everywhere.

Example:

    from services.database_operations import DatabaseOperations

    class UserProfileService(DatabaseOperations):
        async def get_user(self, user_id: UUID):
            async with self.db_session() as session:
                ...

The implementation intentionally stays minimal – it simply delegates to the
shared context-manager while giving the call-site a concise, readable alias.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_db_context


class DatabaseOperations:
    """Mixin offering a unified `db_session()` async context manager."""

    @asynccontextmanager
    async def db_session(self) -> AsyncGenerator[AsyncSession, None]:
        async with get_db_context() as session:
            yield session

    # Backwards-compat alias so that refactors can be mechanical (search/replace)
    session_scope = db_session

__all__ = ["DatabaseOperations"]

