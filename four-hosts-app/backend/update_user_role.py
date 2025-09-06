#!/usr/bin/env python3
"""Update user role to PRO"""

import asyncio
import os
import sys
from sqlalchemy import select, update
from dotenv import load_dotenv

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

# Conditional imports to handle different SQLAlchemy versions
try:
    from database.connection import get_db_context

    use_context = True
except ImportError:
    from database.connection import get_db

    use_context = False

from database.models import User
from services.auth_service import UserRole


async def update_user_role_to_pro(email: str):
    """Update user role to PRO by email"""
    if use_context:
        async with get_db_context() as db:
            return await _update_role(db, email)
    else:
        # Use the generator version
        db_gen = get_db()
        db = await anext(db_gen)
        try:
            return await _update_role(db, email)
        finally:
            await db_gen.aclose()


async def _update_role(db, email: str):
    """Update user role in database"""
    # Find user by email
    result = await db.execute(select(User).filter(User.email == email))
    user = result.scalars().first()

    if not user:
        print(f"❌ User with email {email} not found")
        return False

    print(f"Found user: {user.username} ({user.email})")
    print(f"Current role: {user.role}")

    # Update role to PRO
    await db.execute(
        update(User).where(User.id == user.id).values(role=UserRole.PRO.value)
    )
    await db.commit()

    # Verify update
    await db.refresh(user)
    print(f"✅ Updated role to: {user.role}")
    return True


if __name__ == "__main__":
    email = "hperkins@example.com"
    if len(sys.argv) > 1:
        email = sys.argv[1]

    print(f"Updating role for user: {email}")
    asyncio.run(update_user_role_to_pro(email))
