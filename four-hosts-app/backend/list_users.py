#!/usr/bin/env python3
"""List all users in the database"""

import asyncio
import os
import sys
from sqlalchemy import select
from dotenv import load_dotenv

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from database.connection import get_db
from database.models import User

async def list_users():
    """List all users in the database"""
    db_gen = get_db()
    db = await anext(db_gen)
    try:
        result = await db.execute(select(User))
        users = result.scalars().all()
        
        if not users:
            print("No users found in the database")
            return
            
        print(f"Found {len(users)} user(s):")
        print("-" * 60)
        for user in users:
            print(f"ID: {user.id}")
            print(f"Username: {user.username}")
            print(f"Email: {user.email}")
            print(f"Role: {user.role}")
            print(f"Created: {user.created_at}")
            print("-" * 60)
            
    finally:
        await db_gen.aclose()

if __name__ == "__main__":
    asyncio.run(list_users())