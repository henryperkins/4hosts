#!/usr/bin/env python3
"""Test database connection"""

import asyncio
import os
from dotenv import load_dotenv
import asyncpg

# Load environment variables
load_dotenv()

async def test_connection():
    """Test PostgreSQL connection"""
    try:
        # Get connection details from environment
        host = os.getenv('PGHOST')
        port = os.getenv('PGPORT', '5432')
        user = os.getenv('PGUSER')
        password = os.getenv('PGPASSWORD')
        database = os.getenv('PGDATABASE')
        
        print(f"Testing connection to {host}:{port}/{database} as {user}...")
        
        # Connect to the database
        conn = await asyncpg.connect(
            host=host,
            port=int(port),
            user=user,
            password=password,
            database=database,
            ssl='require'
        )
        
        # Test a simple query
        version = await conn.fetchval('SELECT version()')
        print(f"Connected successfully!")
        print(f"PostgreSQL version: {version}")
        
        # Check if users table exists
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'users'
            );
        """)
        
        print(f"Users table exists: {table_exists}")
        
        if table_exists:
            # Count users
            user_count = await conn.fetchval('SELECT COUNT(*) FROM users')
            print(f"Number of users in database: {user_count}")
            
            # Check for test user
            test_user = await conn.fetchrow(
                "SELECT id, email, username, role FROM users WHERE email = $1",
                'test@example.com'
            )
            if test_user:
                print(f"Test user found: {dict(test_user)}")
            else:
                print("Test user not found")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_connection())
