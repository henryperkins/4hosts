import asyncio
from database.connection import get_db_context
from database.models import User

async def test_db():
    """Test basic database connection"""
    try:
        async with get_db_context() as db:
            print("✓ Database connection successful")
            
            # Try a simple query
            from sqlalchemy import select
            result = await db.execute(select(User).limit(1))
            users = result.scalars().all()
            print(f"✓ Query executed successfully, found {len(users)} users")
            
    except Exception as e:
        print(f"✗ Database error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_db())