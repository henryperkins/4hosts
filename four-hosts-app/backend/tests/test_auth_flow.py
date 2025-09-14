#!/usr/bin/env python3
"""
Test the complete authentication flow
"""

import asyncio
import os
import sys
from pathlib import Path
import json

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

# Load env vars
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, _, value = line.partition('=')
                if key and value:
                    os.environ[key.strip()] = value.strip()

async def test_auth_flow():
    """Test the authentication flow"""
    from services.auth_service import auth_service, UserCreate, UserLogin
    from services.auth.tokens import create_access_token, decode_token
    from database.connection import get_db_context
    import uuid

    print("="*60)
    print("TESTING AUTHENTICATION FLOW")
    print("="*60)

    # Generate unique test user data
    test_id = str(uuid.uuid4())[:8]
    test_email = f"test_{test_id}@example.com"
    test_username = f"testuser_{test_id}"
    test_password = "TestPassword123!"

    print(f"\n1. Creating test user:")
    print(f"   Email: {test_email}")
    print(f"   Username: {test_username}")

    try:
        # Create user
        async with get_db_context() as db:
            user_data = UserCreate(
                email=test_email,
                username=test_username,
                password=test_password,
                role="free"
            )
            user = await auth_service.create_user(user_data, db)
            print(f"   ✓ User created: ID={user.id}")

        # Test login
        print(f"\n2. Testing login:")
        login_data = UserLogin(email=test_email, password=test_password)
        authenticated_user = await auth_service.authenticate_user(login_data)

        if authenticated_user:
            print(f"   ✓ Login successful: {authenticated_user.email}")
        else:
            print(f"   ✗ Login failed")
            return False

        # Generate tokens
        print(f"\n3. Testing token generation:")
        access_token = create_access_token({
            "user_id": str(authenticated_user.id),
            "email": authenticated_user.email,
            "role": authenticated_user.role.value
        })
        print(f"   ✓ Access token generated: {access_token[:50]}...")

        # Validate token
        print(f"\n4. Testing token validation:")
        try:
            payload = await decode_token(access_token)
            print(f"   ✓ Token validated successfully")
            print(f"   ✓ User ID from token: {payload.get('user_id')}")
            print(f"   ✓ Email from token: {payload.get('email')}")
        except Exception as e:
            print(f"   ✗ Token validation failed: {e}")
            return False

        # Test get_current_user dependency
        print(f"\n5. Testing get_current_user dependency:")
        try:
            from services.auth_service import get_current_user, TokenData
            from fastapi.security import HTTPAuthorizationCredentials

            credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=access_token)
            token_data = await get_current_user(credentials)
            print(f"   ✓ get_current_user successful")
            print(f"   ✓ TokenData user_id: {token_data.user_id}")
            print(f"   ✓ TokenData email: {token_data.email}")
        except Exception as e:
            print(f"   ✗ get_current_user failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        print("\n" + "="*60)
        print("✓ ALL AUTHENTICATION TESTS PASSED")
        print("="*60)
        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    success = loop.run_until_complete(test_auth_flow())
    sys.exit(0 if success else 1)