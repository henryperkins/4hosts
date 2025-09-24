#!/usr/bin/env python3
"""
Authentication diagnostic test script
Validates the complete CSRF + login flow
"""

import asyncio
import httpx
import json
import sys
from pathlib import Path


async def test_auth_flow():
    """Test the complete authentication flow"""
    base_url = "http://localhost:8000"

    async with httpx.AsyncClient() as client:
        print("🔍 Testing authentication flow...")

        # Step 1: Get CSRF token
        print("\n1. Getting CSRF token...")
        csrf_response = await client.get(f"{base_url}/api/csrf-token")
        print(f"   Status: {csrf_response.status_code}")

        if csrf_response.status_code != 200:
            print(f"   ❌ Failed to get CSRF token: {csrf_response.text}")
            return False

        csrf_data = csrf_response.json()
        csrf_token = csrf_data.get("csrf_token")
        cookies = csrf_response.cookies

        print(f"   ✅ CSRF token: {csrf_token[:16]}...")
        print(f"   ✅ Cookies: {dict(cookies)}")

        # Step 2: Check debug status (development only)
        print("\n2. Checking auth debug status...")
        debug_response = await client.get(
            f"{base_url}/v1/auth/debug/status",
            cookies=cookies,
            headers={"X-CSRF-Token": csrf_token}
        )

        if debug_response.status_code == 200:
            debug_data = debug_response.json()
            print(f"   ✅ Debug info: {json.dumps(debug_data, indent=2)}")
        else:
            print(f"   ⚠️  Debug endpoint unavailable (production mode?)")

        # Step 3: Attempt login
        print("\n3. Testing login...")
        login_data = {
            "email": "test@example.com",
            "password": "password123"
        }

        login_response = await client.post(
            f"{base_url}/v1/auth/login",
            json=login_data,
            cookies=cookies,
            headers={
                "Content-Type": "application/json",
                "X-CSRF-Token": csrf_token
            }
        )

        print(f"   Status: {login_response.status_code}")
        print(f"   Response: {login_response.text[:200]}...")

        if login_response.status_code == 200:
            print("   ✅ Login successful!")
            # Update cookies with auth tokens
            cookies.update(login_response.cookies)

            # Step 4: Test authenticated endpoint
            print("\n4. Testing authenticated endpoint...")
            user_response = await client.get(
                f"{base_url}/v1/auth/user",
                cookies=cookies
            )

            print(f"   Status: {user_response.status_code}")
            if user_response.status_code == 200:
                user_data = user_response.json()
                print(f"   ✅ User data: {json.dumps(user_data, indent=2)}")
                return True
            else:
                print(f"   ❌ Failed to get user data: {user_response.text}")
                return False

        elif login_response.status_code == 401:
            print("   ⚠️  Login failed - checking reasons...")

            # Check if it's a CSRF issue
            if "CSRF" in login_response.text:
                print("   ❌ CSRF token mismatch")
            elif "Invalid credentials" in login_response.text:
                print("   ❌ Invalid credentials (user may not exist)")
            else:
                print(f"   ❌ Unknown authentication failure: {login_response.text}")
            return False

        elif login_response.status_code == 403:
            print("   ❌ CSRF protection blocked the request")
            return False
        else:
            print(f"   ❌ Unexpected status code: {login_response.status_code}")
            print(f"   Response: {login_response.text}")
            return False


async def create_test_user():
    """Create a test user for authentication testing"""
    base_url = "http://localhost:8000"

    async with httpx.AsyncClient() as client:
        print("👤 Creating test user...")

        # Get CSRF token first
        csrf_response = await client.get(f"{base_url}/api/csrf-token")
        if csrf_response.status_code != 200:
            print(f"   ❌ Failed to get CSRF token for registration")
            return False

        csrf_data = csrf_response.json()
        csrf_token = csrf_data.get("csrf_token")
        cookies = csrf_response.cookies

        # Register test user
        register_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password": "password123",
            "role": "FREE"
        }

        register_response = await client.post(
            f"{base_url}/v1/auth/register",
            json=register_data,
            cookies=cookies,
            headers={
                "Content-Type": "application/json",
                "X-CSRF-Token": csrf_token
            }
        )

        print(f"   Status: {register_response.status_code}")

        if register_response.status_code == 200:
            print("   ✅ Test user created successfully!")
            return True
        elif register_response.status_code == 400:
            if "already exists" in register_response.text.lower():
                print("   ℹ️  Test user already exists")
                return True
            else:
                print(f"   ❌ Registration failed: {register_response.text}")
                return False
        else:
            print(f"   ❌ Unexpected status code: {register_response.status_code}")
            print(f"   Response: {register_response.text}")
            return False


async def main():
    """Main diagnostic routine"""
    print("🚀 Four Hosts Authentication Diagnostic")
    print("=" * 50)

    # Check if server is running
    try:
        async with httpx.AsyncClient() as client:
            health_response = await client.get("http://localhost:8000/health")
            if health_response.status_code != 200:
                print("❌ Server health check failed")
                return 1
    except httpx.ConnectError:
        print("❌ Cannot connect to server at http://localhost:8000")
        print("   Make sure the backend is running:")
        print("   cd four-hosts-app/backend && uvicorn main_new:app --reload")
        return 1

    print("✅ Server is running")

    # Create test user if needed
    user_created = await create_test_user()
    if not user_created:
        print("❌ Failed to create test user")
        return 1

    # Test authentication flow
    auth_success = await test_auth_flow()

    if auth_success:
        print("\n🎉 Authentication diagnostic completed successfully!")
        print("   All auth flows are working correctly.")
        return 0
    else:
        print("\n❌ Authentication diagnostic failed!")
        print("   Check the error messages above for details.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
