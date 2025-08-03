#!/usr/bin/env python3
"""
Test script to verify the authentication flow fix
"""
import asyncio
import httpx
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

async def test_auth_flow():
    """Test the complete authentication flow"""
    async with httpx.AsyncClient() as client:
        print("=== Testing Authentication Flow ===\n")
        
        # 1. Test login
        print("1. Testing login...")
        login_response = await client.post(
            f"{BASE_URL}/auth/login",
            json={"email": "test@example.com", "password": "testpassword123"}
        )
        
        if login_response.status_code != 200:
            print(f"❌ Login failed: {login_response.status_code}")
            print(f"Response: {login_response.text}")
            return
        
        print("✅ Login successful")
        tokens = login_response.json()
        access_token = tokens.get("access_token")
        
        # Extract cookies for refresh token
        cookies = login_response.cookies
        
        # 2. Test /auth/user with access token
        print("\n2. Testing /auth/user endpoint with access token...")
        headers = {"Authorization": f"Bearer {access_token}"}
        
        user_response = await client.get(
            f"{BASE_URL}/auth/user",
            headers=headers
        )
        
        if user_response.status_code != 200:
            print(f"❌ /auth/user failed: {user_response.status_code}")
            print(f"Response: {user_response.text}")
            return
        
        print("✅ /auth/user successful")
        user_data = user_response.json()
        print(f"User data: {json.dumps(user_data, indent=2)}")
        
        # 3. Test token refresh
        print("\n3. Testing token refresh...")
        refresh_response = await client.post(
            f"{BASE_URL}/auth/refresh",
            cookies=cookies
        )
        
        if refresh_response.status_code != 200:
            print(f"❌ Token refresh failed: {refresh_response.status_code}")
            print(f"Response: {refresh_response.text}")
            return
        
        print("✅ Token refresh successful")
        new_tokens = refresh_response.json()
        new_access_token = new_tokens.get("access_token")
        
        # Update cookies with new refresh token
        new_cookies = refresh_response.cookies
        
        # 4. Test /auth/user with new access token
        print("\n4. Testing /auth/user endpoint with refreshed token...")
        new_headers = {"Authorization": f"Bearer {new_access_token}"}
        
        user_response2 = await client.get(
            f"{BASE_URL}/auth/user",
            headers=new_headers
        )
        
        if user_response2.status_code != 200:
            print(f"❌ /auth/user with refreshed token failed: {user_response2.status_code}")
            print(f"Response: {user_response2.text}")
            return
        
        print("✅ /auth/user with refreshed token successful")
        user_data2 = user_response2.json()
        print(f"User data after refresh: {json.dumps(user_data2, indent=2)}")
        
        # 5. Test logout
        print("\n5. Testing logout...")
        logout_response = await client.post(
            f"{BASE_URL}/auth/logout",
            headers=new_headers,
            json={"refresh_token": new_cookies.get("refresh_token")}
        )
        
        if logout_response.status_code != 200:
            print(f"⚠️  Logout returned: {logout_response.status_code}")
            print(f"Response: {logout_response.text}")
        else:
            print("✅ Logout successful")
        
        # 6. Verify access is denied after logout
        print("\n6. Verifying access is denied after logout...")
        denied_response = await client.get(
            f"{BASE_URL}/auth/user",
            headers=new_headers
        )
        
        if denied_response.status_code == 401:
            print("✅ Access correctly denied after logout")
        else:
            print(f"⚠️  Expected 401, got: {denied_response.status_code}")
        
        print("\n=== All tests completed ===")

if __name__ == "__main__":
    asyncio.run(test_auth_flow())