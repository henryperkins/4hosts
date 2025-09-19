#!/usr/bin/env python3
"""Test deep research access for FREE users"""

import asyncio
import aiohttp
import json

API_URL = "http://localhost:8000/v1"

async def test_deep_research():
    async with aiohttp.ClientSession() as session:
        # Skip CSRF for testing
        headers = {"X-Skip-CSRF": "true"}
        
        # Test with a simple curl-like request
        print("Testing deep research endpoint directly...")
        
        # First check if server is up
        try:
            async with session.get(f"{API_URL}/system/limits") as resp:
                if resp.status == 200:
                    print("✅ Server is responding")
                else:
                    print(f"❌ Server returned: {resp.status}")
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            return

        # Test deep research without auth (should work now for all users)
        print("\nTesting if deep research requires authentication...")
        async with session.post(
            f"{API_URL}/research/deep",
            json={
                "query": "Test query for quantum computing",
                "search_context_size": "small"
            },
            headers=headers
        ) as resp:
            if resp.status == 401:
                print("⚠️  Deep research still requires authentication (expected)")
            elif resp.status == 403:
                print("❌ Deep research returned 403 - might still have PRO restriction")
                print(await resp.text())
            elif resp.status == 200:
                print("✅ Deep research accessible!")
            else:
                print(f"Response: {resp.status}")
                print(await resp.text())

asyncio.run(test_deep_research())
