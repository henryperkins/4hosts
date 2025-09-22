#!/usr/bin/env python3
"""
Test the research flow with proper authentication and CSRF
"""

import asyncio
import aiohttp
import json
import sys
from typing import Optional, Dict, Any

BASE_URL = "http://127.0.0.1:8001"


async def test_research_flow():
    """Test the complete research flow"""
    # Create session with cookie jar
    jar = aiohttp.CookieJar()
    async with aiohttp.ClientSession(cookie_jar=jar) as session:
        print("=" * 60)
        print("Testing Four Hosts Research Flow")
        print("=" * 60)

        # Step 1: Get CSRF token
        print("\n1. Getting CSRF token...")
        try:
            async with session.get(f"{BASE_URL}/api/csrf-token") as resp:
                if resp.status != 200:
                    print(f"   Failed to get CSRF token: {resp.status}")
                    text = await resp.text()
                    print(f"   Response: {text}")
                    return False

                csrf_data = await resp.json()
                csrf_token = csrf_data.get("csrf_token")
                print(f"   ✓ Got CSRF token: {csrf_token}")

                # Check if token is also in cookies
                csrf_cookie = None
                for cookie in session.cookie_jar:
                    if cookie.key == 'csrf_token':
                        csrf_cookie = cookie.value
                        print(f"   ✓ CSRF token in cookie: {csrf_cookie}")
                        break

                if not csrf_cookie:
                    print(f"   ⚠ WARNING: CSRF token not found in cookies!")
        except Exception as e:
            print(f"   ✗ Error getting CSRF token: {e}")
            return False

        # Step 2: Create a session (for anonymous access)
        print("\n2. Creating anonymous session...")
        try:
            headers = {
                "X-CSRF-Token": csrf_token,
                "Content-Type": "application/json"
            }
            async with session.post(
                f"{BASE_URL}/api/session/create",
                headers=headers,
                json={}
            ) as resp:
                if resp.status != 200:
                    print(f"   Failed to create session: {resp.status}")
                    text = await resp.text()
                    print(f"   Response: {text}")
                else:
                    session_data = await resp.json()
                    print(f"   ✓ Session created: {session_data.get('session_id')}")
        except Exception as e:
            print(f"   ✗ Error creating session: {e}")

        # Step 3: Test research endpoint
        print("\n3. Testing research endpoint...")
        research_query = {
            "query": "What is artificial intelligence and how does it work?",
            "use_llm": True,
            "max_results": 10,
            "include_pdfs": False
        }

        try:
            headers = {
                "X-CSRF-Token": csrf_token,
                "Content-Type": "application/json"
            }

            # Debug: Print cookies being sent
            print(f"   Sending CSRF token in header: {csrf_token}")
            for cookie in session.cookie_jar:
                if cookie.key == 'csrf_token':
                    print(f"   Sending CSRF cookie: {cookie.value}")

            async with session.post(
                f"{BASE_URL}/api/research",
                headers=headers,
                json=research_query
            ) as resp:
                print(f"   Response status: {resp.status}")

                if resp.status == 200:
                    result = await resp.json()
                    print(f"   ✓ Research submitted successfully!")
                    print(f"   Research ID: {result.get('research_id')}")
                    print(f"   Status: {result.get('status')}")

                    # Check if we got immediate results
                    if "answer" in result:
                        print(f"   Answer preview: {result['answer'][:200]}...")

                    return result.get('research_id')
                else:
                    text = await resp.text()
                    print(f"   ✗ Research failed: {text}")
                    return None

        except Exception as e:
            print(f"   ✗ Error submitting research: {e}")
            return None

        # Step 4: Poll for results if we got a research ID
        if research_id := result.get('research_id'):
            print(f"\n4. Polling for results (research_id: {research_id})...")

            for i in range(30):  # Poll for up to 30 seconds
                await asyncio.sleep(1)

                try:
                    async with session.get(
                        f"{BASE_URL}/api/research/status/{research_id}",
                        headers=headers
                    ) as resp:
                        if resp.status == 200:
                            status_data = await resp.json()
                            status = status_data.get("status")
                            print(f"   [{i+1}s] Status: {status}")

                            if status == "completed":
                                print("   ✓ Research completed!")
                                if "answer" in status_data:
                                    print(f"   Answer preview: {status_data['answer'][:200]}...")
                                if "sources" in status_data:
                                    print(f"   Sources found: {len(status_data['sources'])}")
                                return True
                            elif status == "failed":
                                print(f"   ✗ Research failed: {status_data.get('error')}")
                                return False
                except Exception as e:
                    print(f"   Error polling status: {e}")

            print("   ⚠ Timeout waiting for results")
            return False


async def test_health_endpoints():
    """Test basic health endpoints"""
    async with aiohttp.ClientSession() as session:
        print("\nTesting health endpoints...")

        # Test main health endpoint
        async with session.get(f"{BASE_URL}/health") as resp:
            if resp.status == 200:
                data = await resp.json()
                print(f"✓ Health check passed: {data}")
            else:
                print(f"✗ Health check failed: {resp.status}")

        # Test API health endpoint
        async with session.get(f"{BASE_URL}/api/health") as resp:
            if resp.status == 200:
                data = await resp.json()
                print(f"✓ API health check passed: {data}")
            else:
                print(f"✗ API health check failed: {resp.status}")


async def main():
    """Run all tests"""
    # Test health endpoints first
    await test_health_endpoints()

    # Test the research flow
    success = await test_research_flow()

    print("\n" + "=" * 60)
    if success:
        print("✅ RESEARCH FLOW TEST PASSED")
    else:
        print("❌ RESEARCH FLOW TEST FAILED")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)