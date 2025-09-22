#!/usr/bin/env python3
"""
Test research API with authentication
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8001"

def test_authenticated_research():
    # Create session to maintain cookies
    session = requests.Session()

    print("=" * 60)
    print("Testing Authenticated Research Flow")
    print("=" * 60)

    # Step 1: Get CSRF token
    print("\n1. Getting CSRF token...")
    response = session.get(f"{BASE_URL}/api/csrf-token")
    if response.status_code != 200:
        print(f"   Failed to get CSRF token: {response.status_code}")
        return False

    csrf_data = response.json()
    csrf_token = csrf_data['csrf_token']
    print(f"   ✓ Got CSRF token: {csrf_token}")

    # Step 2: Login with provided credentials
    print("\n2. Logging in...")
    headers = {
        "X-CSRF-Token": csrf_token,
        "Content-Type": "application/json"
    }

    login_data = {
        "email": "htperkins@gmail.com",
        "password": "Twiohmld1234!"
    }

    response = session.post(
        f"{BASE_URL}/v1/auth/login",
        headers=headers,
        json=login_data
    )

    if response.status_code == 200:
        auth_data = response.json()
        print(f"   ✓ Login successful!")
        print(f"   User: {auth_data.get('user', {}).get('email')}")
        print(f"   Role: {auth_data.get('user', {}).get('role')}")

        # Store access token if provided
        access_token = auth_data.get('access_token')
        if access_token:
            headers['Authorization'] = f"Bearer {access_token}"
    else:
        print(f"   ✗ Login failed: {response.status_code}")
        print(f"   {response.text}")
        return False

    # Step 3: Test research endpoint with authentication
    print("\n3. Submitting research query...")
    research_query = {
        "query": "What is artificial intelligence and how does it impact society?",
        "use_llm": True,
        "max_results": 10,
        "include_pdfs": False
    }

    response = session.post(
        f"{BASE_URL}/v1/research/query",
        headers=headers,
        json=research_query
    )

    print(f"   Response status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Research submitted successfully!")
        print(f"   Research ID: {result.get('research_id')}")
        print(f"   Status: {result.get('status')}")

        research_id = result.get('research_id')

        # Step 4: Poll for results
        if research_id:
            print("\n4. Polling for results...")
            max_attempts = 120  # Wait up to 120 seconds
            for i in range(max_attempts):
                time.sleep(1)

                response = session.get(
                    f"{BASE_URL}/v1/research/status/{research_id}",
                    headers=headers
                )

                if response.status_code == 200:
                    status_data = response.json()
                    status = status_data.get('status')
                    print(f"   [{i+1}s] Status: {status}", end="")

                    # Show progress if available
                    if 'progress' in status_data:
                        progress = status_data['progress']
                        print(f" - {progress.get('phase', 'unknown')} ({progress.get('percentage', 0)}%)")
                    else:
                        print()

                    if status == 'completed':
                        print("\n   ✓ Research completed!")

                        # Get full results
                        response = session.get(
                            f"{BASE_URL}/v1/research/results/{research_id}",
                            headers=headers
                        )

                        if response.status_code == 200:
                            results = response.json()
                            print(f"   Paradigm: {results.get('paradigm')}")
                            print(f"   Sources found: {len(results.get('sources', []))}")

                            if results.get('answer'):
                                print(f"   Answer preview: {results['answer'][:300]}...")

                        return True
                    elif status == 'failed':
                        print(f"\n   ✗ Research failed: {status_data.get('error')}")
                        return False
                else:
                    print(f"\n   Error checking status: {response.status_code}")
                    print(f"   {response.text}")
                    break

            print("\n   ⚠ Timeout waiting for results")
            return False

        return True
    else:
        print(f"   ✗ Research submission failed:")
        print(f"   {response.text}")
        return False

if __name__ == "__main__":
    success = test_authenticated_research()

    print("\n" + "=" * 60)
    if success:
        print("✅ AUTHENTICATED RESEARCH TEST PASSED!")
    else:
        print("❌ AUTHENTICATED RESEARCH TEST FAILED!")
    print("=" * 60)