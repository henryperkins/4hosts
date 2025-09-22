#!/usr/bin/env python3
"""
Simple test to verify research API is working
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8001"

def test_research():
    # Create session to maintain cookies
    session = requests.Session()

    print("1. Getting CSRF token...")
    response = session.get(f"{BASE_URL}/api/csrf-token")
    if response.status_code != 200:
        print(f"Failed to get CSRF token: {response.status_code}")
        return False

    csrf_data = response.json()
    csrf_token = csrf_data['csrf_token']
    print(f"   Got CSRF token: {csrf_token}")

    # Check cookies
    print(f"   Cookies: {session.cookies.get_dict()}")

    print("\n2. Creating session...")
    headers = {
        "X-CSRF-Token": csrf_token,
        "Content-Type": "application/json"
    }
    response = session.post(f"{BASE_URL}/api/session/create", headers=headers, json={})
    if response.status_code == 200:
        session_data = response.json()
        print("   Session created successfully")
        print(f"   Session data: {session_data}")

    print("\n3. Testing research endpoint...")
    research_query = {
        "query": "What is quantum computing?",
        "use_llm": False,  # Start without LLM to test basic functionality
        "max_results": 5,
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

        # Pretty print the response
        print("\n   Full response:")
        print(json.dumps(result, indent=2)[:500])
        return True
    else:
        print(f"   ✗ Research failed:")
        print(f"   {response.text}")
        return False

if __name__ == "__main__":
    success = test_research()
    if success:
        print("\n✅ Test PASSED!")
    else:
        print("\n❌ Test FAILED!")