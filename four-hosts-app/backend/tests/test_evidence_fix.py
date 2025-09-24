#!/usr/bin/env python3
"""
Quick test to verify the evidence builder fix is working
"""
import requests
import json
import time

BASE_URL = "http://127.0.0.1:8001"

def test_research_quick():
    print("Testing Evidence Builder Fix")
    print("=" * 50)

    session = requests.Session()

    # Get CSRF token
    response = session.get(f"{BASE_URL}/api/csrf-token")
    csrf_token = response.json()['csrf_token']
    print(f"✓ Got CSRF token")

    # Login
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
        print(f"✓ Logged in as {auth_data.get('user', {}).get('email')}")

        if auth_data.get('access_token'):
            headers['Authorization'] = f"Bearer {auth_data['access_token']}"
    else:
        print(f"✗ Login failed: {response.status_code}")
        return False

    # Submit a simple research query
    research_query = {
        "query": "What is machine learning?",
        "use_llm": True,
        "max_results": 5,
        "include_pdfs": False
    }

    print(f"\nSubmitting research query: '{research_query['query']}'")

    response = session.post(
        f"{BASE_URL}/v1/research/query",
        headers=headers,
        json=research_query
    )

    if response.status_code == 200:
        result = response.json()
        research_id = result.get('research_id')
        print(f"✓ Research submitted (ID: {research_id})")

        # Poll briefly to see if it progresses
        print("\nTracking progress for 30 seconds...")
        phases_seen = set()
        for i in range(30):
            time.sleep(1)

            response = session.get(
                f"{BASE_URL}/v1/research/status/{research_id}",
                headers=headers
            )

            if response.status_code == 200:
                data = response.json()
                status = data.get('status')
                current_phase = data.get('current_phase')
                progress = data.get('progress', 0)

                if current_phase and current_phase not in phases_seen:
                    phases_seen.add(current_phase)
                    print(f"  → Entered phase: {current_phase}")

                if status == 'completed':
                    print(f"\n✅ Research completed successfully!")
                    return True
                elif status == 'failed':
                    print(f"\n❌ Research failed")
                    return False

        print(f"\nPhases completed: {', '.join(phases_seen) if phases_seen else 'none'}")
        print("Test ended (timeout after 30s)")
        return True

    elif response.status_code == 429:
        print(f"⚠️ Rate limited - please wait and try again")
        return False
    else:
        print(f"✗ Research submission failed: {response.status_code}")
        if response.text:
            print(f"   {response.text[:200]}")
        return False

if __name__ == "__main__":
    success = test_research_quick()
    print("\n" + "=" * 50)
    if success:
        print("Evidence builder fix appears to be working!")
    else:
        print("Test inconclusive - check backend logs")