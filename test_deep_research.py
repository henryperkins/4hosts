#!/usr/bin/env python3
"""Test script for the deep research endpoint"""

import requests
import json

# API configuration
API_BASE_URL = "http://localhost:8000"

# Test credentials (assuming these exist in your test setup)
TEST_EMAIL = "test@example.com"
TEST_PASSWORD = "testpassword"

def login():
    """Login and get auth token"""
    response = requests.post(
        f"{API_BASE_URL}/auth/login",
        json={"email": TEST_EMAIL, "password": TEST_PASSWORD}
    )
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        print(f"Login failed: {response.status_code} - {response.text}")
        return None

def test_deep_research(token):
    """Test the deep research endpoint"""
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test 1: Valid request
    print("\n=== Test 1: Valid deep research request ===")
    valid_payload = {
        "query": "What are the latest breakthroughs in quantum computing?",
        "paradigm": "bernard",
        "search_context_size": "medium"
    }
    response = requests.post(
        f"{API_BASE_URL}/research/deep",
        json=valid_payload,
        headers=headers
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test 2: Missing query field
    print("\n=== Test 2: Missing query field ===")
    invalid_payload = {
        "paradigm": "bernard"
    }
    response = requests.post(
        f"{API_BASE_URL}/research/deep",
        json=invalid_payload,
        headers=headers
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test 3: Query too short
    print("\n=== Test 3: Query too short ===")
    short_query_payload = {
        "query": "AI?",
        "paradigm": "bernard"
    }
    response = requests.post(
        f"{API_BASE_URL}/research/deep",
        json=short_query_payload,
        headers=headers
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test 4: Invalid search context size
    print("\n=== Test 4: Invalid search context size ===")
    invalid_size_payload = {
        "query": "Tell me about artificial intelligence advancements",
        "search_context_size": "extra-large"
    }
    response = requests.post(
        f"{API_BASE_URL}/research/deep",
        json=invalid_size_payload,
        headers=headers
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    print("Testing Deep Research Endpoint...")
    
    # First check if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code != 200:
            print("API is not healthy. Please ensure the backend is running.")
            exit(1)
    except requests.exceptions.ConnectionError:
        print(f"Cannot connect to API at {API_BASE_URL}. Please ensure the backend is running.")
        exit(1)
    
    # Login and get token
    token = login()
    if not token:
        print("Cannot proceed without authentication token.")
        print("Please ensure test user exists or create one with:")
        print('curl -X POST "http://localhost:8000/auth/register" -H "Content-Type: application/json" -d \'{"username": "test", "email": "test@example.com", "password": "testpassword", "role": "pro"}\'')
        exit(1)
    
    # Run tests
    test_deep_research(token)