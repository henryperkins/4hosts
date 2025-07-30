#!/usr/bin/env python3
import requests
import json

# Test login endpoint
url = "http://localhost:8000/auth/login"

# Try with the password from test_register.py
passwords_to_try = [
    "Test123!Pass",  # From test_register.py
    "testpassword",  # From test_deep_research.py
    "Test123!",
    "password123"
]

for password in passwords_to_try:
    print(f"\nTrying password: {password}")
    data = {
        "email": "test@example.com",
        "password": password
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("SUCCESS! Login successful")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            break
        else:
            print(f"Failed: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
