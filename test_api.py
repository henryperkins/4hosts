import requests
import json

# Start a session to maintain cookies
s = requests.Session()
base_url = "http://localhost:8000/v1"

# Skip CSRF check (for testing only)
headers = {"X-CSRF-Bypass": "test"}

print("1. Testing server health...")
r = s.get(f"{base_url}/system/limits")
print(f"Server status: {r.status_code}")

print("\n2. Creating test user (FREE tier)...")
# Try to register
reg_data = {
    "username": "testfree",
    "email": "testfree@example.com", 
    "password": "TestPass123!"
}

# First try to login in case user exists
print("Attempting login...")
r = s.post(f"{base_url}/auth/login", 
    json={"email": reg_data["email"], "password": reg_data["password"]},
    headers=headers)

if r.status_code != 200:
    print("Login failed, trying registration...")
    r = s.post(f"{base_url}/auth/register", json=reg_data, headers=headers)
    if r.status_code == 200:
        print("Registration successful")
        auth_data = r.json()
    else:
        print(f"Registration failed: {r.status_code}")
        print(r.text)
        exit(1)
else:
    print("Login successful")
    auth_data = r.json()

# Get the token
token = auth_data.get("access_token")
if token:
    headers["Authorization"] = f"Bearer {token}"
    print(f"Got token: {token[:20]}...")

print("\n3. Checking user info...")
r = s.get(f"{base_url}/auth/user", headers=headers)
if r.status_code == 200:
    user_info = r.json()
    print(f"User role: {user_info.get('role', 'unknown')}")
    print(f"User email: {user_info.get('email', 'unknown')}")

print("\n4. Testing DEEP RESEARCH submission (should work for FREE users now)...")
deep_research_data = {
    "query": "What are the latest breakthroughs in renewable energy?",
    "search_context_size": "medium"
}

r = s.post(f"{base_url}/research/deep", json=deep_research_data, headers=headers)
print(f"Deep research response: {r.status_code}")
if r.status_code == 200:
    print("✅ SUCCESS! Deep research accessible for FREE users!")
    result = r.json()
    print(f"Research ID: {result.get('research_id')}")
elif r.status_code == 403:
    print("❌ FAILED - Still restricted to PRO users")
    print(r.text)
else:
    print(f"Other error: {r.text}")
