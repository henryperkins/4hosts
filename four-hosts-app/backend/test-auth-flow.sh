#!/bin/bash

# Test authentication flow with CSRF protection

API_URL="${API_URL:-http://localhost:8000}"
echo "Testing authentication flow at: $API_URL"

# 1. Get CSRF token
echo -e "\n1. Getting CSRF token..."
CSRF_RESPONSE=$(curl -s -i -X GET "$API_URL/api/csrf-token" \
  -H "Accept: application/json" \
  -c cookies.txt)

CSRF_TOKEN=$(echo "$CSRF_RESPONSE" | grep -oP '{"csrf_token":"\K[^"]+')
echo "CSRF Token: $CSRF_TOKEN"

# Check if we got a valid response
if [ -z "$CSRF_TOKEN" ]; then
  echo "ERROR: Failed to get CSRF token"
  echo "Response:"
  echo "$CSRF_RESPONSE"
  exit 1
fi

# 2. Register a test user (might fail if already exists)
echo -e "\n2. Registering test user..."
REGISTER_RESPONSE=$(curl -s -X POST "$API_URL/auth/register" \
  -H "Content-Type: application/json" \
  -H "X-CSRF-Token: $CSRF_TOKEN" \
  -b cookies.txt \
  -c cookies.txt \
  -d '{
    "email": "test@example.com",
    "password": "TestPassword123!",
    "full_name": "Test User"
  }')
echo "Register response: $REGISTER_RESPONSE"

# 3. Login with the test user
echo -e "\n3. Logging in..."
LOGIN_RESPONSE=$(curl -s -X POST "$API_URL/auth/login" \
  -H "Content-Type: application/json" \
  -H "X-CSRF-Token: $CSRF_TOKEN" \
  -b cookies.txt \
  -c cookies.txt \
  -d '{
    "email": "test@example.com",
    "password": "TestPassword123!"
  }')
echo "Login response: $LOGIN_RESPONSE"

# 4. Get user info (requires authentication)
echo -e "\n4. Getting user info..."
USER_RESPONSE=$(curl -s -X GET "$API_URL/auth/user" \
  -H "X-CSRF-Token: $CSRF_TOKEN" \
  -b cookies.txt)
echo "User info response: $USER_RESPONSE"

# Clean up
rm -f cookies.txt

echo -e "\nTest completed!"