#!/bin/bash

# Test script for authentication flow with CSRF protection

BASE_URL="http://localhost:8000"
EMAIL="test@example.com"
PASSWORD="testpassword123"

echo "Testing Authentication Flow with CSRF Protection"
echo "================================================"

# Step 1: Get CSRF token
echo -e "\n1. Getting CSRF token..."
CSRF_RESPONSE=$(curl -s -c cookies.txt "$BASE_URL/api/csrf-token")
CSRF_TOKEN=$(echo $CSRF_RESPONSE | jq -r '.csrf_token')
echo "CSRF Token: $CSRF_TOKEN"

# Step 2: Check cookie was set
echo -e "\n2. Checking CSRF cookie..."
CSRF_COOKIE=$(grep csrf_token cookies.txt | awk '{print $7}')
echo "CSRF Cookie: $CSRF_COOKIE"

# Step 3: Verify they match
echo -e "\n3. Verifying token matches cookie..."
if [ "$CSRF_TOKEN" = "$CSRF_COOKIE" ]; then
    echo "✓ CSRF token matches cookie"
else
    echo "✗ CSRF token mismatch!"
    echo "Token: $CSRF_TOKEN"
    echo "Cookie: $CSRF_COOKIE"
fi

# Step 4: Test login with CSRF token
echo -e "\n4. Testing login with CSRF token..."
LOGIN_RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "X-CSRF-Token: $CSRF_TOKEN" \
    -b cookies.txt \
    -c cookies.txt \
    -d "{\"email\": \"$EMAIL\", \"password\": \"$PASSWORD\"}" \
    "$BASE_URL/auth/login" \
    -w "\nHTTP_STATUS:%{http_code}")

HTTP_STATUS=$(echo "$LOGIN_RESPONSE" | grep "HTTP_STATUS:" | cut -d: -f2)
BODY=$(echo "$LOGIN_RESPONSE" | sed '/HTTP_STATUS:/d')

echo "HTTP Status: $HTTP_STATUS"
echo "Response: $BODY"

# Step 5: Test getting the same CSRF token again (should return existing)
echo -e "\n5. Getting CSRF token again (should return existing)..."
CSRF_RESPONSE2=$(curl -s -b cookies.txt -c cookies.txt "$BASE_URL/api/csrf-token")
CSRF_TOKEN2=$(echo $CSRF_RESPONSE2 | jq -r '.csrf_token')
echo "Second CSRF Token: $CSRF_TOKEN2"

if [ "$CSRF_TOKEN" = "$CSRF_TOKEN2" ]; then
    echo "✓ CSRF token correctly reused"
else
    echo "✗ CSRF token was regenerated!"
fi

# Cleanup
rm -f cookies.txt

echo -e "\n================================================"
echo "Test completed"