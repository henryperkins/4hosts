#!/bin/bash
# Login via API and run triage verification

set -e

API_BASE="${API_BASE:-http://localhost:8000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=== API Login & Token Extraction ==="
echo ""

# Prompt for credentials
read -p "Email: " EMAIL
read -sp "Password: " PASSWORD
echo ""

# Login request
echo "Attempting login..."
LOGIN_RESPONSE=$(curl -s -X POST "$API_BASE/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"$EMAIL\",\"password\":\"$PASSWORD\"}")

# Extract tokens
ACCESS_TOKEN=$(echo "$LOGIN_RESPONSE" | jq -r '.access_token // empty')
CSRF_TOKEN=$(echo "$LOGIN_RESPONSE" | jq -r '.csrf_token // empty')

if [ -z "$ACCESS_TOKEN" ]; then
  echo -e "${YELLOW}⚠${NC} Login failed or token not in response"
  echo "Response: $LOGIN_RESPONSE"
  exit 1
fi

echo -e "${GREEN}✓${NC} Login successful!"
echo ""

# Export tokens for this session
export AUTH_TOKEN="$ACCESS_TOKEN"
export X_CSRF_TOKEN="${CSRF_TOKEN:-$ACCESS_TOKEN}"

# Save to file for persistence
cat > "$SCRIPT_DIR/.env.tokens" << EOF
# Generated on $(date)
export AUTH_TOKEN="$ACCESS_TOKEN"
export X_CSRF_TOKEN="${CSRF_TOKEN:-$ACCESS_TOKEN}"
EOF

echo "Tokens saved to $SCRIPT_DIR/.env.tokens"
echo "To use in another shell: source $SCRIPT_DIR/.env.tokens"
echo ""

# Display tokens for manual export
echo "=== Token Values ==="
echo "export AUTH_TOKEN=\"$ACCESS_TOKEN\""
echo "export X_CSRF_TOKEN=\"${CSRF_TOKEN:-$ACCESS_TOKEN}\""
echo ""

# Run verification if script exists
if [ -f "$SCRIPT_DIR/verify-triage.sh" ]; then
  echo "=== Running Triage Verification ==="
  "$SCRIPT_DIR/verify-triage.sh"
else
  echo -e "${YELLOW}⚠${NC} verify-triage.sh not found, skipping verification"
fi
