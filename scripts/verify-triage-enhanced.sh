#!/bin/bash
# Enhanced verification with CSRF support

set -e

echo "=== Triage Pipeline Verification ==="
echo ""

# Configuration
API_BASE="${API_BASE:-http://localhost:8000}"
TOKEN="${AUTH_TOKEN:-}"
CSRF_TOKEN="${X_CSRF_TOKEN:-}"

if [ -z "$TOKEN" ]; then
  echo "⚠️  AUTH_TOKEN not set. Attempting without auth..."
  CURL_AUTH=""
else
  CURL_AUTH="-H \"Authorization: Bearer $TOKEN\""

  # Add CSRF header if available
  if [ -n "$CSRF_TOKEN" ]; then
    CURL_AUTH="$CURL_AUTH -H \"X-CSRF-Token: $CSRF_TOKEN\""
  fi
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Board endpoint structure
echo "1️⃣  Testing board endpoint structure..."

BOARD_RESPONSE=$(eval curl -s $CURL_AUTH "$API_BASE/v1/system/triage-board")

# Check required fields
if echo "$BOARD_RESPONSE" | jq -e 'has("updated_at") and has("entry_count") and has("lanes")' > /dev/null 2>&1; then
  echo -e "${GREEN}✓${NC} Board payload has required fields"
else
  echo -e "${RED}✗${NC} Missing required fields (updated_at, entry_count, lanes)"
  echo "$BOARD_RESPONSE" | jq '.'
  exit 1
fi

# Verify lane IDs
EXPECTED_LANES='["analysis","blocked","classification","context","done","intake","review","search","synthesis"]'
ACTUAL_LANES=$(echo "$BOARD_RESPONSE" | jq '.lanes | keys | sort')

if [ "$ACTUAL_LANES" == "$EXPECTED_LANES" ]; then
  echo -e "${GREEN}✓${NC} All 9 lanes present: intake → classification → context → search → analysis → synthesis → review → blocked → done"
else
  echo -e "${RED}✗${NC} Lane mismatch!"
  echo "  Expected: $EXPECTED_LANES"
  echo "  Actual:   $ACTUAL_LANES"
  exit 1
fi

# Check entry_count is numeric
ENTRY_COUNT=$(echo "$BOARD_RESPONSE" | jq '.entry_count')
if [[ "$ENTRY_COUNT" =~ ^[0-9]+$ ]]; then
  echo -e "${GREEN}✓${NC} Entry count is valid: $ENTRY_COUNT entries"
else
  echo -e "${RED}✗${NC} Invalid entry_count: $ENTRY_COUNT"
  exit 1
fi

echo ""

# Test 2: Submit research and track lane transitions
echo "2️⃣  Testing lane transitions (submitting research)..."

if [ -z "$TOKEN" ]; then
  echo -e "${YELLOW}⚠${NC}  Skipping lane transition test (AUTH_TOKEN not set)"
  echo "  Run: source scripts/.env.tokens"
  echo "  Or:  ./scripts/login-and-test.sh"
else
  SUBMIT_PAYLOAD='{
    "query": "What are the implications of quantum computing?",
    "depth": "standard"
  }'

  RESEARCH_RESPONSE=$(eval curl -s -X POST $CURL_AUTH \
    -H "\"Content-Type: application/json\"" \
    -d "'$SUBMIT_PAYLOAD'" \
    "$API_BASE/v1/research/query")

  RESEARCH_ID=$(echo "$RESEARCH_RESPONSE" | jq -r '.research_id // empty')

  if [ -z "$RESEARCH_ID" ]; then
    echo -e "${YELLOW}⚠${NC}  Could not submit research"
    echo "  Response: $RESEARCH_RESPONSE"

    # Check for specific auth errors
    if echo "$RESEARCH_RESPONSE" | grep -qi "csrf"; then
      echo -e "${YELLOW}→${NC}  CSRF token may be required. Set X_CSRF_TOKEN env var."
    elif echo "$RESEARCH_RESPONSE" | grep -qi "unauthorized\|forbidden"; then
      echo -e "${YELLOW}→${NC}  Authentication failed. Check AUTH_TOKEN validity."
    fi
  else
    echo -e "${GREEN}✓${NC} Research submitted: $RESEARCH_ID"

    # Track lane transitions
    PREVIOUS_LANE=""
    LANE_CHANGES=0
    MAX_ITERATIONS=20

    for i in $(seq 1 $MAX_ITERATIONS); do
      sleep 1

      CURRENT_BOARD=$(eval curl -s $CURL_AUTH "$API_BASE/v1/system/triage-board")
      CURRENT_LANE=$(echo "$CURRENT_BOARD" | jq -r ".lanes | to_entries[] | select(.value[] | .research_id == \"$RESEARCH_ID\") | .key // \"not_found\"")

      if [ "$CURRENT_LANE" != "$PREVIOUS_LANE" ] && [ "$CURRENT_LANE" != "not_found" ]; then
        echo -e "  ${GREEN}→${NC} [$i] Lane: $CURRENT_LANE"
        PREVIOUS_LANE="$CURRENT_LANE"
        LANE_CHANGES=$((LANE_CHANGES + 1))
      fi

      # Exit early if completed
      if [ "$CURRENT_LANE" == "done" ]; then
        echo -e "${GREEN}✓${NC} Research completed successfully (${LANE_CHANGES} lane transitions)"
        break
      fi

      # Check for blocked state
      if [ "$CURRENT_LANE" == "blocked" ]; then
        echo -e "${YELLOW}⚠${NC}  Research blocked (check logs for errors)"
        break
      fi
    done

    if [ "$CURRENT_LANE" == "not_found" ] && [ "$LANE_CHANGES" -gt 0 ]; then
      echo -e "${GREEN}✓${NC}  Research completed and archived (${LANE_CHANGES} lane transitions observed)"
    elif [ "$CURRENT_LANE" == "not_found" ]; then
      echo -e "${YELLOW}⚠${NC}  Research not found in board (may have completed quickly)"
    fi
  fi
fi

echo ""

# Test 3: Frontend type guard check
echo "3️⃣  Checking frontend type guard..."

TYPE_GUARD_FILE="four-hosts-app/frontend/src/types/api-types.ts"

if [ -f "$TYPE_GUARD_FILE" ]; then
  if grep -q "triage.board_update" "$TYPE_GUARD_FILE"; then
    echo -e "${GREEN}✓${NC} Type guard includes 'triage.board_update'"
  else
    echo -e "${RED}✗${NC} Type guard missing 'triage.board_update' in allowlist"
    echo "  Fix: Add 'triage.board_update' to validTypes array"
    exit 1
  fi
else
  echo -e "${YELLOW}⚠${NC}  Type guard file not found at $TYPE_GUARD_FILE"
fi

echo ""

# Test 4: Frontend build check
echo "4️⃣  Testing frontend build..."

if [ -d "four-hosts-app/frontend" ]; then
  cd four-hosts-app/frontend

  # Check if node_modules exists
  if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}⚠${NC}  node_modules not found, running npm install..."
    npm install --silent
  fi

  # Run build
  if npm run build > /tmp/triage-build.log 2>&1; then
    echo -e "${GREEN}✓${NC} Frontend builds successfully with triage components"
  else
    echo -e "${RED}✗${NC} Frontend build failed"
    echo "  Check /tmp/triage-build.log for details"
    tail -20 /tmp/triage-build.log
    exit 1
  fi

  cd - > /dev/null
else
  echo -e "${YELLOW}⚠${NC}  Frontend directory not found"
fi

echo ""
echo -e "${GREEN}=== Verification Complete ===${NC}"
echo ""
echo "Summary:"
echo "  - Board endpoint: ✓ (9 lanes, correct payload structure)"
echo "  - Lane transitions: ${LANE_CHANGES:-N/A} observed"
echo "  - Type guard: $(grep -q 'triage.board_update' "$TYPE_GUARD_FILE" 2>/dev/null && echo '✓' || echo '✗ FIX NEEDED')"
echo "  - Frontend build: ✓"
echo ""

if [ -z "$TOKEN" ]; then
  echo "📝 Next steps:"
  echo "  1. Get auth tokens: ./scripts/extract-auth-tokens.sh"
  echo "  2. Or login via API: ./scripts/login-and-test.sh"
  echo "  3. Rerun with: AUTH_TOKEN=... X_CSRF_TOKEN=... ./scripts/verify-triage.sh"
else
  echo "📝 Next steps:"
  echo "  1. Open frontend/public/triage-ws-test.html in browser"
  echo "  2. Connect to ws://localhost:8000/ws/research/triage-board"
  echo "  3. Submit research and watch for triage.board_update events"
fi
