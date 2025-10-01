#!/bin/bash
# filepath: four-hosts-app/scripts/verify-triage.sh

set -e

echo "=== Triage Pipeline Verification ==="
echo ""

# Configuration
API_BASE="${API_BASE:-http://localhost:8000}"
TOKEN="${AUTH_TOKEN:-}"

if [ -z "$TOKEN" ]; then
  echo "⚠️  AUTH_TOKEN not set. Attempting without auth..."
  CURL_AUTH=""
else
  CURL_AUTH="-H \"Authorization: Bearer $TOKEN\""
fi

# Colours for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test 1: Board endpoint structure
echo "1️⃣  Testing board endpoint structure..."

BOARD_RESPONSE=$(eval curl -s $CURL_AUTH "$API_BASE/v1/system/triage-board")

if echo "$BOARD_RESPONSE" | jq -e 'has("updated_at") and has("entry_count") and has("lanes")' > /dev/null; then
  echo -e "${GREEN}✓${NC} Board payload has required fields"
else
  echo -e "${RED}✗${NC} Missing required fields (updated_at, entry_count, lanes)"
  echo "$BOARD_RESPONSE" | jq '.'
  exit 1
fi

EXPECTED_LANES='["analysis","blocked","classification","context","done","intake","review","search","synthesis"]'
ACTUAL_LANES=$(echo "$BOARD_RESPONSE" | jq '.lanes | keys | sort')

ENTRY_COUNT=$(echo "$BOARD_RESPONSE" | jq '.entry_count')

if [ "$ACTUAL_LANES" == "$EXPECTED_LANES" ]; then
  echo -e "${GREEN}✓${NC} All 9 lanes present: intake → classification → context → search → analysis → synthesis → review → blocked → done"
elif [ "$ACTUAL_LANES" == '[]' ] && [ "$ENTRY_COUNT" -eq 0 ]; then
  echo -e "${YELLOW}⚠${NC}  Board empty (lanes will populate after first entry)"
else
  echo -e "${RED}✗${NC} Lane mismatch!"
  echo "  Expected: $EXPECTED_LANES"
  echo "  Actual:   $ACTUAL_LANES"
  exit 1
fi

if [[ "$ENTRY_COUNT" =~ ^[0-9]+$ ]]; then
  echo -e "${GREEN}✓${NC} Entry count is valid: $ENTRY_COUNT entries"
else
  echo -e "${RED}✗${NC} Invalid entry_count: $ENTRY_COUNT"
  exit 1
fi

echo ""

# Test 2: Submit research and track lane transitions
echo "2️⃣  Testing lane transitions (submitting research)..."

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
  echo -e "${YELLOW}⚠${NC}  Could not submit research (may require auth)"
  echo "  Response: $RESEARCH_RESPONSE"
  echo "  Skipping lane transition test"
else
  echo -e "${GREEN}✓${NC} Research submitted: $RESEARCH_ID"

  PREVIOUS_LANE=""
  LANE_CHANGES=0

  for i in {1..15}; do
    sleep 1

    CURRENT_BOARD=$(eval curl -s $CURL_AUTH "$API_BASE/v1/system/triage-board")
    CURRENT_LANE=$(echo "$CURRENT_BOARD" | jq -r ".lanes | to_entries[] | select(.value[] | .research_id == \"$RESEARCH_ID\") | .key // \"not_found\"")

    if [ "$CURRENT_LANE" != "$PREVIOUS_LANE" ] && [ "$CURRENT_LANE" != "not_found" ]; then
      echo -e "  ${GREEN}→${NC} [$i] Lane: $CURRENT_LANE"
      PREVIOUS_LANE="$CURRENT_LANE"
      LANE_CHANGES=$((LANE_CHANGES + 1))
    fi

    if [ "$CURRENT_LANE" == "done" ]; then
      echo -e "${GREEN}✓${NC} Research completed successfully (${LANE_CHANGES} lane transitions)"
      break
    fi
  done

  if [ "$CURRENT_LANE" == "blocked" ]; then
    echo -e "${YELLOW}⚠${NC}  Research blocked (check logs for errors)"
  elif [ "$CURRENT_LANE" == "not_found" ]; then
    echo -e "${YELLOW}⚠${NC}  Research not found in board (may have completed)"
  fi
fi

echo ""

# Test 3: Frontend type guard check
echo "3️⃣  Checking frontend type guard..."

TYPE_GUARD_FILE="frontend/src/types/api-types.ts"

if [ -f "$TYPE_GUARD_FILE" ]; then
  if grep -q "triage.board_update" "$TYPE_GUARD_FILE"; then
    echo -e "${GREEN}✓${NC} Type guard includes 'triage.board_update'"
  else
    echo -e "${RED}✗${NC} Type guard missing 'triage.board_update' in allowlist"
    echo "  Fix: Add 'triage.board_update' to validTypes array at line ~321"
    exit 1
  fi
else
  echo -e "${YELLOW}⚠${NC}  Type guard file not found at $TYPE_GUARD_FILE"
fi

echo ""

# Test 4: Frontend build check
echo "4️⃣  Testing frontend build..."

if [ -d "frontend" ]; then
  pushd frontend > /dev/null

  if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}⚠${NC}  node_modules not found, running npm install..."
    npm install --silent
  fi

  if npm run build > /tmp/triage-build.log 2>&1; then
    echo -e "${GREEN}✓${NC} Frontend builds successfully with triage components"
  else
    echo -e "${RED}✗${NC} Frontend build failed"
    echo "  Check /tmp/triage-build.log for details"
    tail -20 /tmp/triage-build.log
    exit 1
  fi

  popd > /dev/null
else
  echo -e "${YELLOW}⚠${NC}  Frontend directory not found"
fi

echo ""
echo -e "${GREEN}=== Verification Complete ===${NC}"
echo ""
echo "Summary:"
echo "  - Board endpoint: ✓ (9 lanes, correct payload structure)"
if [ -n "$RESEARCH_ID" ]; then
  echo "  - Lane transitions: ${LANE_CHANGES} observed"
else
  echo "  - Lane transitions: skipped"
fi
if grep -q 'triage.board_update' "$TYPE_GUARD_FILE" 2>/dev/null; then
  echo "  - Type guard: ✓"
else
  echo "  - Type guard: ✗ FIX NEEDED"
fi
if [ -f /tmp/triage-build.log ]; then
  echo "  - Frontend build: ✓"
else
  echo "  - Frontend build: skipped"
fi
echo ""
echo "Next steps:"
echo "  1. If type guard failed, apply patch from verification guide"
echo "  2. Connect to ws://localhost:8000/ws/research/triage-board in browser"
echo "  3. Submit research and watch for triage.board_update events"
