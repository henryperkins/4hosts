#!/bin/bash
# Pre-flight checks before triage verification

set -e

API_BASE="${API_BASE:-http://localhost:8000}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=== Pre-flight Checks ==="
echo ""

# Check 1: Backend health
echo -n "1. Backend health check... "
if curl -s -f "$API_BASE/health" > /dev/null 2>&1; then
  echo -e "${GREEN}✓${NC}"
else
  echo -e "${RED}✗${NC}"
  echo ""
  echo "Backend is not responding at $API_BASE"
  echo ""
  echo "Start the backend:"
  echo "  cd four-hosts-app/backend"
  echo "  uvicorn main_new:app --reload"
  echo ""
  echo "Or use the convenience script:"
  echo "  ./start-app.sh"
  exit 1
fi

# Check 2: Frontend dev server (optional)
echo -n "2. Frontend dev server... "
if curl -s -f "http://localhost:5173" > /dev/null 2>&1; then
  echo -e "${GREEN}✓${NC}"
else
  echo -e "${YELLOW}⚠${NC} (not required for API tests)"
fi

# Check 3: Triage endpoint
echo -n "3. Triage endpoint... "
TRIAGE_RESPONSE=$(curl -s "$API_BASE/v1/system/triage-board")
if echo "$TRIAGE_RESPONSE" | jq -e 'has("lanes")' > /dev/null 2>&1; then
  echo -e "${GREEN}✓${NC}"
else
  echo -e "${RED}✗${NC}"
  echo "  Response: $TRIAGE_RESPONSE"
  exit 1
fi

# Check 4: Frontend type guard file
echo -n "4. Frontend type guard... "
if [ -f "four-hosts-app/frontend/src/types/api-types.ts" ]; then
  if grep -q "triage.board_update" "four-hosts-app/frontend/src/types/api-types.ts"; then
    echo -e "${GREEN}✓${NC}"
  else
    echo -e "${YELLOW}⚠${NC} (type guard missing triage.board_update)"
  fi
else
  echo -e "${YELLOW}⚠${NC} (file not found)"
fi

# Check 5: Required tools
echo -n "5. Required tools (jq, curl)... "
if command -v jq > /dev/null && command -v curl > /dev/null; then
  echo -e "${GREEN}✓${NC}"
else
  echo -e "${RED}✗${NC}"
  echo "  Install: sudo apt-get install jq curl"
  exit 1
fi

echo ""
echo -e "${GREEN}All pre-flight checks passed!${NC}"
echo ""
echo "Ready to run:"
echo "  ./scripts/test-triage-board-enhanced.sh"
echo "  ./scripts/verify-triage-enhanced.sh"
echo "  ./scripts/e2e-triage-verification.sh"
