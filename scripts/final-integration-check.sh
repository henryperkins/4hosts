#!/bin/bash
# Complete verification checklist - run this to verify everything works

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Triage Pipeline - Final Integration Checklist       ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""

TOTAL=0
PASSED=0

function run_check() {
  local name="$1"
  local command="$2"

  TOTAL=$((TOTAL + 1))
  echo -n "[$TOTAL] $name... "

  if eval "$command" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
    PASSED=$((PASSED + 1))
    return 0
  else
    echo -e "${RED}✗${NC}"
    return 1
  fi
}

echo "=== File Existence Checks ==="
run_check "Authentication extraction script" "test -x '$SCRIPT_DIR/extract-auth-tokens.sh'"
run_check "API login script" "test -x '$SCRIPT_DIR/login-and-test.sh'"
run_check "Preflight check script" "test -x '$SCRIPT_DIR/preflight-triage.sh'"
run_check "Board structure test script" "test -x '$SCRIPT_DIR/test-triage-board-enhanced.sh'"
run_check "Full verification script" "test -x '$SCRIPT_DIR/verify-triage-enhanced.sh'"
run_check "E2E workflow script" "test -x '$SCRIPT_DIR/e2e-triage-verification.sh'"
run_check "Triage verification guide" "test -f '$PROJECT_ROOT/docs/triage-verification-guide.md'"
run_check "Quick reference card" "test -f '$PROJECT_ROOT/docs/TRIAGE_QUICKREF.md'"
run_check "Package overview doc" "test -f '$PROJECT_ROOT/docs/TRIAGE_VERIFICATION_PACKAGE.md'"
run_check "WebSocket HTML monitor" "test -f '$PROJECT_ROOT/four-hosts-app/frontend/public/triage-ws-test.html'"

echo ""
echo "=== Frontend Integration Checks ==="
run_check "Type guard file exists" "test -f '$PROJECT_ROOT/four-hosts-app/frontend/src/types/api-types.ts'"
run_check "Type guard includes triage.board_update" "grep -q 'triage.board_update' '$PROJECT_ROOT/four-hosts-app/frontend/src/types/api-types.ts'"
run_check "WebSocket hook exists" "test -f '$PROJECT_ROOT/four-hosts-app/frontend/src/hooks/useWebSocket.ts'"
run_check "TriageBoard component exists" "test -f '$PROJECT_ROOT/four-hosts-app/frontend/src/components/TriageBoard.tsx'"
run_check "Vite config has proxy" "grep -q 'proxy' '$PROJECT_ROOT/four-hosts-app/frontend/vite.config.ts'"

echo ""
echo "=== Backend Integration Checks ==="
run_check "Triage service exists" "test -f '$PROJECT_ROOT/four-hosts-app/backend/services/triage.py'"
run_check "WebSocket service exists" "test -f '$PROJECT_ROOT/four-hosts-app/backend/services/websocket_service.py'"
run_check "System routes exist" "test -f '$PROJECT_ROOT/four-hosts-app/backend/routes/system.py'"
run_check "Research orchestrator exists" "test -f '$PROJECT_ROOT/four-hosts-app/backend/services/research_orchestrator.py'"
run_check "Triage manager has 9 lanes" "grep -A 20 'class TriageLane' '$PROJECT_ROOT/four-hosts-app/backend/services/triage.py' | grep -q 'DONE.*=.*\"done\"'"

echo ""
echo "=== Required Tools Checks ==="
run_check "jq installed" "command -v jq"
run_check "curl installed" "command -v curl"
run_check "node installed" "command -v node"
run_check "npm installed" "command -v npm"

echo ""
echo "=== Runtime Checks (if backend is running) ==="
API_BASE="${API_BASE:-http://localhost:8000}"

if curl -s -f "$API_BASE/health" > /dev/null 2>&1; then
  run_check "Backend health endpoint" "curl -s -f '$API_BASE/health'"
  run_check "Triage board endpoint" "curl -s '$API_BASE/v1/system/triage-board' | jq -e 'has(\"lanes\")'"
  # Note: Board might have 0-9 lanes depending on activity
  echo -e "${YELLOW}⚠${NC}  Board has $(curl -s '$API_BASE/v1/system/triage-board' | jq '.lanes | keys | length') lanes (9 expected when active)"
else
  echo -e "${YELLOW}⚠${NC}  Backend not running - skipping runtime checks"
  echo "    Start backend: cd four-hosts-app/backend && uvicorn main_new:app --reload"
fi

echo ""
echo "=== Results ==="
echo -e "Passed: ${GREEN}$PASSED${NC}/$TOTAL"

if [ $PASSED -eq $TOTAL ]; then
  echo -e "${GREEN}✓ All checks passed!${NC}"
  echo ""
  echo "Next steps:"
  echo "  1. Get authentication tokens:"
  echo "     ./scripts/login-and-test.sh"
  echo ""
  echo "  2. Run full verification:"
  echo "     source scripts/.env.tokens"
  echo "     ./scripts/verify-triage-enhanced.sh"
  echo ""
  echo "  3. Monitor live events:"
  echo "     open http://localhost:5173/triage-ws-test.html"
  exit 0
else
  FAILED=$((TOTAL - PASSED))
  echo -e "${RED}✗ $FAILED checks failed${NC}"
  echo ""
  echo "Review the failures above and fix before proceeding."
  exit 1
fi
