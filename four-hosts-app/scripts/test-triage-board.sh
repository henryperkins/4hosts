#!/bin/bash
# filepath: four-hosts-app/scripts/test-triage-board.sh

set -e

API_BASE="${API_BASE:-http://localhost:8000}"
TOKEN="${AUTH_TOKEN:-}"
CURL_AUTH="${TOKEN:+-H \"Authorization: Bearer $TOKEN\"}"

echo "=== Board Snapshot Tests ==="

RESPONSE=$(eval curl -s $CURL_AUTH "$API_BASE/v1/system/triage-board")

echo -n "Test 1: Response structure... "
jq -e 'has("updated_at") and has("entry_count") and has("lanes")' <<< "$RESPONSE" > /dev/null
echo "✓"

echo -n "Test 2: Lane count (expect 9 when populated)... "
LANE_COUNT=$(jq '.lanes | keys | length' <<< "$RESPONSE")
ENTRY_COUNT=$(jq '.entry_count' <<< "$RESPONSE")
if [ "$LANE_COUNT" -eq 9 ]; then
  echo "✓"
elif [ "$LANE_COUNT" -eq 0 ] && [ "$ENTRY_COUNT" -eq 0 ]; then
  echo "⚠ empty board"
else
  echo "✗ Got $LANE_COUNT" && exit 1
fi

echo -n "Test 3: Lane IDs... "
if [ "$LANE_COUNT" -eq 0 ] && [ "$ENTRY_COUNT" -eq 0 ]; then
  echo "⚠ skipped (no lanes yet)"
else
  jq -e '.lanes | has("intake") and has("classification") and has("context") and has("search") and has("analysis") and has("synthesis") and has("review") and has("blocked") and has("done")' <<< "$RESPONSE" > /dev/null
  echo "✓"
fi

echo -n "Test 4: Entry count is number... "
jq -e '.entry_count | type == "number"' <<< "$RESPONSE" > /dev/null
echo "✓"

echo -n "Test 5: Updated timestamp format... "
TIMESTAMP=$(jq -r '.updated_at' <<< "$RESPONSE")
if date -d "$TIMESTAMP" > /dev/null 2>&1; then
  echo "✓"
else
  echo "✗ Invalid ISO 8601" && exit 1
fi

echo -n "Test 6: No totals field... "
jq -e 'has("totals") | not' <<< "$RESPONSE" > /dev/null
echo "✓"

echo -n "Test 7: All lanes are arrays... "
jq -e '.lanes | to_entries | all(.value | type == "array")' <<< "$RESPONSE" > /dev/null
echo "✓"

echo ""
echo "All tests passed! ✓"
echo ""
echo "Sample board state:"
jq '{
  updated_at,
  entry_count,
  lane_summary: (.lanes | map_values(length))
}' <<< "$RESPONSE"
