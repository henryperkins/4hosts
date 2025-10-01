#!/bin/bash
# Enhanced board snapshot tests with auth support

set -e

API_BASE="${API_BASE:-http://localhost:8000}"
TOKEN="${AUTH_TOKEN:-}"
CSRF_TOKEN="${X_CSRF_TOKEN:-}"

if [ -z "$TOKEN" ]; then
  CURL_AUTH=""
else
  CURL_AUTH="-H \"Authorization: Bearer $TOKEN\""

  if [ -n "$CSRF_TOKEN" ]; then
    CURL_AUTH="$CURL_AUTH -H \"X-CSRF-Token: $CSRF_TOKEN\""
  fi
fi

echo "=== Board Snapshot Tests ==="

# Test 1: Response structure
echo -n "Test 1: Response structure... "
RESPONSE=$(eval curl -s $CURL_AUTH "$API_BASE/v1/system/triage-board")

if ! echo "$RESPONSE" | jq -e 'has("updated_at") and has("entry_count") and has("lanes")' > /dev/null 2>&1; then
  echo "✗"
  echo "Response: $RESPONSE"
  exit 1
fi
echo "✓"

# Test 2: Lane count
echo -n "Test 2: Lane count (expect 9)... "
LANE_COUNT=$(echo "$RESPONSE" | jq '.lanes | keys | length')
if [ "$LANE_COUNT" -eq 9 ]; then
  echo "✓"
else
  echo "✗ Got $LANE_COUNT"
  exit 1
fi

# Test 3: Lane IDs
echo -n "Test 3: Lane IDs... "
if echo "$RESPONSE" | jq -e '.lanes | has("intake") and has("classification") and has("context") and has("search") and has("analysis") and has("synthesis") and has("review") and has("blocked") and has("done")' > /dev/null 2>&1; then
  echo "✓"
else
  echo "✗"
  exit 1
fi

# Test 4: Entry count type
echo -n "Test 4: Entry count is number... "
if echo "$RESPONSE" | jq -e '.entry_count | type == "number"' > /dev/null 2>&1; then
  echo "✓"
else
  echo "✗"
  exit 1
fi

# Test 5: Updated timestamp
echo -n "Test 5: Updated timestamp format... "
TIMESTAMP=$(echo "$RESPONSE" | jq -r '.updated_at')
if date -d "$TIMESTAMP" > /dev/null 2>&1; then
  echo "✓"
else
  echo "✗ Invalid ISO 8601"
fi

# Test 6: No totals field (should NOT exist)
echo -n "Test 6: No totals field... "
if echo "$RESPONSE" | jq -e 'has("totals") | not' > /dev/null 2>&1; then
  echo "✓"
else
  echo "✗"
  exit 1
fi

# Test 7: Lanes are arrays
echo -n "Test 7: All lanes are arrays... "
if echo "$RESPONSE" | jq -e '.lanes | to_entries | all(.value | type == "array")' > /dev/null 2>&1; then
  echo "✓"
else
  echo "✗"
  exit 1
fi

echo ""
echo "All tests passed! ✓"
echo ""
echo "Sample board state:"
echo "$RESPONSE" | jq '{
  updated_at,
  entry_count,
  lane_summary: (.lanes | map_values(length))
}'
