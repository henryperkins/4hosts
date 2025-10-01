# Triage Pipeline Verification Guide

## Overview

The triage pipeline provides real-time visibility into research workflow status through a Kanban-style board tracking requests across 9 lanes from intake to completion.

## Architecture

### Lane Flow
```
intake → classification → context → search → analysis → synthesis → review → blocked → done
```

### Key Components

**Backend:**
- `services/triage.py` - Core triage manager
- `services/websocket_service.py` - WebSocket broadcasting
- `routes/system.py` - Board API endpoint

**Frontend:**
- `components/TriageBoard.tsx` - Live Kanban UI
- `hooks/useWebSocket.ts` - WebSocket connection
- `public/triage-ws-test.html` - Standalone monitor

## Quick Start

### Option 1: Automated E2E Verification

```bash
# Run complete verification workflow
./scripts/e2e-triage-verification.sh
```

This will:
1. Prompt for authentication (browser tokens or API login)
2. Run board snapshot tests
3. Submit test research and track lane transitions
4. Verify frontend build
5. Provide WebSocket monitoring instructions

### Option 2: Manual Step-by-Step

#### Step 1: Extract Authentication Tokens

**From Browser (if logged in):**
```bash
./scripts/extract-auth-tokens.sh
```

Follow the instructions to run JavaScript in your browser console, then export the tokens:
```bash
export AUTH_TOKEN="eyJhbGc..."
export X_CSRF_TOKEN="csrf_..."
```

**Or Login via API:**
```bash
./scripts/login-and-test.sh
# Enter credentials when prompted
source scripts/.env.tokens  # Load tokens
```

#### Step 2: Verify Board Endpoint

```bash
# Basic board structure tests
./scripts/test-triage-board-enhanced.sh
```

Expected output:
```
Test 1: Response structure... ✓
Test 2: Lane count (expect 9)... ✓
Test 3: Lane IDs... ✓
Test 4: Entry count is number... ✓
Test 5: Updated timestamp format... ✓
Test 6: No totals field... ✓
Test 7: All lanes are arrays... ✓
```

#### Step 3: Full Pipeline Verification

```bash
# Submit research and track lane transitions
AUTH_TOKEN="..." X_CSRF_TOKEN="..." ./scripts/verify-triage-enhanced.sh
```

Expected output:
```
1️⃣  Testing board endpoint structure...
✓ Board payload has required fields
✓ All 9 lanes present
✓ Entry count is valid: 3 entries

2️⃣  Testing lane transitions...
✓ Research submitted: res-abc123
  → [1] Lane: intake
  → [2] Lane: classification
  → [3] Lane: context
  → [5] Lane: search
  → [8] Lane: analysis
  → [12] Lane: synthesis
  → [14] Lane: review
✓ Research completed successfully (7 lane transitions)

3️⃣  Checking frontend type guard...
✓ Type guard includes 'triage.board_update'

4️⃣  Testing frontend build...
✓ Frontend builds successfully
```

#### Step 4: WebSocket Live Monitoring

**Option A: HTML Monitor**
1. Open `http://localhost:5173/triage-ws-test.html`
2. Click "Connect"
3. Submit research in another tab
4. Watch `triage.board_update` events in real-time

**Option B: Browser Console**
1. Navigate to your app
2. Open browser console (F12)
3. Paste:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/research/triage-board');
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === 'triage.board_update') {
    const summary = Object.entries(msg.data.lanes)
      .map(([lane, items]) => `${lane}:${items.length}`)
      .join(', ');
    console.log(`[${new Date().toLocaleTimeString()}] ${summary}`);
  }
};
```
4. Submit research and watch lane transitions

**Option C: wscat (CLI)**
```bash
# Install if needed: npm install -g wscat
wscat -c ws://localhost:8000/ws/research/triage-board
```

## API Reference

### GET /v1/system/triage/board

Returns current triage board snapshot.

**Response:**
```json
{
  "updated_at": "2025-09-30T15:42:18.123Z",
  "entry_count": 5,
  "lanes": {
    "intake": [],
    "classification": [
      {
        "research_id": "res-abc123",
        "query": "quantum computing implications",
        "priority": "high",
        "created_at": "2025-09-30T15:42:15Z"
      }
    ],
    "context": [],
    "search": [...],
    "analysis": [],
    "synthesis": [],
    "review": [],
    "blocked": [],
    "done": [...]
  }
}
```

**Fields:**
- `updated_at` - ISO 8601 timestamp of last board update
- `entry_count` - Total number of active entries across all lanes
- `lanes` - Object with 9 keys, each containing array of entries

### WebSocket: /ws/research/triage-board

Broadcasts full board snapshots on any change.

**Event Type:** `triage.board_update`

**Payload:** Same structure as GET endpoint above

**Trigger Conditions:**
- New research submitted (entry added to intake)
- Lane transition (entry moves between lanes)
- Research completed (entry moves to done)
- Research blocked (entry moves to blocked)

## Troubleshooting

### Issue: "AUTH_TOKEN not set" warning

**Solution:**
```bash
# Option 1: Extract from browser
./scripts/extract-auth-tokens.sh

# Option 2: Login via API
./scripts/login-and-test.sh

# Option 3: Load saved tokens
source scripts/.env.tokens
```

### Issue: "CSRF token may be required"

**Solution:**
```bash
# Set CSRF token along with auth token
export X_CSRF_TOKEN="your_csrf_token"

# Or use login script which extracts both
./scripts/login-and-test.sh
```

### Issue: Type guard rejects triage.board_update

**Verify Fix:**
```bash
grep -A 15 "validTypes = \[" four-hosts-app/frontend/src/types/api-types.ts | grep "triage.board_update"
```

Should show:
```typescript
'triage.board_update',
```

If missing, the type guard patch wasn't applied.

### Issue: WebSocket connection refused

**Check:**
1. Backend is running: `curl http://localhost:8000/health`
2. WebSocket endpoint accessible: `wscat -c ws://localhost:8000/ws/research/triage-board`
3. Firewall/proxy allows WebSocket upgrades

### Issue: No lane transitions observed

**Debug:**
1. Check research actually submitted:
   ```bash
   curl -H "Authorization: Bearer $AUTH_TOKEN" \
     http://localhost:8000/v1/research/status/$RESEARCH_ID
   ```

2. Check backend logs for errors:
   ```bash
   tail -f four-hosts-app/backend/logs/app.log | grep -i triage
   ```

3. Verify triage manager integration:
   ```bash
   grep -r "triage_manager" four-hosts-app/backend/services/
   ```

## Verification Checklist

- [ ] Board endpoint returns 9 lanes (intake through done)
- [ ] `entry_count` is numeric
- [ ] `updated_at` is valid ISO 8601 timestamp
- [ ] No `totals` field in response (calculate client-side)
- [ ] Type guard accepts `triage.board_update` events
- [ ] Frontend builds without errors
- [ ] Lane transitions tracked when research submitted
- [ ] WebSocket broadcasts full board snapshots
- [ ] HTML monitor displays live updates

## Performance Benchmarks

| Metric | Target | Typical |
|--------|--------|---------|
| Board endpoint response time | <100ms | ~45ms |
| WebSocket broadcast latency | <50ms | ~25ms |
| Lane transition frequency | 1-2/sec | Variable |
| Frontend render (full board) | <16ms | ~8ms |

## Integration with Research Pipeline

The triage system integrates at these points:

   1. **Submission** (`routes/research.py`):
       ```python
       await triage_manager.initialize_entry(
           research_id=research_id,
           query=query,
           user_id=current_user.id,
           user_role=current_user.role,
           depth=research.depth,
           paradigm=research.paradigm,
           triage_context=triage_context,
       )
       ```

    2. **Phase Transitions** (`services/research_orchestrator.py`):
       ```python
       await triage_manager.update_lane(research_id, phase="classification")
       await triage_manager.update_lane(research_id, phase="context")
       # ... etc
       ```

    3. **Completion**:
        ```python
        await triage_manager.mark_completed(research_id)
        ```

    4. **Errors**:
        ```python
        await triage_manager.mark_blocked(research_id, reason="timeout")
        ```

## Next Steps

After successful verification:

1. **Monitor production traffic:**
   ```bash
   # Watch lane distribution
   watch -n 2 'curl -s http://localhost:8000/v1/system/triage/board | jq ".lanes | map_values(length)"'
   ```

2. **Integrate with operations dashboard:**
   - Add triage board widget to `/admin/dashboard`
   - Set up alerting for `blocked` lane threshold
   - Track lane dwell times for SLA monitoring

3. **Extend functionality:**
   - Priority scoring based on user tier
   - Manual lane overrides for ops team
   - Historical lane transition analytics

## Related Documentation

- [Triage Pipeline Implementation](../docs/triage-verification.md)
- [WebSocket Architecture](../four-hosts-app/backend/services/websocket_service.py)
- [Research Orchestration Flow](../RESEARCH_WORKFLOW_ANALYSIS.md)
