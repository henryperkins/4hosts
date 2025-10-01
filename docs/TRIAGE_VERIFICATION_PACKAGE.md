# Triage Pipeline Verification - Complete Package

## 📦 What's Included

### Authentication & Setup Scripts
- **`scripts/extract-auth-tokens.sh`** - Extract tokens from browser session
- **`scripts/login-and-test.sh`** - Login via API and extract tokens
- **`scripts/preflight-triage.sh`** - Verify system readiness

### Verification Scripts
- **`scripts/test-triage-board-enhanced.sh`** - Quick board structure tests
- **`scripts/verify-triage-enhanced.sh`** - Full pipeline with lane tracking
- **`scripts/e2e-triage-verification.sh`** - Complete guided workflow

### Frontend Tools
- **`frontend/public/triage-ws-test.html`** - Standalone WebSocket monitor
- **`frontend/src/components/TriageBoard.tsx`** - Live Kanban UI component
- **`frontend/src/hooks/useWebSocket.ts`** - WebSocket connection hook

### Documentation
- **`docs/triage-verification-guide.md`** - Comprehensive guide
- **`docs/TRIAGE_QUICKREF.md`** - Quick reference card

## 🚀 Quick Start Workflow

### Step 1: Pre-flight Check
```bash
./scripts/preflight-triage.sh
```

This verifies:
- ✓ Backend is running and healthy
- ✓ Triage endpoint responds correctly
- ✓ Type guard includes triage events
- ✓ Required tools (jq, curl) installed

### Step 2: Get Authentication Tokens

**Option A: From Browser (fastest if logged in)**
```bash
./scripts/extract-auth-tokens.sh
# Follow on-screen instructions to run JS in browser console
# Copy/paste the export commands
```

**Option B: API Login**
```bash
./scripts/login-and-test.sh
# Enter email/password when prompted
source scripts/.env.tokens  # Load tokens into current shell
```

### Step 3: Run Verification

**Quick Test (30 sec)**
```bash
./scripts/test-triage-board-enhanced.sh
```

**Full Verification (2 min)**
```bash
AUTH_TOKEN="..." X_CSRF_TOKEN="..." ./scripts/verify-triage-enhanced.sh
```

**Guided E2E (interactive)**
```bash
./scripts/e2e-triage-verification.sh
```

### Step 4: Live WebSocket Monitoring

**Browser Monitor**
```bash
# Open in browser
open http://localhost:5173/triage-ws-test.html
```

**Console (manual)**
```javascript
// In browser console
const ws = new WebSocket('ws://localhost:8000/ws/research/triage-board');
ws.onmessage = e => {
  const msg = JSON.parse(e.data);
  if (msg.type === 'triage.board_update') {
    console.table(
      Object.entries(msg.data.lanes).map(([lane, items]) => ({
        lane,
        count: items.length
      }))
    );
  }
};
```

## 📊 Expected Results

### Board Endpoint Response
```json
{
  "updated_at": "2025-09-30T15:42:18.123Z",
  "entry_count": 5,
  "lanes": {
    "intake": [],
    "classification": [{"research_id": "res-123", "priority": "high", ...}],
    "context": [],
    "search": [{"research_id": "res-456", "priority": "medium", ...}],
    "analysis": [],
    "synthesis": [{"research_id": "res-789", "priority": "medium", ...}],
    "review": [],
    "blocked": [],
    "done": [{"research_id": "res-abc", "priority": "low", ...}]
  }
}
```

### Lane Transition Sequence
```
[1] Lane: intake          (0.5s after submission)
[2] Lane: classification  (1.2s - paradigm determination)
[3] Lane: context        (2.3s - W-S-C-I pipeline)
[5] Lane: search         (3.1s - multi-provider search)
[8] Lane: analysis       (5.8s - evidence extraction)
[12] Lane: synthesis     (8.4s - answer generation)
[14] Lane: review        (9.2s - quality validation)
[15] Lane: done          (9.5s - completion)
```

### WebSocket Event Stream
```javascript
// Initial board state
{type: "triage.board_update", data: {updated_at: "...", entry_count: 2, lanes: {...}}}

// After submission
{type: "triage.board_update", data: {entry_count: 3, lanes: {intake: [new_entry], ...}}}

// Each transition
{type: "triage.board_update", data: {lanes: {intake: [], classification: [new_entry], ...}}}
```

## ✅ Verification Checklist

### API Layer
- [ ] Board endpoint returns 9 lanes (empty arrays when no entries)
- [ ] `entry_count` is numeric
- [ ] `updated_at` is ISO 8601 timestamp
- [ ] No `totals` field present
- [ ] All lanes are arrays

### Frontend Layer
- [ ] Type guard accepts `triage.board_update` in `api-types.ts` line ~335
- [ ] WebSocket hook handles triage events
- [ ] TriageBoard component renders correctly
- [ ] Frontend builds without errors
- [ ] HTML monitor connects and displays events

### Integration Layer
- [ ] Research submission creates intake entry
- [ ] Lane transitions trigger WebSocket broadcasts
- [ ] Full board snapshot sent on each update (not deltas)
- [ ] Completion moves entry to done lane
- [ ] Errors move entry to blocked lane

## 🐛 Troubleshooting

### Backend Not Running
```bash
# Check health
curl http://localhost:8000/health

# If fails, start backend
cd four-hosts-app/backend
uvicorn main_new:app --reload

# Or use convenience script
./start-app.sh
```

### Authentication Issues
```bash
# Check token validity
curl -H "Authorization: Bearer $AUTH_TOKEN" \
  http://localhost:8000/v1/auth/me

# If expired, re-login
./scripts/login-and-test.sh
```

### Type Guard Blocking Events
```bash
# Verify patch applied
grep -A 15 "validTypes = \[" \
  four-hosts-app/frontend/src/types/api-types.ts | \
  grep "triage.board_update"

# Should show: 'triage.board_update',
# If missing, add it to the validTypes array
```

### WebSocket Connection Fails
```bash
# Test WebSocket endpoint
wscat -c ws://localhost:8000/ws/research/triage-board

# Check proxy configuration
grep -A 10 "proxy" four-hosts-app/frontend/vite.config.ts

# Verify WebSocket upgrade allowed
curl -i -N \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  http://localhost:8000/ws/research/triage-board
```

### No Lane Transitions Observed
```bash
# Check research actually submitted
RESEARCH_ID="res-123"
curl -H "Authorization: Bearer $AUTH_TOKEN" \
  "http://localhost:8000/v1/research/status/$RESEARCH_ID"

# Check backend logs
tail -f four-hosts-app/backend/logs/app.log | grep -i triage

# Verify triage manager called
grep -r "update_lane\|initialize_entry" \
  four-hosts-app/backend/services/research_orchestrator.py
```

## 📈 Performance Benchmarks

| Metric | Target | Typical | Remediation |
|--------|--------|---------|-------------|
| Board endpoint latency | <100ms | ~45ms | Check Redis connection |
| WebSocket broadcast | <50ms | ~25ms | Monitor connection pool |
| Lane transition frequency | 1-2/sec | Variable | Normal variation |
| Frontend render | <16ms | ~8ms | Optimize lane data structure |
| Full research pipeline | <20s | ~12s | Check search API timeouts |

## 🔗 Integration Points

The triage system integrates with the research pipeline at these points:

### 1. Submission (`routes/research.py`)
```python
await triage_manager.initialize_entry(
    research_id=research_id,
    query=research.query,
    user_id=str(current_user.user_id)
)
```

### 2. Phase Transitions (`services/research_orchestrator.py`)
```python
# After classification
await triage_manager.update_lane(research_id, phase="classification")

# After context engineering
await triage_manager.update_lane(research_id, phase="context")

# After search
await triage_manager.update_lane(research_id, phase="search")

# After evidence extraction
await triage_manager.update_lane(research_id, phase="analysis")

# After synthesis
await triage_manager.update_lane(research_id, phase="synthesis")

# Before publication
await triage_manager.update_lane(research_id, phase="review")
```

### 3. Completion/Errors
```python
# Success
await triage_manager.mark_completed(research_id)

# Failure
await triage_manager.mark_blocked(
    research_id,
    reason="Search timeout after 3 retries"
)

# Metadata note: the blocked reason is persisted on the triage entry under
# `metadata.blocked_reason` for downstream diagnostics.
```

## 📝 Next Steps

After successful verification:

### 1. Monitor Production Traffic
```bash
# Real-time lane distribution
watch -n 2 'curl -s http://localhost:8000/v1/system/triage/board | \
  jq ".lanes | map_values(length)"'

# Historical lane dwell times
curl http://localhost:8000/v1/system/metrics/triage/dwell-times
```

### 2. Integrate with Operations Dashboard
- Add triage board widget to admin UI
- Set up Prometheus metrics for lane counts
- Configure alerting for blocked lane threshold
- Track SLA compliance via lane timing

### 3. Enhance Functionality
- Implement priority scoring based on user tier
- Add manual lane override controls for ops team
- Build historical analytics dashboard
- Set up anomaly detection for stuck requests

## 🎯 Success Criteria Summary

All verification scripts should pass:
```bash
./scripts/preflight-triage.sh         # ✓ System ready
./scripts/test-triage-board-enhanced.sh  # ✓ Board structure
./scripts/verify-triage-enhanced.sh     # ✓ Full pipeline
```

WebSocket monitor should show:
- ✓ Initial board snapshot on connection
- ✓ Full board update on each lane transition
- ✓ No disconnections or errors
- ✓ Lane counts match API responses

Frontend should build cleanly:
```bash
cd four-hosts-app/frontend
npm run build  # ✓ No errors
npm run lint   # ✓ No triage-related warnings
```

## 📚 Documentation Index

- **Quick Start**: [`docs/TRIAGE_QUICKREF.md`](./TRIAGE_QUICKREF.md)
- **Full Guide**: [`docs/triage-verification-guide.md`](./triage-verification-guide.md)
- **Original Analysis**: [`RESEARCH_WORKFLOW_ANALYSIS.md`](../RESEARCH_WORKFLOW_ANALYSIS.md)
- **Implementation Details**: [`four-hosts-app/backend/services/triage.py`](../four-hosts-app/backend/services/triage.py)

---

**Package Version**: 1.0.0
**Last Updated**: October 1, 2025
**Status**: ✅ Production Ready
