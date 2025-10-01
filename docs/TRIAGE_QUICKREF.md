# Triage Verification Quick Reference

## 🚀 Quick Start (30 seconds)

```bash
# One-command verification
./scripts/e2e-triage-verification.sh
```

## 🔑 Authentication

```bash
# Browser tokens (if logged in)
./scripts/extract-auth-tokens.sh

# API login
./scripts/login-and-test.sh

# Load saved tokens
source scripts/.env.tokens
```

## ✅ Verification Commands

```bash
# Board structure only
./scripts/test-triage-board-enhanced.sh

# Full pipeline with lane tracking
AUTH_TOKEN="..." ./scripts/verify-triage-enhanced.sh

# Manual board check
curl http://localhost:8000/v1/system/triage/board | jq '.lanes | keys'
# Expected: ["analysis","blocked","classification","context","done","intake","review","search","synthesis"]
```

## 🔌 WebSocket Testing

### Browser Console
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/research/triage-board');
ws.onmessage = e => {
  const msg = JSON.parse(e.data);
  if (msg.type === 'triage.board_update') {
    console.log(Object.entries(msg.data.lanes).map(([k,v]) => `${k}:${v.length}`));
  }
};
```

### HTML Monitor
```bash
open http://localhost:5173/triage-ws-test.html
```

### CLI (wscat)
```bash
wscat -c ws://localhost:8000/ws/research/triage-board
```

## 📊 Lane Flow

```
intake → classification → context → search → analysis → synthesis → review → blocked → done
```

## 🐛 Troubleshooting

| Issue | Fix |
|-------|-----|
| AUTH_TOKEN not set | `./scripts/login-and-test.sh` |
| CSRF error | `export X_CSRF_TOKEN="..."` |
| Type guard rejects events | Verify `'triage.board_update'` in `frontend/src/types/api-types.ts` line ~335 |
| No lane transitions | Check backend logs: `tail -f backend/logs/app.log \| grep triage` |
| WebSocket connection fails | Verify backend running: `curl http://localhost:8000/health` |

## 📝 Expected Responses

### Board Endpoint
```json
{
  "updated_at": "2025-09-30T15:42:18Z",
  "entry_count": 5,
  "lanes": {
    "intake": [...],
    "classification": [...],
    ...
  }
}
```

### WebSocket Event
```json
{
  "type": "triage.board_update",
  "data": {
    "updated_at": "...",
    "entry_count": 5,
    "lanes": {...}
  }
}
```

## 🎯 Success Criteria

- [ ] 9 lanes present in board response
- [ ] `entry_count` is numeric
- [ ] Type guard accepts `triage.board_update`
- [ ] Frontend builds successfully
- [ ] Lane transitions tracked on research submission
- [ ] WebSocket broadcasts full board snapshots

## 🔗 Quick Links

- Full Guide: [`docs/triage-verification-guide.md`](./triage-verification-guide.md)
- Implementation: [`backend/services/triage.py`](../four-hosts-app/backend/services/triage.py)
- Frontend Component: [`frontend/src/components/TriageBoard.tsx`](../four-hosts-app/frontend/src/components/TriageBoard.tsx)
- WebSocket Test: [`frontend/public/triage-ws-test.html`](../four-hosts-app/frontend/public/triage-ws-test.html)
