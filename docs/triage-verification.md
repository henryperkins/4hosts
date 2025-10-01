# Triage Workflow Verification

This playbook exercises the full triage workflow—from intake to cancellation—using only
public HTTP endpoints. It can be executed against a locally running Four Hosts stack
(`./start-app.sh`) or any staging environment that exposes the same API surface.

## Prerequisites

1. Four Hosts backend and frontend running (`./start-app.sh` or `./start_backend.sh` + frontend).
2. Authenticated user session with API access token (PRO tier or higher recommended).
3. `httpie` or `curl` available locally.

## 1. Submit a Research Request

```bash
http POST http://localhost:8000/v1/research/query \
  "Authorization: Bearer <ACCESS_TOKEN>" \
  query="How are municipalities funding climate adaptation projects in 2025?" \
  options:='{
    "depth": "deep_research",
    "enable_real_search": true,
    "max_sources": 120,
    "priority_tags": ["escalation"],
    "paradigm_override": "maeve"
  }'
```

Copy the `research_id` from the response. The request will appear in the Metrics page
under the new "Research Intake Board" column with a **High** priority badge.

## 2. Monitor the Board & Telemetry

1. Navigate to `/metrics` in the frontend.
2. The triage board should now display the new request under **Classification** once the
   backend begins processing. The websocket channel `wss://…/ws/research/triage-board`
   streams updates—use the browser dev tools → Network → WS to confirm messages.
3. The "Telemetry Insights" section will update on the next telemetry flush (within ~30s);
   check that the run count, provider usage, and recent run list acknowledge your request.

## 3. Cancel the Research Run

```bash
http POST http://localhost:8000/v1/research/cancel/${RESEARCH_ID} \
  "Authorization: Bearer <ACCESS_TOKEN>"
```

The triage board lane will slide the card into **Needs Attention** (blocked) and the
websocket payload `triage.board_update` will carry the new state. In the frontend the
board updates instantly; telemetry summary will record the run once the cancellation
propagates.

## 4. Validate System State

- `GET /v1/system/triage-board` should reflect the cancellation:
  ```bash
  http http://localhost:8000/v1/system/triage-board \
    "Authorization: Bearer <ACCESS_TOKEN>"
  ```
- `GET /v1/system/telemetry/summary` should include your run in `recent_events` and update
  `totals.runs` by +1.

Repeat the flow with different depths or user roles to observe the triage scoring changing
in real time. This checklist confirms that intake scoring, websocket broadcasting, and the
telemetry aggregator are wired end-to-end.
