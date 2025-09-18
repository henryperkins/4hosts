# Feedback UI Integration

- Components
  - `frontend/src/components/feedback/ClassificationFeedback.tsx`
  - `frontend/src/components/feedback/AnswerFeedback.tsx`

- Backend endpoints
  - `POST /v1/feedback/classification`
  - `POST /v1/feedback/answer`

- Frontend wiring
  - Classification feedback renders beneath paradigm displays in `ResearchPage.tsx` (for both live preview and final results).
  - Answer feedback renders inside a "Feedback" card in `ResultsDisplayEnhanced.tsx`.

- Feature flags (backend/.env)
  - `ENABLE_QUERY_LLM=1`
  - `ENABLE_DYNAMIC_ACTIONS=1`
  - `DEEP_RESEARCH_MODEL=o3`

- Notes
  - Classification feedback accepts an optional `research_id`; events are still stored per-user without it.
  - Answer feedback uses 1–5 stars mapped to a normalized 0–1 rating expected by the API.
