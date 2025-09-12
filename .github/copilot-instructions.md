## Four Hosts Codebase – AI Assistant Working Guide

Purpose: Give AI coding agents the minimum high‑latency context to act productively without re‑deriving architecture. Keep answers grounded in THESE patterns; do not introduce speculative abstractions.

### 1. Core Domain & Paradigms
- Four paradigms (Dolores/Teddy/Bernard/Maeve) drive tone, search strategy, synthesis style. Classification result flows through every pipeline stage.
- Primary engine: `backend/services/classification_engine.py` (rule first, optional LLM refinement, JSON reasoning, confidence). Respect feature flags and fallback when Azure unavailable.
- Downstream consumers: `routes/research.py`, `deep_research_service.py`, `context_engineering.py`, `answer_generator.py`.

### 2. High-Level Pipeline (Request → Answer)
1. API entry (`routes/research.py`) – classify immediately.
2. Context Engineering W‑S‑C‑I (`services/context_engineering.py`) – rewrite & isolate key concepts; optional LLM rewrite w/ fallback.
3. Search Planning (`services/paradigm_search.py`) + Query Optimization (rule or LLM via `services/llm_query_optimizer.py` behind `ENABLE_QUERY_LLM`).
4. Multi‑API search + dedupe + credibility (`services/search_apis.py`, `services/credibility.py`).
5. Deep research (optional) multi‑stage autonomous run (`services/deep_research_service.py` + `openai_responses_client.py` + `background_llm.py`).
6. Answer synthesis (`services/answer_generator.py`) paradigm‑specific section generation with strict evidence grounding.
7. Critique / QA (`services/llm_critic.py`).
8. Dynamic action items (`services/action_items.py` when `ENABLE_DYNAMIC_ACTIONS=1`).
9. Feedback ingestion (`routes/feedback.py`) persisted for later tuning.

### 3. LLM Interaction Surfaces
- Central facade: `services/llm_client.py` (model routing, retries, structured JSON, tool use, background submissions).
- Background & long tasks: `services/background_llm.py` and `services/openai_responses_client.py` (Responses API, tools: WebSearch, CodeInterpreter, MCP).
- Always pass through paradigm to preserve persona unless explicitly model‑agnostic.
- When extending: prefer adding a thin method on `llm_client.py` rather than ad‑hoc OpenAI calls.

### 4. Key Conventions
- Models & Schemas: Pydantic in `backend/models/` (stay consistent; add new response shapes here, not inline dicts).
- Progress events: Use `services/progress.py` broadcaster—emit phases not verbose token logs.
- Feature flags via environment vars (examples: `ENABLE_QUERY_LLM`, `ENABLE_DYNAMIC_ACTIONS`, background deep research model via `DEEP_RESEARCH_MODEL`). Add docstring + README snippet when introducing new flags.
- Fallback Philosophy: Graceful degradation to heuristic/rule modules if LLM / Redis / external API unavailable—never hard crash user path.
- Paradigm enums: Distinguish between internal enum names (lowercase) and display names/emojis—convert at the edge (serialization layer) not deep services.

### 5. Common Developer Workflows
Backend dev:
```bash
cd four-hosts-app/backend
uvicorn main_new:app --reload
pytest -q                             # unit tests
pytest -m integration                 # external / slower
alembic upgrade head                  # migrations
```
Frontend dev:
```bash
cd four-hosts-app/frontend
npm run dev
```
Full stack helper (auto .env, port prep, optional MCP): `./start-app.sh` (root). Reads/creates `backend/.env` and may spin Brave MCP if key present.
Docker compose (db+redis+backend+frontend): `docker-compose up -d` inside `four-hosts-app/`.

### 6. Testing & Adding Tests
- Place new backend tests in `backend/tests/`; mimic existing naming (e.g. `test_<service>.py`).
- Mark network/LLM heavy paths with `@pytest.mark.integration`.
- For LLM JSON outputs use `generate_structured_output` and assert schema fields to avoid brittle string checks.
- When adding classification enhancements, extend fixtures & expected probability distributions—do not reintroduce legacy Simple/Complex/Ambiguous labels.

### 7. Extending the Pipeline Safely
- New research stage: Add service module + integrate in orchestrator or call site; emit progress event; update `LLM_Analysis_Report.md` if LLM‑involved.
- New paradigm behavior: Update `_PARADIGM_MODEL_MAP`, `_PARADIGM_TEMPERATURE`, system prompts in `llm_client.py`, answer generator subclass, and classification weighting constants.
- Tool additions: Define MCP tool (or schema) then expose via `llm_client.generate_with_tools`; keep tool metadata minimal & deterministic.

### 8. Performance & Resilience Hooks
- Rate limiting / backoff: `services/rate_limiter.py`—reuse not reinvent.
- Caching: Transparent; if introducing high‑volume new queries, add cache key derivation near existing patterns (hash normalized query + paradigm).
- Background timeouts: Avoid hardcoding; prefer configurable value (follow pattern in `background_llm.py`).

### 9. Evidence Grounding Rules
- Answer generation prompts explicitly forbid fabrication—do not strip guardrail instructions.
- All synthesis sections must pass curated evidence blocks (quotes + summaries). Any new generator must maintain this contract.

### 10. Feedback & Adaptive Path
- Feedback endpoints store classification corrections & answer quality; when introducing adaptive logic, consume these events asynchronously (no blocking user request). Keep write path O(1) DB ops.

### 11. What NOT To Do
- Do NOT introduce direct raw OpenAI client calls outside `llm_client.py` / responses client wrappers.
- Do NOT bypass existing retry / timeout wrappers.
- Do NOT couple frontend directly to internal model names—use API JSON fields.
- Do NOT remove rule-based fallbacks when adding LLM features.

### 12. Quick File Landmarks
- Classification: `services/classification_engine.py`
- Deep Research: `services/deep_research_service.py`
- LLM Facade: `services/llm_client.py`
- Search APIs: `services/search_apis.py`
- Answer Synthesis: `services/answer_generator.py`
- Critic: `services/llm_critic.py`
- Query Optimizer (LLM): `services/llm_query_optimizer.py`
- Action Items: `services/action_items.py`
- Feedback Routes: `routes/feedback.py`

### 13. Frontend Integration Notes
- Real-time updates via WebSocket events—when adding new progress phases ensure frontend enumerations updated (search for existing phase strings).
- Paradigm styling & tone handled in React components; reuse existing mapping utilities rather than inlining.

### 14. Adding Environment Variables
- Define in code with sensible default (None or disabled) → document in README section → reference in `start-app.sh` if needed → add to docker compose only if required for container startup.

### 15. Security & Tokens
- Never log raw API keys or JWT secrets.
- Classification & research endpoints must validate minimum query length (see existing Pydantic validators); replicate when adding new entrypoints.

Maintain this file as the single concise source for AI agent operational context. Keep under ~120 lines; prune obsolete references when architecture evolves.
