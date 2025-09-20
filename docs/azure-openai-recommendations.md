# Azure OpenAI (AOAI) — Practical Utilization Recommendations

This document proposes concrete, low‑risk improvements to better leverage Azure OpenAI across the Four‑Hosts application. It is based on the current code paths and the Azure‑compatible OpenAI v1 endpoints documented in `docs/v1preview.json`.

## Current State (observed)
- Primary LLM flows use the OpenAI Responses API for o‑series models (o1/o3) with background mode and chaining; non‑o models use Chat Completions.
- Azure endpoints are used via `base_url={endpoint}/openai/v1` with `api-version` (now fixed for Chat Completions). Responses API calls are made with `httpx` and attach `api-version` for Azure.
- Background processing is correct: `store=true`, `background=true`, SSE streaming, cancel, resume, and `previous_response_id` for chaining.
- New include[] support is available on retrieval and streaming to fetch `reasoning.encrypted_content`, tool outputs, etc.

## Quick Wins (high value, low risk)
- Prefer structured outputs via JSON Schema
  - Use `text.format=json_schema` for structured results (already supported in `LLMClient`).
  - Clamp `max_output_tokens`, `max_tool_calls`, and `reasoning.effort` per route to cap costs.
- Enrich diagnostics with Azure correlation IDs
  - Log `apim-request-id` on every AOAI call and propagate it to error payloads. This greatly improves supportability in Azure.
- Centralize model/deployment mapping
  - Create a small registry mapping logical model names (o3, gpt‑4o‑mini, embeddings) → deployment name, region, capabilities; reference everywhere instead of using env in multiple places. Enables per‑env switches without code edits.
- Retry/backoff tuned for Azure
  - Respect `Retry-After` and 429/5xx semantics; add jitter and a circuit breaker to avoid thundering herds.

## Data & Retrieval (RAG) — Azure‑native options
- Vector Stores + File Search (built‑in)
  - Manage vector stores through the service (`/vector_stores`, `/vector_stores/*`) and use the Responses `file_search` tool to ground answers. Scope stores per user/tenant or per project.
  - Benefits: no separate vector DB to run, consistent security and billing, simple attach/detach of uploaded files.
- Azure AI Search (Cognitive Search)
  - For enterprise content, integrate `azure_search` as a data source (per spec) with either ‘integrated’ or ‘deployment_name’ vectorization.
  - Use strictness, top_n, and in_scope controls to tune relevance; cache result snippets and citations.

## Security & Compliance
- Azure AD tokens and Managed Identity
  - Move from `api-key` to AAD bearer tokens (client credentials now; Managed Identity later). Store secrets in Key Vault.
- Private networking & data boundaries
  - Use Private Link to keep traffic off the public internet; restrict outbound; ensure logs do not contain sensitive content.
- Content safety
  - Capture and persist content filter categories, reasons, and actions. Expose an admin view to audit blocked/redacted content flows.

## Observability & Cost Control
- Application Insights integration
  - Emit: success/error rates, latency, request size, token usage per deployment, 429s/5xx, and `apim-request-id`.
- Per‑deployment health + quotas
  - Track rolling health and capacity by deployment; degrade gracefully, fail over on persistent errors; enforce per‑plan quotas in the rate limiter.
- Budget guards in business logic
  - Gate `max_output_tokens`, reasoning effort, tool counts, and background concurrency by subscription tier.

## Resilience & Throughput
- Deployment failover & region diversity
  - Configure multiple deployments for the same capability; fail over on long‑lived errors; annotate responses with chosen deployment.
- Connection reuse
  - Reuse `httpx.AsyncClient` per process to reduce TLS handshake overhead; keep timeouts sensible; configure DNS caching.

## Product Enhancements Enabled by AOAI
- Developer diagnostics (optional UI)
  - Use `include[]` to show `reasoning.encrypted_content` and code interpreter outputs in a hidden “debug drawer” on research results pages.
- Deeper tools in deep research
  - Carefully enable `code_interpreter` and `mcp` for internal tenants; add explicit approvals and output redaction for safety.
- AOAI Evals
  - Introduce an internal eval harness using AOAI Evals (`aoai-evals` header) to quantify prompt/schema changes before rollout.
- Multimedia
  - Where appropriate, adopt image understanding (gpt‑4o‑mini), image generation, and (future) video generation in analysis or export workflows.

## Concrete Changes (proposed sequence)
1) Diagnostics & stabilization (1–2 days)
   - Add `apim-request-id` logging and error propagation.
   - Backoff/retry with `Retry-After` + jitter on 429/5xx.
   - Extract a shared `HttpClient` for AOAI requests (connection pooling).
2) Model registry + capability probe (1–2 days)
   - Model registry (logical → deployment) & startup probe to `/models` to populate `app.state.azure_capabilities`; expose `/v1/system/azure-capabilities`.
   - Feature‑flag UI (hide unsupported tools/resources).
3) Vector Stores & File Search (3–5 days incremental)
   - Backend proxy routes for vector stores/files; attach vector store IDs in research options; enable `file_search` tool.
   - Minimal UI: “Knowledge” page to upload/list files and attach a store to a research run.
4) AAD bearer auth & Key Vault (2–4 days)
   - Add bearer path alongside `api-key`; switch to client credentials in non‑local envs; fetch secrets from Key Vault.
5) Content Safety + admin audit (2–3 days)
   - Persist safety signals with each response; add admin view for auditing.

## Code Pointers (where to implement)
- AOAI clients & retries
  - `four-hosts-app/backend/services/openai_responses_client.py` (error handling, `apim-request-id`, retries, shared client)
  - `four-hosts-app/backend/services/llm_client.py` (model registry, bearer auth option)
- Research & tools
  - `four-hosts-app/backend/services/deep_research_service.py` (attach vector store / azure_search configs)
  - `four-hosts-app/backend/services/mcp_integration.py` (tool gating & approvals)
- System & metrics
  - `four-hosts-app/backend/routes/system.py` (capabilities endpoint)
  - `four-hosts-app/backend/services/monitoring.py` (Insights metrics + traces)
- UI touchpoints
  - `four-hosts-app/frontend/src/services/api.ts` (new endpoints + debug include[] fetch)
  - Research results view (optional debug drawer)

## Guardrails & Defaults
- Defaults for cost and latency
  - `reasoning.effort=low|medium`, `max_output_tokens` sized by plan, `max_tool_calls` conservative for background.
- Safety by default
  - Disable `web_search` and `code_interpreter` on Azure unless tenant explicitly enabled; sanitize tool outputs before rendering.
- Data retention
  - `store=true` only for background/mining; purge retention beyond policy; annotate responses with retention policy.

## Suggested Acceptance Criteria
- AOAI errors log a correlation ID; 429s are retried respecting `Retry-After`.
- A `/v1/system/azure-capabilities` endpoint reports reachable features and deployments.
- Vector store proxy routes exist; research runs can attach a vector store and use `file_search`.
- AAD bearer auth path is available and configurable without code changes.
- Optional: a debug toggle that fetches `include[]` fields for a completed response.

---
If you’d like, I can start with (1) diagnostics and stabilization and (2) the capability probe next, then draft the vector store routes as a focused follow‑up PR.
