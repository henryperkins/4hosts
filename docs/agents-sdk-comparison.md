---
title: "Agents SDK Options for Four Hosts: OpenAI Agents SDK vs Azure AI Agents (Python)"
created: 2025-09-22
authors:
  - engineering
description: A practical comparison tailored to the four-hosts-app codebase, with recommendations and a low‑risk pilot plan.
---

# Summary

Short answer: keep the current research pipeline on our existing Responses + orchestrator stack, and pilot agents where they add clear net-new value (chat with memory or realtime/voice). Adopt Azure AI Agents when we need Azure-native governance and tools; adopt OpenAI Agents SDK when we want lightweight, in-app orchestration with sessions and traces.


# What We Already Have (Repo Snapshot)

- Responses API client and routes with Azure compatibility, background jobs, and SSE streaming:
  - `four-hosts-app/backend/services/openai_responses_client.py`
  - `four-hosts-app/backend/routes/responses.py`
- Paradigm-aware LLM façade and deep research service built on Responses:
  - `four-hosts-app/backend/services/llm_client.py`
  - `four-hosts-app/backend/services/deep_research_service.py`
- Agentic query planning and follow-ups integrated into the orchestrator:
  - `four-hosts-app/backend/search/query_planner/planner.py`
  - `four-hosts-app/backend/services/agentic_process.py`
  - `four-hosts-app/backend/services/research_orchestrator.py`
- MCP integration (incl. Brave MCP) and optional docker compose:
  - `four-hosts-app/backend/services/mcp_integration.py`
  - `four-hosts-app/backend/services/brave_mcp_integration.py`
  - `four-hosts-app/backend/docker-compose.mcp.yml`
- Frontend support for agent traces is already present (metadata.agent_trace UI panel).


# OpenAI Agents SDK vs Azure AI Agents (Python)

The two ecosystems solve related problems but target different operational models.

| Area | OpenAI Agents SDK (Python) | Azure AI Agents client library for Python |
|---|---|---|
| Primary target | In-app agent orchestration on top of OpenAI Responses (sessions, tools, handoffs, traces) | Managed Agents in Azure AI Foundry Projects (agents/threads/runs persisted by the service) |
| Auth & setup | `OPENAI_API_KEY`; import and define `Agent`, run with `Runner.run(...)` | Azure identity (Entra ID/`DefaultAzureCredential`) and Project connection; create `AgentsClient`/`AIProjectClient` |
| State model | App-local sessions/memory (e.g., SQLiteSession) managed by SDK | Durable resources in Azure (agents, threads, messages, runs) |
| Tools | SDK tool layer incl. MCP; standard function tools and guardrails; handoffs between agents | Azure-first tools: Azure Functions, Azure AI Search/Enterprise file search, Bing grounding, Browser automation, Fabric data, OpenAPI tools, Code Interpreter, MCP |
| Tracing/obs | OpenAI Traces; event streaming from Responses integrated in SDK | Azure OpenTelemetry and service-side run metadata |
| Realtime/voice | Built-in Realtime/voice (`RealtimeAgent`, `VoicePipeline`) | Not exposed as a comparable agent primitive in this client library |
| Fit with our code | Minimal friction: pairs with our Responses wrapper and agentic loop for chat/memory or voice pilots | Strong for enterprise governance and Azure-native tooling if/when we need centrally managed agents |

References:

- OpenAI Agents SDK docs: https://openai.github.io/openai-agents-python/quickstart/
- Azure AI Agents README (Python): see local clipping `docs/Azure AI Agents client library for Python.md` and upstream links inside it.


# When To Use Which

- Use OpenAI Agents SDK when:
  - We want a lightweight, in-process agent experience (sessions/memory, handoffs, guardrails) without new Azure resources.
  - We want to pilot a conversational assistant or realtime/voice features alongside the existing research stack.

- Use Azure AI Agents when:
  - We need Azure-native governance, Entra ID, and service-hosted agent state.
  - We want built-in access to Azure tools (Functions, AI Search, Bing grounding, Fabric) as first-class capabilities.


# Recommendation For This Repo

1) Do not replace the core research pipeline with either SDK right now.
   - Our orchestrator + Responses + MCP path already implements tool use, chaining, and background runs with Azure compatibility.

2) Start a low-risk pilot where agents add unique value:
   - Pilot A (OpenAI Agents SDK): “Chat with memory” API for user support/triage next to research flows. Benefits: sessions/memory, traces, and minimal infra.
   - Pilot B (OpenAI Realtime/Voice): Voice brief of research results using `RealtimeAgent`/`VoicePipeline` as an optional UX.
   - Optional Azure Pilot: A small Azure Agents demo using Azure Functions or Azure AI Search as a tool, to validate enterprise integration patterns.


# Pilot A: OpenAI Agents SDK (Chat with Memory)

Scope
- Add `openai-agents` under an `ENABLE_AGENTS_SDK=1` feature flag.
- New service: `four-hosts-app/backend/services/agents_preview.py` defining a minimal `Agent` and `Runner` with session persistence.
- New routes: `four-hosts-app/backend/routes/agents.py` with `POST /agents/run` and `POST /agents/session/{id}` returning final text and a compact trace.

Implementation Notes
- Reuse our existing auth/env loader; do not disrupt current `llm_client` or `responses` routes.
- Keep tool list empty or minimal initially; add MCP later if useful.
- Pipe the SDK trace ID into our logs to correlate with Prometheus/OTel.

Success Criteria
- Deterministic startup (flag off by default).
- End-to-end manual test with memory across 3+ turns.
- No regression in existing `/research` endpoints or SSE streaming.


# Pilot B: OpenAI Realtime/Voice (Optional)

Scope
- Use `RealtimeAgent`/`RealtimeRunner` to read back a synthesized summary from the orchestrator.
- Expose a simple WS endpoint gated by `ENABLE_REALTIME=1`.

Success Criteria
- Latency acceptable for a 30–60s “briefing” flow.
- Works without altering existing research orchestration.


# Optional Azure Agents Pilot

Scope
- Add `azure-ai-agents` (and, if needed, `azure-ai-projects`) under `ENABLE_AZURE_AGENTS=1`.
- Create a sample agent with one Azure-native tool (e.g., Azure Function tool) and run a thread with streaming events.

Success Criteria
- Auth via DefaultAzureCredential works in dev.
- Demonstrate a single function tool call via Azure Agents run/stream.


# Risks and Mitigations

- SDK surface churn: keep pilots isolated under feature flags and separate routes.
- Azure vs OpenAI capability drift: pick pilots that don’t lock us in (chat memory, voice brief) and evaluate Azure-native tooling separately.
- Observability fragmentation: log correlation IDs (agent session/run → request ID) and continue Prometheus/OTel for system metrics.


# Decision Checklist (to revisit after pilots)

- Do we need durable, centrally managed agents with Azure governance? → Favor Azure AI Agents.
- Do we primarily need in-app orchestration, session memory, and traces with minimal infra? → Favor OpenAI Agents SDK.
- Do we want Realtime/voice now? → OpenAI Agents SDK.
- Do we need Azure Functions, AI Search, Bing grounding as first-class tools? → Azure AI Agents.


# Appendix A: Links

- OpenAI Agents SDK Quickstart: https://openai.github.io/openai-agents-python/quickstart/
- OpenAI Agents SDK Realtime: https://openai.github.io/openai-agents-python/realtime/quickstart/
- Azure AI Agents (Python) overview: see `docs/Azure AI Agents client library for Python.md` (local); upstream docs and samples are linked inside that file.


# Appendix B: Proposed File Additions (pilots)

- `four-hosts-app/backend/services/agents_preview.py` (OpenAI SDK pilot)
- `four-hosts-app/backend/routes/agents.py` (OpenAPI endpoints for pilot)

All changes behind env flags; no changes to existing orchestrator or Responses routes.

