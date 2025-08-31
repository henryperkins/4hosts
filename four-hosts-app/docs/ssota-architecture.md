# SSOTA Architecture

This is the canonical architecture synthesis for the Four Hosts agentic research system, aligned to SSOTA naming. It consolidates prior Architecture, Classification & Context, Integration, Visual Flow, API, and MVP docs in this folder.

---

## System View

```
User/UI → Classification Engine → Context Engineering (W‑S‑C‑I) → Research Execution → Synthesis & Presentation
```

### Layers

- User Interface: query input, options (depth, overrides), progress, results, paradigm visualization (“why this paradigm?”).
- Classification Engine: Query analysis (entities, intent, domain, urgency, complexity, valence) → rule/LLM hybrid → distribution + confidence.
- Context Engineering: Write, Select, Compress, Isolate layers generate engineered queries, source/filters, token budgets, extraction structures.
- Research Execution: multi‑source search, credibility scoring, deduplication, fact extraction; caching and quotas.
- Synthesis & Presentation: paradigm‑specific templates; multi‑paradigm integration when required; citations and exports.

Normalization note: API paths use a versioned root without an extra `/api` segment (e.g., `/v1/research/query`). See “API Surface Summary”.

## Key Components

### Classification Engine

- Features: tokenization, simple entity detection, intent signals, domain bias, urgency/complexity/valence scoring.  
- Hybrid scoring: rule patterns + LLM augmentation; normalization to distribution; primary/secondary selection; confidence calculation.  
- Performance: caching and async execution for throughput.

### Context Engineering (W‑S‑C‑I)

- Write: paradigm strategies yield focus, themes, narrative, search priorities.
- Select: enhanced queries (original, paradigm‑modified, theme‑enhanced, entity‑focused), source preferences and tools, optional secondary queries.
- Compress: paradigm‑specific ratios (e.g., Maeve ~0.4, Bernard ~0.5), priority vs. removed elements, token budget from complexity.
- Isolate: paradigm‑specific extraction patterns, focus areas, and output schemas.

### Memory & State Model

- Instruction store: paradigm system prompts, tool descriptions, few‑shot exemplars.  
- Working memory: rolling message/state with aggressive summarization/trim policies.  
- Long‑term memory: user/project profiles and collections of prior findings for retrieval.  
- Tool feedback store: raw outputs from web/search/fact extraction kept outside the prompt until referenced.  
- State schema: typed fields for each of the above; only the minimal slice is injected per step.

### Research Execution

- Orchestrator: executes searches aligned to paradigm; re‑ranking and filtering; source‑type prioritization.
- Credibility: paradigm‑weighted source evaluation (e.g., peer‑reviewed for Bernard; industry/consultancy for Maeve, etc.).
- Caching: classification, search, and credibility results with appropriate TTLs.
 - Isolation & sandboxes: run tool calls in restricted contexts; sanitize and gate outputs before injection.
 - Tool selection at scale: when many tools are available, use semantic search over tool descriptions to pick candidates.

### Synthesis & Presentation

- Answer templates per paradigm (opening, body, action/metrics, limitations as applicable).  
- Integration: combine secondary paradigm insights; surface conflicts and trade‑offs.  
- Exports: PDF/Markdown/JSON; show citations inline and in bibliography.

### API Surface Summary (normalized)

- Base path: `/v1` (domain/environment configured outside this doc).  
- Endpoints (selected):  
  - `POST /v1/research/query` – submit a query (options: depth, include_secondary, region/language, max_sources).  
  - `GET  /v1/research/status/{id}` – polling status.  
  - `GET  /v1/research/results/{id}` – final results + sections + citations + metadata.  
  - `POST /v1/paradigms/classify` – classification only.  
  - `POST /v1/paradigms/override` – switch paradigm for an existing research.  
  - `POST /v1/search/paradigm-aware` – single search with paradigm context.  
  - `GET  /v1/paradigms/explanation/{paradigm}` – descriptive info.  
- Error model: `{ error: { code, message, details? } }` with conventional 4xx/5xx.  
- Auth: Bearer token; rate limits per plan (see API v1 doc in this folder for detail).

## Data Contracts (reference shape)

```json
// ClassificationResult (essential fields)
{
  "query": "...",
  "primary": "maeve",
  "secondary": "dolores" | null,
  "distribution": {"maeve": 0.40, "dolores": 0.25, "bernard": 0.20, "teddy": 0.15},
  "confidence": 0.78,
  "features": {"entities": [], "intent": [], "domain": null, "urgency": 0, "complexity": 0, "valence": 0}
}
```

```json
// ContextEngineeredQuery (essential fields)
{
  "write": {"focus": "...", "themes": ["..."], "narrative": "...", "priorities": ["..."]},
  "select": {"queries": [{"query": "...", "type": "paradigm_modified", "weight": 0.8}],
             "sources": ["industry"], "exclude": ["opinion"], "tools": ["market_analysis"], "max_sources": 100},
  "compress": {"ratio": 0.4, "strategy": "action_extraction", "token_budget": 1300},
  "isolate": {"strategy": "strategic_intelligence", "patterns": ["..."], "structure": {"strategic_opportunities": []}}
}
```

Notes: Keys and shapes mirror the concept docs’ examples; treat them as reference contracts. Keep field names consistent across services and SDKs.

## Operational Concerns

- Caching: classification (∼1h), search (∼24h), credibility (∼1w); invalidate intelligently.  
- Cost/latency controls: adjustable depth, parallelization, progressive loading, dedupe.  
- Observability: accuracy tracking, paradigm distribution, cache metrics, error rates, P95 latencies.
- Security & Ethics: JWT auth, rate limits, prompt‑injection awareness for fetched content, clear citations, privacy for user queries.
  - Threats to consider: SSRF via fetched URLs, prompt‑injection in retrieved text, over‑collection of PII, citation spoofing.  
  - Controls: allow‑listing fetchers, content sanitization, source credibility scoring, PII minimization, audit logs.

### Context Hygiene Policies

- Token budgets per step and per layer (W‑S‑C‑I); enforce hard caps.  
- Summarize/trim working memory periodically; remove low‑signal text.  
- De‑duplicate and merge near‑duplicates from search results.  
- Contradiction detection: flag clashes across sources; surface in synthesis.  
- Quarantine suspicious tool outputs; require secondary verification before use.

## Deployment Sketch (from materials)

- Backend: FastAPI (Python), async I/O.  
- Search: multiple providers; quotas and rate‑limit handling.  
- Storage: Postgres; Redis for cache.  
- Container/K8s; monitoring with Prometheus/Grafana; CDN for static assets.

## Targets & SLO Framing (as documented)

- Classification accuracy target ≥85% (hybrid).  
- Classification + W‑S‑C‑I: sub‑second under local conditions (excludes external search latency).  
- Answer synthesis with correct citations; user‑visible confidence.  
- Cost per query controllable via depth and caching.

## Module Boundaries (names from concept code samples)

- `classification_engine` – analyzer + classifier + engine wrapper (cache).  
- `context_engineering_pipeline` – Write/Select/Compress/Isolate + orchestration.  
- `integrated_system` – ties classification and context pipeline; prepares search queries.  

These names appear in prototype code within the docs; treat as illustrative until matched by real modules in `src/`.

---

This architecture is an aligned consolidation; it does not introduce new features beyond those described in the concept docs. Where numbers appear, treat them as targets or prototype‑mode measurements unless otherwise stated.
