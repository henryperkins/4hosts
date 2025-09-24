# Context Engineering Workflow Overview

This note captures the end-to-end flow we now run when the research service
handles a request.  It distils the guidance from recent clippings into a
checklist the engineering team can reference while iterating on agents.

## Lifecycle

1. **Plan / Route** – The orchestrator classifies the request and chooses the
   appropriate workflow.  For single-hop work we still favour prompt chaining;
   the full agent loop is only engaged when the query merits it.
2. **Context Engineering (W-S-C-I)** – `ContextEngineeringPipeline` produces
   write/select/compress/isolate outputs.  These feed the new
   `ContextPackager`, which turns them into a budgeted payload with explicit
   segment budgets (instructions/knowledge/tools/scratchpad).
3. **Memory Management** – Per-research sessions are registered via
   `session_manager`.  Short workflows default to a trim policy (last 4 user
   turns) while iterative agent loops switch to summarisation so older state is
   retained compactly.
4. **Act / Retrieve** – Query planner executes the deterministic search batch
   and the agentic follow-up loop.  Each phase writes a short summary back into
   the session so we can replay the trace deterministically.
5. **Reason / Write** – Deep research synthesis uses the packaged context when
   building its system prompt and user prompt.  The same package is exposed via
   `DeepResearchService.get_packaged_context` for debugging.
6. **Reflect / Evaluate** – The `services.evaluation.context_evaluator`
   helpers compute quick precision/utilisation/groundedness heuristics.  They
   serve as a lightweight gate before heavier LLM-based evals.

## Operational Notes

- The `ContextPackager` enforces per-segment budgets derived from the
  compressor layer.  Trimming is logged per item, making it easy to inspect
  what was dropped if we miss important context.
- `TrimmingSession` and `SummarizingSession` accept metadata-free dictionaries,
  which keeps the integration surface extremely small.  Each session exposes a
  `get_items` coroutine that can be wired directly into prompt templates.
- Evaluation helpers are heuristic on purpose – they require no network IO.
  For deeper evaluation harnesses we can plug in LLM-as-judge metrics while
  keeping these as a fast “smoke test”.

This workflow should be treated as the default starting point for any new
research feature.  Deviations (e.g., bypassing the packager) should be called
out explicitly in code reviews.

