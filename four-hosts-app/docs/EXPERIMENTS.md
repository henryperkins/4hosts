Experiments (A/B) – Minimal Scaffold
====================================

Overview
- Adds a tiny, dependency-free experimentation utility to support A/B prompt testing.
- First integration point: Deep Research system prompts per paradigm.
- Deterministic, sticky assignment per `unit_id` derived from `research_id` (or the query string fallback).

Enable
- Set `ENABLE_PROMPT_AB=1` in your environment to activate experiments.
- Optional: override variant weights via JSON `PROMPT_AB_WEIGHTS`.
  - Example: `PROMPT_AB_WEIGHTS='{"v1":0.5,"v2":0.5}'` (default)
  - Force a single variant: `PROMPT_AB_WEIGHTS='{"v2":1.0}'`

Current Experiment
- Name: `deep_research_paradigm_prompt`
- Variants:
  - `v1`: existing prompts (no behavior change)
  - `v2`: tighter phrasing and explicit grounding/citation guidance for each paradigm

Metrics
- `backend/services/metrics.py` already supports `prompt_version`.
- Deep Research Stage 1/2 now records `prompt_version` when experiments are enabled.

Code Touchpoints
- `backend/services/experiments.py`: assignment + env config
- `backend/services/deep_research_service.py`:
  - `PARADIGM_SYSTEM_PROMPTS_V2` – new variant prompts
  - `_build_system_prompt(..., research_id=...)` – variant selection + propagation
  - `DeepResearchConfig.prompt_variant` – captured and recorded in metrics
 - `backend/services/context_engineering.py`:
   - RewriteLayer uses `context_rewrite_prompt` variant (v1/v2)
   - `ContextEngineeringPipeline.process_query(..., research_id=...)` seeds unit_id/variant
 - `backend/services/research_orchestrator.py`:
   - Reads `experiment.prompt_variant` from research record and applies to Deep Research and Answer Generation
 - `backend/services/answer_generator.py`:
   - Variant addenda injected per paradigm when `prompt_variant=v2`

Notes
- Assignment uses a SHA-256 based bucket on `experiment_name::unit_id`.
- No persistence or cross-process coordination is required.
- The scaffold can be reused for future experiments (e.g., context engineering prompts, answer synthesis styles).

Force Variant (QA)
- Send `X-Experiment: v1` or `X-Experiment: v2` with `POST /v1/research/query`.
- The override is stored in the research record and applied across:
  - Deep Research system prompts
  - Context Engineering rewrite prompt
  - Answer Generation prompt addenda
