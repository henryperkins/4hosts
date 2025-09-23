# Progress Gaps – Analysis & Remediation Road-map

> Prepared after a deep dive into **websocket_service.ProgressTracker** and
> the research execution flow (2025-09-23).

This document captures the disconnects we observed between real-time work done
in the backend **(classification → search → analysis → agentic_loop →
synthesis)** and what users actually see on the front-end progress bar.
Each gap lists the root cause, impact, and the remediation plan/status.

| # | Gap / Symptom | Root Cause | Impact | Remediation | Status |
|---|----------------|------------|--------|-------------|--------|
| 1 | **Agentic loop appears to “stall” at ~75 %** | `run_followups()` performs multi-iteration work but never calls `ProgressTracker.update_progress()` | Users believe research is frozen during iterative follow-ups | Emit weighted `agentic_loop` updates at start, each iteration and on completion | **Fixed in code** (commit `agentic-loop-progress`) |
| 2 | **Silent credibility checks** | `get_source_credibility_safe()` runs in parallel but doesn’t notify the tracker | No feedback⇒ perceived freeze during large credibility batches | Call `update_progress()` while iterating through domains (analysis phase) | **Partially fixed** – batching loop now emits updates; deeper integration TBD |
| 3 | **No explicit “ranking” phase** | Ranking happens inline inside `analysis` code path | Progress % jumps from search→analysis without visible sub-state | Either: (a) fold ranking weight into analysis, or (b) introduce `ranking` pseudo-phase | **Open** – design discussion pending |
| 4 | **Classification LLM spike looks like freeze** | Long‐running call in `classification_engine.py` without granular updates | UX freeze in the first 10 % of bar | Inject token-level or step-level updates inside engine | **Planned** |
| 5 | **Search APIs – no per-API feedback** | Only phase start/end recorded; each provider call can take seconds | Users cannot see progress inside a 300-query sweep | Wire provider callbacks to `ProgressTracker.update_progress(items_done/total)` | **Planned** (requires SearchManager support) |
| 6 | **Synthesis lacks sub-phases (evidence, citations, draft, refine)** | Back-end collapses them into one “synthesis” phase | Bar freezes at 90 % for long answers | Expose sub-steps via `items_total` units and fine-grained messages | **Planned** |
| 7 | **Inconsistent error recovery events** | Many try/except blocks log but skip `RESEARCH_FAILED` WebSocket event | Front-end never learns about fatal errors | Standardise `ProgressTracker.report_error()` helper and use except-level instrumentation | **Open** |
| 8 | **Phase race conditions** | Phase variable is mutated *before* prior phase is marked complete | Double counting / incorrect weight accumulator | Call `update_progress()` *after* completing phase actions; tracker now canonicalises order | **Partially fixed** – but audit continues |
| 9 | **Snapshot persistence may drop fast events** | 2 s debounce sometimes misses final updates before process exit | Last-second UI blink | Force-persist on every phase change + graceful shutdown hook | **Open** |
|10 | **Weight model abrupt 100 % jump** | Terminal `complete` weight is 0→ abrupt jump from 95 % → 100 % | UX jank | Smooth ramp by giving `complete` a small (e.g. 3 %) weight or tween on FE | **Design** (needs PM/UX input) |

## Next Steps

1. Finalise design for ranking/synthesis sub-phases – update `_phase_weights` accordingly.
2. Extend SearchManager to surface per-provider progress hooks.
3. Create `ProgressTracker.report_error()` and refactor error paths.
4. Implement shutdown hook to flush final progress snapshot.

–– *End of document*

