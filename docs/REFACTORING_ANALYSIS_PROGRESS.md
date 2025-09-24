# Refactoring Plan: Granular Progress for Analysis Phase

## 1. Background and Context

**Problem:** The user interface stalls during the `analysis` phase, where source credibility is checked. This is a primary source of negative user feedback regarding "frozen" or "stuck" research tasks.

**Root Cause:** The issue is detailed in `docs/PROGRESS_GAPS_ANALYSIS.md` (Gap #2, "Silent credibility checks"). The current implementation in `research_orchestrator.py` calls the `analyze_source_credibility_batch` function, which uses `asyncio.gather` to run all credibility checks concurrently. While efficient, this approach waits for **all** checks to finish before reporting any progress. For a research task with dozens of sources, this creates a single, long-running, silent operation, causing the UI to stall.

This refactoring will instrument this concurrent batch operation to provide real-time, per-source feedback.

## 2. Proposed Solution

The core of this refactoring is to replace the `asyncio.gather` pattern with `asyncio.as_completed`. This will allow the system to iterate through credibility check tasks *as they finish*, rather than waiting for the entire batch.

On each completion, we will call the progress tracker to update the UI, creating a smooth and responsive progress bar. The plan also includes robust error handling, cancellation logic, and edge-case management.

## 3. Detailed Implementation Plan

The required changes are localized to two key files: `credibility.py` and `research_orchestrator.py`.

---

### 3.1 File: `four-hosts-app/backend/services/credibility.py`

The main logic change occurs in this file.

#### **Target Function: `analyze_source_credibility_batch` (starting at line 1084)**

This function will be rewritten to manage the `as_completed` loop and report progress.

##### **Action 1: Modify Function Signature**

The signature must be updated to accept the `progress_tracker` object, the `research_id`, and a `check_cancelled` callback.

*   **Current Signature (line 1084):**
    ```python
    async def analyze_source_credibility_batch(
        sources: List[Dict[str, Any]], 
        paradigm: str = "bernard"
    ) -> Dict[str, Any]:
    ```

*   **New Signature:**
    ```python
    from typing import Callable, Awaitable 

    async def analyze_source_credibility_batch(
        sources: List[Dict[str, Any]],
        paradigm: str,
        progress_tracker: Any,
        research_id: str,
        check_cancelled: Callable[[], Awaitable[bool]]
    ) -> Dict[str, Any]:
    ```

##### **Action 2: Implement Core Refactoring**

Replace the body of the function with the new `as_completed` pattern.

*   **New Implementation Logic:**
    ```python
    import asyncio
    from . import get_source_credibility_safe # Ensure this is imported

    async def analyze_source_credibility_batch(...):
        # 1. Handle the zero-source edge case to prevent phase-skipping.
        if not sources:
            if progress_tracker and research_id:
                await progress_tracker.update_progress(
                    research_id, phase="analysis", items_done=1, items_total=1
                )
            return { "total_sources": 0, "average_credibility": 0.0, ... }

        # 2. Create a list of safe, concurrent tasks using the _safe wrapper.
        tasks = [
            get_source_credibility_safe(
                domain=source.get("domain", ""),
                paradigm=paradigm
            )
            for source in sources
        ]
        
        total_tasks = len(tasks)
        credibility_scores = []

        # 3. Process tasks as they complete.
        for i, future in enumerate(asyncio.as_completed(tasks)):
            # 3a. Check for cancellation on each iteration for responsiveness.
            if await check_cancelled():
                for task in tasks:
                    if not task.done():
                        task.cancel()
                break 

            # 3b. Await the next result. No try/except is needed here because
            # get_source_credibility_safe is guaranteed not to raise an exception.
            score_obj, explanation, status = await future
            if status == 'success':
                credibility_scores.append(score_obj)

            # 3c. Report progress for every completed task (success or fail).
            if progress_tracker and research_id:
                await progress_tracker.update_progress(
                    research_id,
                    phase="analysis",
                    message=f"Analyzing source credibility {i + 1}/{total_tasks}",
                    items_done=i + 1,
                    items_total=total_tasks
                )

        # 4. Aggregate and return results (existing logic can be adapted).
        # ...
        return { ... }
    ```

---

### 3.2 File: `four-hosts-app/backend/services/research_orchestrator.py`

This file only requires a minor change to pass the required arguments to the updated function.

#### **Target Location: `execute_research` method (around line 1003)**

The call to `analyze_source_credibility_batch` inside the `try...except` block for credibility aggregation needs to be updated.

*   **Current Call (line 1003):**
    ```python
    stats = await analyze_source_credibility_batch(
        sources_for_analysis,
        paradigm=normalize_to_internal_code(classification.primary_paradigm),
    )
    ```

*   **New Call:**
    ```python
    # The required variables (progress_callback, research_id, check_cancelled)
    # are already available in the scope of the execute_research method.

    stats = await analyze_source_credibility_batch(
        sources=sources_for_analysis,
        paradigm=normalize_to_internal_code(classification.primary_paradigm),
        progress_tracker=progress_callback,
        research_id=research_id,
        check_cancelled=check_cancelled
    )
    ```

## 4. Expected Outcome

1.  The `analysis` phase progress bar will update smoothly and incrementally for each source checked.
2.  The "stall-then-jump" behavior for this phase will be completely eliminated.
3.  The analysis process will be responsive to user-initiated cancellation requests.
4.  System robustness will be improved by handling individual API failures during credibility checks without halting the entire research process.

## 5. Risk Assessment & Mitigation

-   **Risk:** A single failing credibility check could halt progress.
    -   **Mitigation:** This is mitigated by using the `get_source_credibility_safe` wrapper, which catches exceptions and returns a status tuple, ensuring the processing loop continues uninterrupted.

-   **Risk:** The process could become unresponsive to cancellation.
    -   **Mitigation:** The plan explicitly includes passing a `check_cancelled` function and calling it on each loop iteration, ensuring the process can be terminated gracefully.

-   **Risk:** Incorrect progress calculation if no sources are found.
    -   **Mitigation:** The explicit check for an empty `sources` list ensures the phase is marked as complete, maintaining the integrity of the overall progress calculation.