"""
Agentic Research Process utilities: lightweight planning, coverage critique,
and query proposal to drive iterative research.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any

import structlog

# Optional OpenTelemetry helper (failsafe import)
try:
    from utils.otel import otel_span as _otel_span  # type: ignore
except Exception:  # pragma: no cover – tracing is optional
    # Provide a no-op fallback so code always runs even if OTEL is disabled
    from contextlib import contextmanager

    @contextmanager
    def _otel_span(*_args: Any, **_kwargs: Any):  # type: ignore
        yield None
import re


logger = structlog.get_logger(__name__)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if t]


def _token_overlap(a: str, b: str) -> float:
    """Jaccard token overlap between two strings (0..1)."""
    sa, sb = set(_tokenize(a)), set(_tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def evaluate_coverage_from_sources(
    original_query: str,
    context_engineered: Any,
    sources: List[Dict[str, Any]],
) -> Tuple[float, List[str]]:
    """
    Heuristic coverage evaluation.

    Strategy:
      1. Collect targets from Write layer key_themes and Isolate layer
         focus_areas (lower-cased, deduped).
      2. For each target term, mark covered if any source satisfies:
         a) exact substring match (case-insensitive)
         b) token-overlap ≥ 0.6 (handles reordered/partial phrases)
         c) all non-stop-words of target appear individually in text.
      3. Coverage = |covered| / |targets|.

    Returns:
      (coverage, missing_terms)
    """
    # Collect themes
    try:
        write_output = getattr(context_engineered, "write_output", None)
        themes_raw = (
            getattr(write_output, "key_themes", []) if write_output else []
        )
        themes = set([t.lower() for t in (themes_raw or [])])
    except Exception:
        themes = set()

    # Collect focus areas
    try:
        isolate_output = getattr(context_engineered, "isolate_output", None)
        focus_raw = (
            getattr(isolate_output, "focus_areas", [])
            if isolate_output
            else []
        )
        focus = set([t.lower() for t in (focus_raw or [])])
    except Exception:
        focus = set()

    targets = [
        t.strip() for t in list(themes.union(focus)) if len(t.strip()) > 2
    ]
    if not targets:
        return 1.0, []

    covered = set()
    joined = [
        _normalize(f"{s.get('title', '')} {s.get('snippet', '')}")
        for s in (sources or [])
    ]
    for term in targets:
        term_norm = _normalize(term)
        words = [w for w in term_norm.split() if w]
        for text in joined:
            if not term_norm:
                continue
            # a) exact substring
            if term_norm in text:
                covered.add(term)
                break
            # b) fuzzy token overlap ≥ 0.6
            if _token_overlap(term_norm, text) >= 0.6:
                covered.add(term)
                break
            # c) all content words present individually (for 3+ word phrases)
            if len(words) >= 3 and all(w in text for w in words):
                covered.add(term)
                break

    coverage = len(covered) / float(len(targets)) if targets else 1.0
    missing = [t for t in targets if t not in covered]
    return coverage, missing


def propose_queries_from_missing(
    original_query: str,
    paradigm: str,
    missing_terms: List[str],
    max_new: int = 4,
) -> List[str]:
    """
    Propose new queries mixing missing themes with paradigm-flavored
    modifiers.
    """
    # Guard against None input
    base = (original_query or "").strip()
    modifiers = {
        "dolores": ["investigation", "expose", "scandal"],
        "bernard": ["study", "evidence", "site:.edu OR site:.gov"],
        "maeve": ["strategy", "framework", "tactics"],
        "teddy": ["resources", "support", "guide"],
    }.get(paradigm, [])

    proposals: List[str] = []
    for term in missing_terms[: max_new * 2]:
        term = (term or "").strip()
        if not term:
            continue
        # Combine with top modifiers
        for m in modifiers[:2]:
            q = f"{term} {base} {m}".strip()
            if q not in proposals:
                proposals.append(q)
                if len(proposals) >= max_new:
                    return proposals
        # Plain term + base fallback
        if len(proposals) < max_new:
            q2 = f"{term} {base}".strip()
            if q2 not in proposals:
                proposals.append(q2)
                if len(proposals) >= max_new:
                    return proposals

    return proposals[:max_new]


def summarize_domain_gaps(sources: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Return simple counts by domain class:
    academic, government, nonprofit, media, industry.
    """
    # Dynamic import with fallbacks to avoid static-analysis import errors
    try:
        _dc = __import__("utils.domain_categorizer", fromlist=["categorize"])
        categorize = getattr(_dc, "categorize")  # type: ignore[assignment]
    except Exception:
        def categorize(host: str) -> str:  # type: ignore[no-redef]
            return "other"

    try:
        _uu = __import__("utils.url_utils", fromlist=["extract_domain"])
        extract_domain = getattr(_uu, "extract_domain")  # type: ignore[assignment]
    except Exception:
        def extract_domain(u: str) -> str:  # type: ignore[no-redef]
            try:
                from urllib.parse import urlparse
                host = (urlparse(u).netloc or "").lower()
                if host.startswith("www."):
                    host = host[4:]
                return host
            except Exception:
                return ""

    counts: Dict[str, int] = {
        "academic": 0,
        "government": 0,
        "nonprofit": 0,
        "media": 0,
        "industry": 0,
    }
    for s in sources or []:
        u = s.get("url") or ""
        host = extract_domain(u)
        category = categorize(host)

        # Map domain categorizer categories to agentic categories
        if category == "academic":
            counts["academic"] += 1
        elif category == "government":
            counts["government"] += 1
        elif category in ["blog", "reference"]:
            counts["nonprofit"] += 1
        elif category == "news":
            counts["media"] += 1
        else:
            counts["industry"] += 1
    return counts


async def run_followups(
    original_query: str,
    context_engineered: Any,
    paradigm_code: str,
    planner: Any,
    seed_query: str,
    executed_queries: set[str],
    coverage_sources: List[Dict[str, Any]],
    *,
    max_iterations: int = 2,
    coverage_threshold: float = 0.75,
    max_new_per_iter: int = 4,
    estimate_cost=None,
    can_spend=None,
    execute_candidates=None,
    to_coverage_sources=None,
    check_cancelled=None,
    progress_callback=None,
    research_id: str | None = None,
) -> Tuple[
    List[Any],
    Dict[str, List[Any]],
    float,
    List[str],
    List[Dict[str, Any]],
]:
    """
    Agentic follow-up loop helper.

    Arguments:
      - estimate_cost(cand) -> float: per-candidate estimated cost
      - can_spend(total_cost) -> bool: budget guard
      - execute_candidates(cands) -> Dict[str, List[Any]]:
        run search for new cands
      - to_coverage_sources(res_list) -> List[Dict]:
        map results to coverage rows
      - check_cancelled(): optional awaitable to check cancellation

    Returns:
      (new_candidates, followup_results_map, final_coverage_ratio,
       missing_terms, coverage_sources)
    """
    new_candidates: List[Any] = []
    followup_results: Dict[str, List[Any]] = {}

    # Helper to emit progress updates (agentic_loop phase) if a tracker is
    # provided.  We centralise the logic so individual call-sites stay tidy.
    async def _emit_progress(current_iter: int, total_iter: int, msg: str | None = None):
        if progress_callback and research_id:
            try:
                await progress_callback.update_progress(
                    research_id,
                    phase="agentic_loop",
                    items_done=current_iter,
                    items_total=max(1, total_iter),
                    message=msg,
                )
            except Exception:
                # We never fail the follow-up loop due to progress errors.
                pass

    # Initial coverage
    coverage_ratio, missing_terms = evaluate_coverage_from_sources(
        original_query,
        context_engineered,
        coverage_sources,
    )

    # Emit initial progress snapshot (iteration 0 / max_iterations)
    await _emit_progress(0, max_iterations, msg="Starting agentic follow-up planning")

    followup_triggered = (
        max_iterations > 0
        and coverage_ratio < coverage_threshold
        and bool(missing_terms)
    )
    logger.info(
        "agentic_coverage_evaluation",
        stage="initial",
        coverage_ratio=float(coverage_ratio or 0.0),
        coverage_threshold=float(coverage_threshold),
        missing_terms=len(missing_terms or []),
        paradigm=paradigm_code,
        research_id=research_id,
        followup_triggered=bool(followup_triggered),
        max_iterations=int(max_iterations),
    )

    if (
        max_iterations <= 0
        or coverage_ratio >= coverage_threshold
        or not missing_terms
    ):
        if max_iterations <= 0:
            skipped_reason = "max_iterations"
        elif coverage_ratio >= coverage_threshold:
            skipped_reason = "coverage_met"
        elif not missing_terms:
            skipped_reason = "no_missing_terms"
        else:
            skipped_reason = "skipped"
        # Report agentic loop complete even when skipped
        if progress_callback and research_id:
            await progress_callback.update_progress(
                research_id,
                phase="agentic_loop",
                items_done=max_iterations, items_total=max_iterations,
                message="Agentic loop skipped – coverage met"
            )
        logger.info(
            "agentic_loop_complete",
            iterations_executed=0,
            final_coverage=float(coverage_ratio or 0.0),
            remaining_terms=len(missing_terms or []),
            paradigm=paradigm_code,
            research_id=research_id,
            followup_triggered=False,
            skipped_reason=skipped_reason,
        )
        return (
            new_candidates,
            followup_results,
            coverage_ratio,
            missing_terms,
            coverage_sources,
        )

    iteration = 0
    while (
        iteration < max_iterations
        and coverage_ratio < coverage_threshold
        and missing_terms
    ):
        # ----------------------------------------------------------------------------
        # Trace each follow-up iteration so we can analyse agent behaviour in OTLP.
        # ----------------------------------------------------------------------------
        with _otel_span(
            "agent.loop.iteration",
            {
                "iteration": int(iteration),
                "current_coverage": float(coverage_ratio or 0.0),
                "missing_terms": int(len(missing_terms)),
            },
        ):
            # Progress update before heavy planner work
            await _emit_progress(
                iteration,
                max_iterations,
                msg=f"Follow-up iteration {iteration + 1}/{max_iterations} – planning",
            )

            # -----------------------------
            # Planner proposes follow-ups
            # -----------------------------
            try:
                proposed = await planner.followups(
                    seed_query=seed_query,
                    paradigm=paradigm_code,
                    missing_terms=missing_terms,
                    coverage_sources=coverage_sources,
                )
            except Exception:
                break

            # Filter out already executed and cap per-iteration
            filtered: List[Any] = []
            for cand in proposed or []:
                q = getattr(cand, "query", None)
                if not q or q in executed_queries:
                    continue
                filtered.append(cand)
                if 0 < max_new_per_iter <= len(filtered):
                    break

            if not filtered:
                break

            # Budget check
            try:
                if estimate_cost is not None:
                    estimated = sum(float(estimate_cost(c)) for c in filtered)
                else:
                    estimated = 0.0
            except Exception:
                estimated = 0.0

            try:
                if can_spend is not None and not can_spend(estimated):
                    break
            except Exception:
                # If budget callback fails, stop follow-ups
                break

            # Cancellation gate
            try:
                if check_cancelled and await check_cancelled():
                    break
            except Exception:
                pass

            # Execute selected candidates
            try:
                results_map = {}
                if execute_candidates:
                    results_map = await execute_candidates(filtered)
            except Exception:
                results_map = {}

            # Merge new results and extend coverage sources
            for query, res_list in (results_map or {}).items():
                followup_results[query] = res_list
                if to_coverage_sources:
                    try:
                        rows = to_coverage_sources(res_list)
                        coverage_sources.extend(rows or [])
                    except Exception:
                        pass

            # Update tracking
            for cand in filtered:
                q = getattr(cand, "query", None)
                if q:
                    executed_queries.add(q)
            new_candidates.extend(filtered)

            # Recompute coverage
            coverage_ratio, missing_terms = evaluate_coverage_from_sources(
                original_query,
                context_engineered,
                coverage_sources,
            )

            logger.info(
                "agentic_coverage_evaluation",
                stage="iteration",
                iteration=int(iteration + 1),
                coverage_ratio=float(coverage_ratio or 0.0),
                coverage_threshold=float(coverage_threshold),
                missing_terms=len(missing_terms or []),
                paradigm=paradigm_code,
                research_id=research_id,
            )

            iteration += 1

            # Emit progress after iteration completes with updated coverage
            await _emit_progress(
                iteration,
                max_iterations,
                msg=f"Follow-up iteration {iteration}/{max_iterations} completed (coverage {coverage_ratio:.2%})",
            )

    # End while loop

    logger.info(
        "agentic_loop_complete",
        iterations_executed=int(iteration),
        final_coverage=float(coverage_ratio or 0.0),
        remaining_terms=len(missing_terms or []),
        paradigm=paradigm_code,
        research_id=research_id,
        followup_triggered=bool(iteration > 0),
    )

    # Final progress snap – mark loop complete
    await _emit_progress(
        max_iterations,
        max_iterations,
        msg="Agentic follow-up planning finished",
    )

    return (
        new_candidates,
        followup_results,
        coverage_ratio,
        missing_terms,
        coverage_sources,
    )


def propose_queries_enriched(
    base_query: str,
    paradigm: str,
    missing_terms: List[str],
    gap_counts: Dict[str, int],
    max_new: int = 6,
) -> List[str]:
    """
    Enrich proposals using source-type gaps and paradigm strategies.
    """
    proposals: List[str] = []
    # Base modifier pools by paradigm
    extra_by_paradigm = {
        "bernard": ["site:.edu", "arxiv", "pubmed"],
        "maeve": ["case study", "benchmark", "mckinsey OR bcg OR gartner"],
        "dolores": ["propublica", "icij", "investigation", "whistleblower"],
        "teddy": ["resources", "nonprofit", "guide"],
    }.get(paradigm, [])

    # If academic gap, bias towards scholarly modifiers
    if gap_counts.get("academic", 0) < 2:
        extra_by_paradigm = ["site:.edu", "arxiv", *extra_by_paradigm]
    # If industry gap and paradigm strategic, add consultancy lenses
    if gap_counts.get("industry", 0) < 2 and paradigm in {"maeve"}:
        extra_by_paradigm = ["mckinsey", "bcg", "gartner", *extra_by_paradigm]
    # If media/nonprofit gap and paradigm dolores/teddy, add community lenses
    if (
        (gap_counts.get("media", 0) + gap_counts.get("nonprofit", 0)) < 2
        and paradigm in {"dolores", "teddy"}
    ):
        extra_by_paradigm = [
            "investigation",
            "report",
            "community",
            *extra_by_paradigm,
        ]

    # Create proposals combining missing terms + enriched modifiers
    for term in missing_terms[: max_new * 2]:
        for m in extra_by_paradigm[:3]:
            q = f"{term} {base_query} {m}".strip()
            if q not in proposals:
                proposals.append(q)
                if len(proposals) >= max_new:
                    return proposals
    # Fallback to simple combo
    return proposals or propose_queries_from_missing(
        base_query, paradigm, missing_terms, max_new=max_new
    )
