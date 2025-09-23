#!/usr/bin/env python3
"""Sequential search debugging utility.

Purpose: run each provider sequentially (no concurrency) and optionally
bypass EarlyRelevanceFilter to diagnose disappearing results or overly
strict filtering.

Usage examples:

  python -m scripts.debug_sequential_search \
      --query "2025 grid-scale energy storage innovations" \
      --max-results 8 \
      --paradigm bernard \
      --disable-early-filter \
      --theme-threshold 0.0

Environment overrides (fallback to defaults if flags absent):
  EARLY_THEME_OVERLAP_MIN          -> --theme-threshold
  DISABLE_EARLY_FILTER=1           -> --disable-early-filter
  SEARCH_PROVIDER_TIMEOUT_SEC      (ignored here; we go sequential)

Outputs a JSON + human readable summary of:
  - Raw results per provider
  - Early filter drops & reasons (if enabled)
  - Kept results with minimal fields
  - Jaccard overlap stats (query vs kept snippet/title)

This script is non-invasive: it imports existing code without modifying
core services.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Any
import structlog

# Adjust path so we can import project modules when executed directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import logging config to ensure consistent logging
from logging_config import configure_logging
configure_logging()

logger = structlog.get_logger(__name__)

from services.search_apis import create_search_manager, SearchConfig  # type: ignore
from search.query_planner import QueryCandidate
from services.research_orchestrator import EarlyRelevanceFilter  # type: ignore
try:
    from models.search import SearchResult  # type: ignore
except Exception:
    from services.search_apis import SearchResult  # type: ignore


@dataclass
class DropRecord:
    provider: str
    domain: str
    title: str
    reason: str


class InstrumentedEarlyFilter(EarlyRelevanceFilter):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def is_relevant(
        self, result: SearchResult, query: str, paradigm: str
    ) -> bool:  # type: ignore[override]
        # Copy of parent logic with added reason tracking
        title_val = (getattr(result, "title", "") or "")
        snippet_val = (getattr(result, "snippet", "") or "")
        combined_text = f"{title_val} {snippet_val}".lower()
        domain = (getattr(result, "domain", "") or "").lower()

        def fail(r: str):
            raise ValueError(r)

        # Spam
        if any(spam in combined_text for spam in self.spam_indicators):
            fail("spam")
        if domain in self.low_quality_domains:
            fail("low_quality_domain")
        if not isinstance(title_val, str) or len(title_val.strip()) < 10:
            fail("short_title")
        if not isinstance(snippet_val, str) or len(snippet_val.strip()) < 20:
            fail("short_snippet")
        non_ascii_count = sum(1 for c in combined_text if ord(c) > 127)
        if non_ascii_count > len(combined_text) * 0.3:
            fail("non_english")
        query_terms = [term.lower() for term in query.split() if len(term) > 3]
        if query_terms and not any(term in combined_text for term in query_terms):
            fail("no_query_term")
        if self._is_likely_duplicate_site(domain):
            fail("duplicate_site")
        # Paradigm specific (simplified: we reuse parent heuristics?)
        if (
            paradigm == "bernard"
            and getattr(result, "result_type", "web") == "web"
        ):
            authoritative_indicators = [
                '.edu', '.gov', 'journal', 'research', 'study', 'analysis',
                'technology', 'innovation', 'science', 'ieee', 'acm', 'mit',
                'stanford', 'harvard', 'arxiv', 'nature', 'springer'
            ]
            tech_indicators = [
                'ai', 'artificial intelligence', 'machine learning',
                'deep learning', 'neural', 'algorithm', 'technology',
                'computing', 'software', 'innovation'
            ]
            has_authority = any(ind in domain or ind in combined_text for ind in authoritative_indicators)
            has_tech = any(ind in combined_text for ind in tech_indicators)
            if not (has_authority or has_tech):
                academic_terms = ['methodology', 'hypothesis', 'conclusion', 'abstract', 'citation',
                                  'analysis', 'framework', 'approach', 'technique', 'evaluation']
                if not any(t in combined_text for t in academic_terms):
                    fail("bernard_missing_authority")
        # Theme overlap (Jaccard)
        try:
            import re as _re
            q_terms = set([t for t in _re.findall(r"[A-Za-z0-9]+", query.lower()) if len(t) > 2])
            if q_terms:
                rtoks = set([t for t in _re.findall(r"[A-Za-z0-9]+", combined_text) if len(t) > 2])
                if rtoks:
                    inter = len(q_terms & rtoks)
                    union = len(q_terms | rtoks)
                    jac = (inter / union) if union else 0.0
                    if jac < self.threshold:
                        fail(f"theme_jaccard<{self.threshold:.2f}")
        except Exception:
            pass
        return True


def compute_query_overlap(query: str, text: str) -> float:
    import re as _re
    q_terms = set([t for t in _re.findall(r"[A-Za-z0-9]+", query.lower()) if len(t) > 2])
    if not q_terms:
        return 0.0
    rtoks = set([t for t in _re.findall(r"[A-Za-z0-9]+", (text or "").lower()) if len(t) > 2])
    if not rtoks:
        return 0.0
    inter = len(q_terms & rtoks)
    union = len(q_terms | rtoks)
    return (inter / union) if union else 0.0


async def run(query: str, paradigm: str, max_results: int, disable_filter: bool, threshold: float) -> Dict[str, Any]:
    async with create_search_manager() as mgr:  # type: ignore[misc]
        cfg = SearchConfig(max_results=max_results)

        # sequential provider loop
        all_results: List[SearchResult] = []
        per_provider: Dict[str, List[SearchResult]] = {}
        for name, api in mgr.apis.items():
            try:
                planned = [
                    QueryCandidate(query=query, stage="context", label="manual")
                ]
                res = await api.search_with_variations(
                    query,
                    cfg,
                    planned=planned,
                )
            except Exception as e:
                logger.warning("Provider failed", provider=name, error=str(e))
                res = []
            per_provider[name] = res
            all_results.extend(res)

        drop_records: List[DropRecord] = []
        kept: List[SearchResult] = []
        if disable_filter:
            kept = all_results
        else:
            filt = InstrumentedEarlyFilter(threshold)
            for r in all_results:
                try:
                    if filt.is_relevant(r, query, paradigm):
                        kept.append(r)
                    else:
                        drop_records.append(DropRecord(r.source, getattr(r, 'domain', '') or '', r.title[:120], 'unknown'))
                except ValueError as ve:
                    drop_records.append(DropRecord(r.source, getattr(r, 'domain', '') or '', r.title[:120], str(ve)))
                except Exception as e:
                    drop_records.append(DropRecord(r.source, getattr(r, 'domain', '') or '', r.title[:120], f"exception:{e}"))

        # build histogram
        reason_hist: Dict[str, int] = {}
        for d in drop_records:
            reason_hist[d.reason] = reason_hist.get(d.reason, 0) + 1

        # Overlap stats for kept
        overlaps: List[float] = []
        for r in kept:
            overlaps.append(compute_query_overlap(query, f"{r.title} {r.snippet}"))
        overlap_avg = sum(overlaps)/len(overlaps) if overlaps else 0.0

        # Compact kept view
        compact_kept = [
            {
                "provider": r.source,
                "title": r.title,
                "domain": getattr(r, 'domain', ''),
                "overlap": round(compute_query_overlap(query, f"{r.title} {r.snippet}"), 4),
                "url": r.url,
            }
            for r in kept
        ]

        # Effective state snapshot for diagnostics
        try:
            domain_block_enabled = os.getenv("EARLY_FILTER_BLOCK_DOMAINS", "1").lower() in {"1", "true", "yes", "on"}
        except Exception:
            domain_block_enabled = True
        keep_rate = (len(kept) / float(len(all_results))) if all_results else 0.0

        report = {
            "report_version": 1,
            "query": query,
            "paradigm": paradigm,
            "max_results": max_results,
            "providers": {k: len(v) for k, v in per_provider.items()},
            "total_raw": len(all_results),
            "kept": len(kept),
            "dropped": len(drop_records),
            "keep_rate": round(keep_rate, 4),
            "drop_reason_histogram": reason_hist,
            "avg_overlap_kept": round(overlap_avg, 4),
            "early_filter_disabled": bool(disable_filter),
            "filter_threshold_effective": (threshold if not disable_filter else None),
            "domain_block_enabled": (domain_block_enabled if not disable_filter else None),
            "kept_samples": compact_kept[:20],
            "kept_urls": [r.url for r in kept if getattr(r, "url", "")],
        }

        # Log structured output for analysis & emit JSON for CLI
        logger.info(
            "Search debug report generated",
            query=query,
            paradigm=paradigm,
            total_raw=len(all_results),
            kept=len(kept),
            keep_rate=round(keep_rate, 4),
        )
        logger.info("summary", report=report)

        if drop_records:
            logger.info("Drop reasons summary", reasons=reason_hist)

        return report


def main():
    parser = argparse.ArgumentParser(description="Sequential search debugger")
    parser.add_argument("--query", required=False)
    parser.add_argument("--queries", nargs="+", help="Multiple queries (overrides --query)")
    parser.add_argument("--paradigm", default="bernard")
    parser.add_argument("--max-results", type=int, default=8)
    parser.add_argument("--disable-early-filter", action="store_true")
    parser.add_argument("--theme-threshold", type=float, default=None, help="Override EARLY_THEME_OVERLAP_MIN")
    parser.add_argument("--out", help="Write aggregated JSON to file")
    parser.add_argument("--no-domain-block", action="store_true", help="Disable domain blocklist inside EarlyRelevanceFilter")
    parser.add_argument("--dedupe-across-queries", action="store_true", help="Aggregate unique kept URLs across all queries")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first query error")
    parser.add_argument("--min-raw", type=int, default=0, help="Mark queries as starved if total_raw < N")
    args = parser.parse_args()

    # Require either --query or --queries
    if not args.query and not args.queries:
        parser.print_usage()
        logger.error("Argument validation failed: missing query input", \
                     provided_query=args.query, provided_queries=args.queries)
        sys.exit(2)

    thr_env = os.getenv("EARLY_THEME_OVERLAP_MIN")
    if args.theme_threshold is not None:
        threshold = args.theme_threshold
    elif thr_env is not None:
        try:
            threshold = float(thr_env)
        except Exception:
            threshold = 0.08
    else:
        threshold = 0.08

    # Environment flag may override the CLI
    if os.getenv("DISABLE_EARLY_FILTER", "0").lower() in {"1", "true", "yes"}:
        args.disable_early_filter = True

    # Optional: disable domain blocklist via flag
    if getattr(args, "no_domain_block", False):
        os.environ["EARLY_FILTER_BLOCK_DOMAINS"] = "0"

    # Notice when flags are moot due to full filter disable
    if args.disable_early_filter and (args.theme_threshold is not None or
                                      getattr(args, "no_domain_block", False)):
        logger.warning("Conflicting flags",
                      message="--disable-early-filter is set; "
                              "--theme-threshold/--no-domain-block have no effect")

    # Multi-query aggregated path
    if args.queries:
        all_reports: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        for q in args.queries:
            logger.info("Processing query", query=q, paradigm=args.paradigm)
            logger.info("\n##### QUERY", query=q)
            try:
                rep = asyncio.run(run(q, args.paradigm, args.max_results,
                                    args.disable_early_filter, threshold))
            except Exception as e:
                err = {"query": q, "error": str(e)}
                errors.append(err)
                logger.error("Query failed", query=q, error=str(e))
                if args.fail_fast:
                    sys.exit(1)
                continue
            # Optional starvation marker based on --min-raw
            if getattr(args, "min_raw", 0) and rep.get("total_raw", 0) < int(args.min_raw):
                rep["starved"] = True
            all_reports.append(rep)

        total_raw = sum(r.get("total_raw", 0) for r in all_reports)
        total_kept = sum(r.get("kept", 0) for r in all_reports)

        reason_totals: Dict[str, int] = {}
        for r in all_reports:
            for k, v in (r.get("drop_reason_histogram", {}) or {}).items():
                reason_totals[k] = reason_totals.get(k, 0) + int(v)
        total_dropped_all = sum(r.get("dropped", 0) for r in all_reports)
        reason_totals_percent = {
            k: round((v / float(total_dropped_all)), 4) if total_dropped_all else 0.0
            for k, v in reason_totals.items()
        }

        agg: Dict[str, Any] = {
            "report_version": 1,
            "queries": all_reports,
            "total_raw": total_raw,
            "total_kept": total_kept,
            "overall_keep_rate": round((total_kept / float(total_raw)) if total_raw else 0.0, 4),
            "reason_totals": reason_totals,
            "reason_totals_percent": reason_totals_percent,
        }

        if getattr(args, "dedupe_across_queries", False):
            all_kept_urls: List[str] = []
            for r in all_reports:
                all_kept_urls.extend(r.get("kept_urls", []))
            all_kept_urls = [u for u in all_kept_urls if isinstance(u, str) and u]
            by_url: Dict[str, int] = {}
            for u in all_kept_urls:
                by_url[u] = by_url.get(u, 0) + 1
            # Appearance histogram
            freq: Dict[int, int] = {}
            for cnt in by_url.values():
                freq[cnt] = freq.get(cnt, 0) + 1
            unique_urls = set(by_url.keys())
            total_kept_deduped = len(unique_urls)
            agg["unique_kept_urls"] = total_kept_deduped
            agg["kept_url_appearance_distribution"] = freq
            agg["overall_keep_rate_deduped"] = round((total_kept_deduped / float(total_raw)) if total_raw else 0.0, 4)
            agg["top_repeated_urls"] = sorted(by_url.items(), key=lambda kv: kv[1], reverse=True)[:10]

        # Top queries by low keep_rate for triage focus
        try:
            sorted_low = sorted(
                [r for r in all_reports if "keep_rate" in r],
                key=lambda r: r.get("keep_rate", 0.0)
            )[: min(5, len(all_reports))]
            agg["top_queries_low_keep"] = [
                {
                    "query": r.get("query", ""),
                    "keep_rate": r.get("keep_rate", 0.0),
                    "total_raw": r.get("total_raw", 0),
                    "dropped": r.get("dropped", 0),
                }
                for r in sorted_low
            ]
        except Exception:
            pass

        if errors:
            agg["errors"] = errors

        if args.out:
            try:
                with open(args.out, "w") as f:
                    json.dump(agg, f, indent=2)
                logger.info("Aggregated results written", output_file=args.out)
                logger.info("Aggregated JSON written", path=args.out)
            except Exception as e:
                logger.error("Failed to write output", file=args.out, error=str(e))

        logger.info("\n=== AGGREGATED ===")
        logger.info(json.dumps(agg, indent=2))
        sys.exit(0)

    # Single-query path
    _ = asyncio.run(run(args.query, args.paradigm, args.max_results, args.disable_early_filter, threshold))
    sys.exit(0)


if __name__ == "__main__":
    main()
