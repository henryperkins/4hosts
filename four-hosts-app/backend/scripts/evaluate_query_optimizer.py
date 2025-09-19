#!/usr/bin/env python3
"""
Evaluate LLM Query Optimizer impact on search coverage/quality.

Runs two conditions per query:
  A) Baseline (heuristic optimizer only)
  B) Baseline + LLM semantic variations (ENABLE_QUERY_LLM=1)

Collects simple metrics:
  - unique domains
  - avg credibility score
  - count of high-credibility results (>= 0.8)
  - tokenized overlap relevance (Jaccard on title+snippet vs. query tokens)

Usage:
  python -m backend.scripts.evaluate_query_optimizer \
      --queries "query1" "query2" \
      --max-results 20

  # or load last N research queries from research_store (if available)
  python -m backend.scripts.evaluate_query_optimizer --from-store 20

Notes:
  - This script is best-effort and network-bound (uses configured providers).
  - If LLM is unavailable, the B condition will skip LLM variations and report 0 lift.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from backend.services.search_apis import create_search_manager, SearchConfig
from backend.services.context_engineering import ContextEngineeringPipeline
from backend.services.classification_engine import ClassificationResult, HostParadigm, QueryFeatures
from backend.services.context_engineering import QueryOptimizer
from search.query_planner import QueryCandidate


def _tokenize(text: str) -> List[str]:
    import re
    text = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return [t for t in text.split() if len(t) > 2]


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _classify_stub(query: str) -> ClassificationResult:
    # Minimal stub sufficient for context pipeline optimize layer
    feats = QueryFeatures(
        text=query,
        tokens=_tokenize(query),
        entities=[],
        intent_signals=[],
        domain=None,
        urgency_score=0.5,
        complexity_score=0.5,
        emotional_valence=0.0,
    )
    return ClassificationResult(
        query=query,
        primary_paradigm=HostParadigm.BERNARD,
        secondary_paradigm=None,
        distribution={HostParadigm.BERNARD: 1.0},
        confidence=0.8,
        features=feats,
        reasoning={HostParadigm.BERNARD: ["stub"]},
        signals={},
    )


@dataclass
class EvalMetrics:
    unique_domains: int
    avg_cred: float
    hi_cred_count: int
    mean_relevance: float


async def _search_with_variations(query: str, variations: Dict[str, str], max_results: int) -> Tuple[EvalMetrics, List[Dict]]:
    tokens = _tokenize(query)
    async with create_search_manager() as mgr:
        cfg = SearchConfig(max_results=max_results)
        results = []
        # include primary + a few variations
        all_queries = [variations.get("primary", query)] + list({v for k, v in variations.items() if k != "primary"})
        for idx, q in enumerate(all_queries[:5]):
            try:
                planned = [
                    QueryCandidate(
                        query=q,
                        stage="context",
                        label=f"eval_{idx+1}",
                    )
                ]
                res = await mgr.search_all(planned, cfg)
                for r in res:
                    results.append({
                        "title": r.title or "",
                        "snippet": r.snippet or "",
                        "domain": r.domain or "",
                        "credibility_score": float(r.credibility_score or 0.5),
                    })
            except Exception:
                continue

    # Deduplicate by URL domain+title
    seen = set()
    dedup = []
    for r in results:
        key = (r.get("domain"), r.get("title"))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)

    domains = set([(r.get("domain") or "").lower() for r in dedup if r.get("domain")])
    creds = [float(r.get("credibility_score", 0.5)) for r in dedup]
    hi_count = sum(1 for c in creds if c >= 0.8)
    rels = []
    for r in dedup:
        rels.append(_jaccard(tokens, _tokenize((r.get("title") or "") + " " + (r.get("snippet") or ""))))
    m = EvalMetrics(
        unique_domains=len(domains),
        avg_cred=round(sum(creds) / len(creds), 3) if creds else 0.0,
        hi_cred_count=hi_count,
        mean_relevance=round(sum(rels) / len(rels), 3) if rels else 0.0,
    )
    return m, dedup


async def evaluate_query(query: str, *, max_results: int = 20) -> Dict[str, Dict]:
    # Build baseline variations via OptimizeLayer (heuristics only)
    cep = ContextEngineeringPipeline()
    cls = _classify_stub(query)

    # Force-disable LLM path
    old = os.getenv("ENABLE_QUERY_LLM")
    os.environ["ENABLE_QUERY_LLM"] = "0"
    baseline = await cep.optimize_layer.process(cls, previous_outputs=None)

    base_vars = {"primary": baseline.get("primary_query", query)}
    base_vars.update({k: v for k, v in (baseline.get("variations") or {}).items()})

    base_metrics, _ = await _search_with_variations(query, base_vars, max_results)

    # Try with LLM expansions enabled
    if old is not None:
        os.environ["ENABLE_QUERY_LLM"] = old
    else:
        os.environ.pop("ENABLE_QUERY_LLM", None)
    os.environ["ENABLE_QUERY_LLM"] = "1"

    llm_run = await cep.optimize_layer.process(cls, previous_outputs=None)
    llm_vars = {"primary": llm_run.get("primary_query", query)}
    llm_vars.update({k: v for k, v in (llm_run.get("variations") or {}).items()})

    llm_metrics, _ = await _search_with_variations(query, llm_vars, max_results)

    # Compute lifts
    def _lift(a: float, b: float) -> float:
        return round(((b - a) / a) * 100.0, 2) if a > 0 else (100.0 if b > 0 else 0.0)

    summary = {
        "baseline": base_metrics.__dict__,
        "with_llm": llm_metrics.__dict__,
        "lift_percent": {
            "unique_domains": _lift(base_metrics.unique_domains, llm_metrics.unique_domains),
            "avg_cred": round(llm_metrics.avg_cred - base_metrics.avg_cred, 3),
            "hi_cred_count": _lift(base_metrics.hi_cred_count, llm_metrics.hi_cred_count),
            "mean_relevance": round(llm_metrics.mean_relevance - base_metrics.mean_relevance, 3),
        },
    }
    return summary


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", nargs="*", help="Queries to evaluate")
    ap.add_argument("--from-store", type=int, default=0, help="Load last N queries from research_store")
    ap.add_argument("--max-results", type=int, default=20)
    args = ap.parse_args()

    queries: List[str] = []
    if args.from_store:
        try:
            from backend.services.research_store import research_store
            await research_store.initialize()
            # Best effort: collect queries from fallback/redis records
            items = list(getattr(research_store, "fallback_store", {}).values())
            for r in reversed(items[-args.from_store:]):
                q = r.get("query") or (r.get("results") or {}).get("query")
                if isinstance(q, str) and len(q) >= 8:
                    queries.append(q)
        except Exception:
            pass

    if args.queries:
        queries.extend(args.queries)

    if not queries:
        queries = [
            "best techniques to reduce cloud spend in kubernetes",
            "evidence on remote work productivity in 2023-2025",
            "how to prepare a small business for a ransomware incident",
        ]

    reports = []
    for q in queries[:10]:
        rpt = await evaluate_query(q, max_results=args.max_results)
        reports.append({"query": q, **rpt})

    # Print compact summary
    import json
    print(json.dumps({"evaluations": reports}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
