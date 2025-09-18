# LLM Query Optimizer Evaluation

This script evaluates the impact of enabling the LLM‑powered query optimizer on search coverage and quality.

- Script: `backend/scripts/evaluate_query_optimizer.py`
- What it measures per query:
  - Unique domains
  - Average credibility score
  - High‑credibility count (>= 0.8)
  - Mean relevance (Jaccard overlap between query tokens and title+snippet)

## Usage

Run with explicit queries:

```
python -m backend.scripts.evaluate_query_optimizer --queries \
  "best techniques to reduce cloud spend in kubernetes" \
  "evidence on remote work productivity in 2023-2025"
```

Or sample the last N queries from the in‑process research store (if present):

```
python -m backend.scripts.evaluate_query_optimizer --from-store 20
```

Optional flags:

- `--max-results 20` controls per‑provider page size.

The tool runs two conditions for each query:

1) Baseline (heuristic optimizer only)
2) Baseline + LLM semantic variations (`ENABLE_QUERY_LLM=1`)

It outputs a JSON with baseline, with‑LLM, and lift percentages.

> Note: If the LLM is unavailable, the script will still run and report zero lift for LLM‑only metrics.
