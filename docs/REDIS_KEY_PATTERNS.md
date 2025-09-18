# Redis Key Patterns (Four Hosts App)

This document standardises the **Redis** key names used by the Four Hosts backend.  
Follow these conventions in code, dashboards, and operations tooling. Older names are now **deprecated**.

| Namespace | Key Pattern | TTL | Purpose |
|-----------|-------------|-----|---------|
| Research Status | `research_status:{id}` | 10 s | Short-lived indicator of the current execution state (used by WebSocket progress updates and polling). |
| Research Results | `research_results:{id}` | 300 s | Cached, synthesised result package returned by `GET /research/results/{id}`. |
| Search Results | `search_results:{hash}` | 24 h | Cached list of SERP items for identical search queries to avoid duplicate external calls. |
| Paradigm Classification | `paradigm_classification:{query_hash}` | 7 d | Stores primary/secondary paradigm plus confidence distribution for a query. |
| Source Credibility | `source_credibility:{domain}` | 30 d | Cached credibility scores for external domains. |

## Migration Notes

* Any **legacy** references such as `research:res_<id>` or `res_status:<id>` have been removed.  
  Update scripts, tests, dashboards, and documentation accordingly.
* The backend **services/cache.py** already implements these patterns; this document merely aligns the written documentation.
* Production Redis instances should enable **AOF** persistence (`appendonly yes`) to minimise data loss for keys with longer TTLs.
