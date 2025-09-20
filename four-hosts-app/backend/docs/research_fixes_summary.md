# Remediation Plan – Research Orchestration Pipeline  
*(updated 2025-09-20)*

## A. Planned Code Changes (surgical)

| # | File & lines | Change Type | Patch Summary |
|---|--------------|-------------|---------------|
| 1 | `utils/normalization_helpers.py` **(NEW)** | add file | Implement `normalize_distribution_map`, `safe_credibility_summary`, and `build_classification_details`. |
| 2 | `services/research_orchestrator.py` | apply_diff | a) Replace ad-hoc distribution_map block (1240-1251) with call to `normalize_distribution_map`. <br> b) After credibility fallback (953-974) call `safe_credibility_summary`. <br> c) Remove local classification_details builder; import helper. |
| 3 | `routes/research.py` | apply_diff | Replace `_build_classification_details_for_ui` inline function with import from helpers. Remove duplicate logic. |
| 4 | `services/research_orchestrator.py` & various | apply_diff | Ensure any `apis_used` temporary sets are converted to lists immediately. |
| 5 | `utils/source_normalization.py` | minor | Ensure `categorize` failures default to `"general"`. |
| 6 | Tests `tests/test_metadata_typing_and_serialization.py` | extend | Add cases: (a) non-numeric distribution entries handled correctly, (b) credibility summary always has keys. |
| 7 | NEW tests `tests/test_distribution_map_normalization.py` | add | Edge-case coverage (None keys, invalid types). |

## B. Detailed Diff Sketch

```diff
# utils/normalization_helpers.py  (new)
+ from typing import Any, Dict, List
+ from models.paradigms import normalize_to_internal_code
+ import structlog
+ logger = structlog.get_logger(__name__)
+
+ def normalize_distribution_map(raw: Any) -> Dict[str, float]:
+     if not isinstance(raw, dict):
+         return {}
+     dist: Dict[str, float] = {}
+     for k, v in raw.items():
+         if k is None:
+             continue
+         try:
+             key = normalize_to_internal_code(k)
+             dist[key] = float(v or 0.0)
+         except (ValueError, TypeError):
+             logger.debug("invalid_dist_entry", host=k, value=v)
+     return dist
+
+ def safe_credibility_summary(cs: Dict[str, Any]) -> Dict[str, Any]:
+     cs.setdefault("average_score", 0.0)
+     cs.setdefault("score_distribution", {})
+     cs.setdefault("high_credibility_count", 0)
+     cs.setdefault("high_credibility_ratio", 0.0)
+     return cs
+
+ def build_classification_details(cls) -> Dict[str, Any]:
+     from models.context_models import ClassificationDetailsSchema
+     from models.base import HOST_TO_MAIN_PARADIGM
+     dist = {HOST_TO_MAIN_PARADIGM.get(k).value: float(v or 0.0)  # type: ignore
+             for k, v in getattr(cls, "distribution", {}).items()
+             if k in HOST_TO_MAIN_PARADIGM}
+     reasoning = {HOST_TO_MAIN_PARADIGM.get(k).value: list(r)[:4]  # type: ignore
+                  for k, r in getattr(cls, "reasoning", {}).items()
+                  if k in HOST_TO_MAIN_PARADIGM}
+     return ClassificationDetailsSchema(distribution=dist, reasoning=reasoning).model_dump()
```

(Actual diffs will be applied via `apply_diff` in Code mode.)

## C. Test Matrix

| Test | Scenario | Expected |
|------|----------|----------|
| distribution_map_normalization | mixed invalid inputs | returns cleaned dict w/o exception |
| metadata_typing_and_serialization (update) | orchestrator with bad distribution | classification_details.distribution populated & numeric |
| credibility_summary_defaults | orchestrator forced to fallback path | keys present, UI safe |
| route_classification_details | `/research/query` path | returns unified builder output identical to orchestrator |

## D. Sequence to Implement

1. **Add helper module** `utils/normalization_helpers.py`.  
2. Patch **research_orchestrator** to use helpers.  
3. Patch **routes/research.py** to import and use same helper.  
4. Adjust `apis_used` list casting.  
5. Update & add tests; run `pytest -k "metadata_typing or distribution_map"` to verify.  
6. `npm run lint` & `pytest` full suite.  
7. Commit with message:  
   `fix(metadata): centralize normalization, harden distribution map & credibility summary`.

Once approved we can switch to Code mode and apply the above diffs.