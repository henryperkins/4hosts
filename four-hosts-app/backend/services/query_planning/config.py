from __future__ import annotations

import os
from typing import Iterable

from .types import PlannerConfig, StageName


def build_planner_config(
    *,
    base: PlannerConfig | None = None,
    stage_order: Iterable[StageName] | None = None,
) -> PlannerConfig:
    cfg = PlannerConfig() if base is None else PlannerConfig(
        max_candidates=base.max_candidates,
        enable_llm=base.enable_llm,
        enable_agentic=base.enable_agentic,
        stage_order=list(base.stage_order),
        per_stage_caps=dict(base.per_stage_caps),
        dedup_jaccard=base.dedup_jaccard,
    )

    if stage_order is not None:
        validated = [stage for stage in stage_order if stage in cfg.per_stage_caps]
        if validated:
            cfg.stage_order = validated

    # Environment overrides
    max_candidates_env = os.getenv("UNIFIED_QUERY_MAX_VARIATIONS")
    if max_candidates_env:
        try:
            cfg.max_candidates = max(1, int(max_candidates_env))
        except Exception:
            pass

    llm_flag = os.getenv("UNIFIED_QUERY_ENABLE_LLM")
    if llm_flag:
        cfg.enable_llm = llm_flag.lower() in {"1", "true", "yes"}

    follow_flag = os.getenv("UNIFIED_QUERY_ENABLE_FOLLOW_UP")
    if follow_flag:
        cfg.enable_agentic = follow_flag.lower() not in {"0", "false", "no"}

    stage_order_env = os.getenv("UNIFIED_QUERY_STAGE_ORDER")
    if stage_order_env:
        parts = [p.strip().lower() for p in stage_order_env.split(",") if p.strip()]
        validated = [stage for stage in parts if stage in cfg.per_stage_caps]
        if validated:
            cfg.stage_order = validated

    for stage in cfg.per_stage_caps.keys():
        env_key = f"UNIFIED_QUERY_{stage.upper()}_CAP"
        cap = os.getenv(env_key)
        if not cap:
            continue
        try:
            cfg.per_stage_caps[stage] = max(0, int(cap))
        except Exception:
            continue

    dedup_env = os.getenv("UNIFIED_QUERY_DEDUP_JACCARD")
    if dedup_env:
        try:
            cfg.dedup_jaccard = float(dedup_env)
        except Exception:
            pass

    return cfg
