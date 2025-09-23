from __future__ import annotations

import json
import os
from typing import Iterable, Dict

import structlog

from .types import PlannerConfig, StageName


logger = structlog.get_logger(__name__)


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
        stage_prior=dict(base.stage_prior),
    )

    if stage_order is not None:
        validated = [
            stage for stage in stage_order
            if stage in cfg.per_stage_caps
        ]
        if validated:
            cfg.stage_order = validated

    # Environment overrides
    max_candidates_env = os.getenv("UNIFIED_QUERY_MAX_VARIATIONS")
    if not max_candidates_env:
        # Legacy fallback
        max_candidates_env = os.getenv("SEARCH_QUERY_VARIATIONS_LIMIT")
    if max_candidates_env:
        try:
            cfg.max_candidates = max(1, int(max_candidates_env))
        except Exception as exc:
            logger.warning(
                "Invalid UNIFIED_QUERY_MAX_VARIATIONS override ignored",
                raw_value=max_candidates_env,
                error=str(exc),
            )

    llm_flag = os.getenv("UNIFIED_QUERY_ENABLE_LLM")
    if llm_flag is not None:
        cfg.enable_llm = llm_flag.lower() in {"1", "true", "yes"}
    else:
        # Legacy fallback
        legacy_llm = os.getenv("ENABLE_QUERY_LLM")
        if legacy_llm is not None:
            cfg.enable_llm = legacy_llm.lower() in {"1", "true", "yes"}

    follow_flag = os.getenv("UNIFIED_QUERY_ENABLE_FOLLOW_UP")
    if follow_flag is not None:
        cfg.enable_agentic = follow_flag.lower() not in {"0", "false", "no"}
    else:
        # Legacy fallback (disable flag)
        legacy_disable = os.getenv("SEARCH_DISABLE_AGENTIC")
        if legacy_disable is not None:
            cfg.enable_agentic = legacy_disable.lower() not in {
                "1", "true", "yes"
            }

    stage_order_env = os.getenv("UNIFIED_QUERY_STAGE_ORDER")
    if stage_order_env:
        parts = [
            p.strip().lower()
            for p in stage_order_env.split(",")
            if p.strip()
        ]
        validated = [
            stage for stage in parts
            if stage in cfg.per_stage_caps
        ]
        if validated:
            cfg.stage_order = validated

    for stage in cfg.per_stage_caps.keys():
        env_key = f"UNIFIED_QUERY_{stage.upper()}_CAP"
        cap = os.getenv(env_key)
        if not cap:
            continue
        try:
            cfg.per_stage_caps[stage] = max(0, int(cap))
        except Exception as exc:
            logger.warning(
                "Invalid per-stage cap override ignored",
                env_key=env_key,
                raw_value=cap,
                error=str(exc),
            )
            continue

    dedup_env = os.getenv("UNIFIED_QUERY_DEDUP_JACCARD")
    if dedup_env is not None:
        dedup_env_stripped = dedup_env.strip()
        if dedup_env_stripped:
            try:
                cfg.dedup_jaccard = max(0.0, min(1.0, float(dedup_env_stripped)))
            except Exception as exc:
                logger.warning(
                    "Invalid UNIFIED_QUERY_DEDUP_JACCARD override ignored",
                    raw_value=dedup_env,
                    error=str(exc),
                )

    # ------------------------------------------------------------------
    # Stage-prior weight overrides
    #   Accepted formats (env: UNIFIED_QUERY_STAGE_PRIORS):
    #     1) JSON string: '{"rule_based":0.9,"llm":0.8}'
    #     2) Comma list : 'rule_based:0.9,llm:0.8'
    # ------------------------------------------------------------------
    priors_raw = os.getenv("UNIFIED_QUERY_STAGE_PRIORS")
    if priors_raw:
        candidate: Dict[str, float] = {}
        try:
            if priors_raw.strip().startswith("{"):
                candidate = json.loads(priors_raw)
            else:
                for pair in priors_raw.split(","):
                    if ":" not in pair:
                        continue
                    k, v = pair.split(":", 1)
                    candidate[k.strip().lower()] = float(v)
        except Exception as exc:  # pragma: no cover – config error path
            logger.warning(
                "Invalid UNIFIED_QUERY_STAGE_PRIORS override ignored",
                raw_value=priors_raw,
                error=str(exc),
            )
            candidate = {}

        if candidate:
            # Only apply known stages and sane weight bounds (0..1.5)
            for stage, weight in candidate.items():
                if stage not in cfg.stage_prior:
                    continue
                try:
                    w = float(weight)
                    cfg.stage_prior[stage] = max(0.0, min(1.5, w))
                except Exception:
                    continue

    return cfg
