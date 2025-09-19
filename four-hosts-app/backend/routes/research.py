"""
Research routes for the Four Hosts Research API
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional, cast, Dict

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Header, Response, Request

from models.research import (
    ResearchQuery,
    ResearchOptions,
    ResearchDeepQuery,
)
from models.base import (
    ResearchDepth,
    ResearchStatus,
    UserRole,
    HOST_TO_MAIN_PARADIGM,
    ParadigmClassification,
    HostParadigm,
)
from core.dependencies import get_current_user
from services.research_store import research_store
from services.websocket_service import (
    progress_tracker as _ws_progress_tracker,
    WSEventType,
    WSMessage,
)
from services.enhanced_integration import (
    enhanced_classification_engine as classification_engine,
)
from services.context_engineering import context_pipeline
from services.research_orchestrator import research_orchestrator
from services.background_llm import background_llm_manager
from services.research_persistence import (
    persist_completion,
    persist_failure,
    record_submission,
)
from core.config import SYNTHESIS_MAX_LENGTH_DEFAULT, ENABLE_MESH_NETWORK
from services.result_adapter import ResultAdapter
from models.result_models import ResearchFinalResult
from services.mesh_network import mesh_negotiator
import html as _html
import re as _re
from models.base import Paradigm
from models.context_models import ClassificationDetailsSchema
from services.webhook_manager import WebhookEvent, WebhookManager
from services.rate_limiter import RateLimitExceeded
from utils.error_handling import log_exception

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/research", tags=["research"])


# Mock services for now - these will be injected
progress_tracker = _ws_progress_tracker
webhook_manager: WebhookManager | None = None


def _build_classification_details_for_ui(cls: Any) -> Dict[str, Any]:
    """
    Normalize a classification object into the UI-facing classification_details
    payload using the explicit ClassificationDetailsSchema to guarantee a
    mapping-like shape and JSON-serializable values.
    """
    try:
        dist_raw = getattr(cls, "distribution", {}) or {}
        rn_raw = getattr(cls, "reasoning", {}) or {}

        # Guard against unexpected shapes
        if not isinstance(dist_raw, dict):
            dist_raw = {}
        if not isinstance(rn_raw, dict):
            rn_raw = {}

        # Build distribution mapping without long lines
        distribution: Dict[str, float] = {}
        for p, v in dist_raw.items():
            key = HOST_TO_MAIN_PARADIGM[p].value  # type: ignore[index]
            distribution[key] = float(v or 0.0)

        # Build reasoning mapping (limit to 4 items per paradigm)
        reasoning: Dict[str, list[str]] = {}
        for p, r in rn_raw.items():
            key = HOST_TO_MAIN_PARADIGM[p].value  # type: ignore[index]
            steps = list((r or [])[:4])
            reasoning[key] = steps

        schema = ClassificationDetailsSchema(
            distribution=distribution,
            reasoning=reasoning,
        )
        return schema.model_dump()
    except Exception:
        # Best effort – fall back to empty schema-compatible object
        return ClassificationDetailsSchema().model_dump()


async def execute_real_research(
    research_id: str,
    research: ResearchQuery,
    user_id: str,
    user_role: UserRole | str = UserRole.PRO,
    limiter=None,
    limiter_identifier: str | None = None,
): 
    """Execute the full research pipeline and persist results.

    Steps:
    - Classify query (enhanced engine)
    - Context engineering (W-S-C-I pipeline)
    - Search + retrieval via UnifiedResearchOrchestrator (with WS progress)
    - Paradigm-aware synthesis (enhanced answer orchestrator)
    - Persist results to research_store and broadcast completion
    """

    # Collect non-fatal warnings to include in metadata
    _warnings: list[dict[str, str]] = []

    async def check_cancelled():
        """Check if research has been cancelled"""
        research_data = await research_store.get(research_id)
        return research_data and research_data.get("status") == ResearchStatus.CANCELLED

    # Track concurrent usage if a limiter was provided (enforce U-014)
    if limiter is not None and limiter_identifier:
        try:
            await limiter.increment_concurrent(limiter_identifier)
        except Exception as e:
            log_exception("limiter.increment_concurrent", e, research_id=research_id)

    try:
        # Mark in-progress
        await research_store.update_field(research_id, "status", ResearchStatus.IN_PROGRESS)

        # 1) Classification
        if await check_cancelled():
            logger.info(f"Research {research_id} cancelled before classification")
            return

        if progress_tracker:
            await progress_tracker.update_progress(research_id, phase="classification")

        # Respect per-request AI classification toggle (LLM on/off)
        _prev_use_llm = None
        classification_error: Exception | None = None
        cls = None
        prev_llm_weight: float | None = None
        try:
            _prev_use_llm = getattr(classification_engine.classifier, "use_llm", None)
            desired = bool(getattr(getattr(research, "options", object()), "enable_ai_classification", _prev_use_llm if _prev_use_llm is not None else True))
            if _prev_use_llm is not None:
                classification_engine.classifier.use_llm = desired  # type: ignore[attr-defined]
        except Exception as e:
            log_exception("metadata.override_flag", e, research_id=research_id)
        try:
            cls = await classification_engine.classify_query(research.query)
        except Exception as e:
            classification_error = e
            log_exception("classification.llm", e, research_id=research_id)

            # Retry with rule-based classification only when LLM path fails
            try:
                classifier = getattr(classification_engine, "classifier", None)
                if classifier and getattr(classifier, "use_llm", False):
                    prev_llm_weight = getattr(classifier, "llm_weight", None)
                    classifier.use_llm = False  # type: ignore[attr-defined]
                    try:
                        classifier.llm_weight = 0.0  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    logger.warning(
                        "Retrying classification without LLM after JSON parse failure (research_id=%s)",
                        research_id,
                    )
                    cls = await classifier.classify(research.query, research_id)
                    classification_error = None
            except Exception as retry_exc:
                log_exception("classification.retry_rule_only", retry_exc, research_id=research_id)
                classification_error = retry_exc
        finally:
            # Restore classifier configuration
            try:
                if prev_llm_weight is not None:
                    classification_engine.classifier.llm_weight = prev_llm_weight  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                if _prev_use_llm is not None:
                    classification_engine.classifier.use_llm = _prev_use_llm  # type: ignore[attr-defined]
            except Exception:
                pass

        if classification_error or cls is None:
            error_payload = {
                "stage": "classification",
                "message": str(classification_error) if classification_error else "classification_failed",
            }
            await research_store.update_fields(
                research_id,
                {
                    "status": ResearchStatus.FAILED,
                    "error": error_payload,
                },
            )
            if progress_tracker:
                try:
                    await progress_tracker.fail_research(
                        research_id,
                        error="classification_failed",
                        error_details=error_payload,
                    )
                except Exception:
                    pass
            return

        # Reinforce stored classification (from submission) to avoid divergence
        try:
            stored = await research_store.get(research_id)
            pc = (stored or {}).get("paradigm_classification") if stored else None
            if isinstance(pc, dict):
                primary_data = pc.get("primary")
                if isinstance(primary_data, dict):
                    primary_ui = primary_data.get("paradigm")
                else:
                    primary_ui = None
            elif isinstance(pc, str):
                # Handle string case - may be a paradigm name directly
                primary_ui = pc
            else:
                primary_ui = None

            if primary_ui:
                    from models.base import HOST_TO_MAIN_PARADIGM as _HTM
                    # Map UI paradigm (Paradigm enum value) back to HostParadigm for services
                    host_primary = None
                    for hp, ui in _HTM.items():
                        if str(ui.value) == str(primary_ui):
                            host_primary = hp
                            break
                    if host_primary and host_primary != getattr(cls, "primary_paradigm", None):
                        prev_primary = getattr(cls, "primary_paradigm", None)
                        cls.primary_paradigm = host_primary  # type: ignore[attr-defined]
                        # Preserve previous as secondary when available, but do not mutate confidence/distribution
                        try:
                            if prev_primary:
                                cls.secondary_paradigm = prev_primary  # type: ignore[attr-defined]
                        except Exception:
                            pass
        except Exception as e:
            log_exception("metadata.classification_details", e, research_id=research_id)

        # Stream classification summary
        try:
            if progress_tracker:
                primary = getattr(cls, "primary_paradigm", None)
                secondary = getattr(cls, "secondary_paradigm", None)
                confidence = float(getattr(cls, "confidence", 0.0) or 0.0)
                dist = getattr(cls, "distribution", {}) or {}
                # Build a compact distribution preview
                dist_preview = ", ".join(
                    f"{p.name.lower()}: {dist.get(p, 0.0):.2f}" for p in list(dist.keys())[:2]
                ) if hasattr(primary, 'name') else ""
                await progress_tracker.update_progress(
                    research_id,
                    message=(
                        f"Classification → primary: {getattr(primary, 'name', primary)}, "
                        f"secondary: {getattr(secondary, 'name', secondary) or 'n/a'}, "
                        f"confidence: {confidence:.2f}"
                        + (f"; dist: {dist_preview}" if dist_preview else "")
                    ),
                )
        except Exception as e:
            log_exception("progress.agent_trace_emit", e, research_id=research_id)

        # 2) Context Engineering
        if await check_cancelled():
            logger.info(f"Research {research_id} cancelled before context engineering")
            return

        if progress_tracker:
            await progress_tracker.update_progress(research_id, phase="context")

        # Use global pipeline to accumulate metrics
        ce = await context_pipeline.process_query(cls, research_id=research_id)

        # Stream context engineering summary for transparency
        try:
            if progress_tracker and ce:
                doc_focus = getattr(ce.write_output, "documentation_focus", "") if hasattr(ce, "write_output") else ""
                comp_ratio = getattr(ce.compress_output, "compression_ratio", None) if hasattr(ce, "compress_output") else None
                token_budget = getattr(ce.compress_output, "token_budget", None) if hasattr(ce, "compress_output") else None
                iso_strategy = getattr(ce.isolate_output, "isolation_strategy", "") if hasattr(ce, "isolate_output") else ""
                searches = len(getattr(ce.select_output, "search_queries", []) or []) if hasattr(ce, "select_output") else 0
                rewrites = len((getattr(ce, "rewrite_output", {}) or {}).get("alternatives", [])) if hasattr(ce, "rewrite_output") else 0
                variations = len(((getattr(ce, "optimize_output", {}) or {}).get("variations") or {})) if hasattr(ce, "optimize_output") else 0
                await progress_tracker.update_progress(
                    research_id,
                    message=(
                        "Context: "
                        + (f"focus='{doc_focus[:40]}', " if doc_focus else "")
                        + (f"search_queries={searches}, " if searches else "")
                        + (f"rewrites={rewrites}, " if rewrites else "")
                        + (f"variations={variations}, " if variations else "")
                        + (f"compression={comp_ratio*100:.0f}%, " if isinstance(comp_ratio, (int, float)) else "")
                        + (f"token_budget={token_budget}, " if token_budget else "")
                        + (f"isolation='{iso_strategy}'" if iso_strategy else "")
                    ).rstrip(', '),
                )
        except Exception:
            pass

        # 3) Search, Retrieval, and Synthesis via unified orchestrator
        if await check_cancelled():
            logger.info(f"Research {research_id} cancelled before search/retrieval")
            return

        if progress_tracker:
            # Align with FE "Search & Retrieval" phase key
            await progress_tracker.update_progress(research_id, phase="search")

        class _UserCtxShim:
            def __init__(self, role: str, source_limit: int, language: str, region: str, enable_real_search: bool, depth: str, query_concurrency: int):
                self.role = role
                self.source_limit = source_limit
                self.language = language
                self.region = region
                self.enable_real_search = enable_real_search
                self.depth = depth
                self.query_concurrency = query_concurrency

        # Resolve user role string (e.g., 'PRO') for orchestrator context
        user_role_name = user_role.name if isinstance(user_role, UserRole) else str(user_role).upper()
        max_sources = int(research.options.max_sources)
        depth_name = research.options.depth.value if hasattr(research, 'options') else 'standard'
        # Depth-aware tuning of sources and concurrency
        if depth_name == 'quick':
            max_sources = max(10, int(max_sources * 0.5))
            query_concurrency = 2
        elif depth_name == 'deep':
            max_sources = min(200, int(max_sources * 2))
            query_concurrency = 6
        else:
            query_concurrency = 0  # use default from env

        user_ctx = _UserCtxShim(
            user_role_name,
            max_sources,
            getattr(research.options, "language", "en"),
            getattr(research.options, "region", "us"),
            bool(getattr(research.options, "enable_real_search", True)),
            depth_name,
            query_concurrency,
        )

        # Determine deep research activation:
        # - Only auto-enable when the request explicitly asks for deep research
        #   (depth == deep_research). Do NOT enable deep when real search is
        #   disabled — deep research augments search, it does not replace it.
        enable_deep = (research.options.depth == ResearchDepth.DEEP_RESEARCH)

        orch_resp = await research_orchestrator.execute_research(
            classification=cls,  # type: ignore[arg-type]
            context_engineered=ce,  # legacy shape is supported by the orchestrator  # type: ignore[arg-type]
            user_context=user_ctx,
            progress_callback=progress_tracker,
            research_id=research_id,
            enable_deep_research=enable_deep,
            deep_research_mode=None,
            synthesize_answer=True,
            answer_options={"research_id": research_id, "max_length": SYNTHESIS_MAX_LENGTH_DEFAULT},
        )

        # Normalize answer for frontend
        answer_obj = orch_resp.get("answer")
        # Normalize answer to frontend-friendly primitives
        sections_payload = []
        citations_payload = []
        action_items_payload = []
        summary_text = ""
        if answer_obj:
            try:
                summary_text = (
                    getattr(answer_obj, "summary", None)
                    or getattr(answer_obj, "content_md", None)
                    or ""
                )

                # Sections
                raw_sections = getattr(answer_obj, "sections", []) or []
                for s in raw_sections:
                    try:
                        sections_payload.append(
                            {
                                "title": getattr(s, "title", ""),
                                "paradigm": getattr(s, "paradigm", "bernard"),
                                "content": getattr(s, "content", ""),
                                "confidence": float(getattr(s, "confidence", 0.8) or 0.8),
                                # Approximate: use number of citations referenced in the section
                                "sources_count": len(getattr(s, "citations", []) or []),
                                "citations": list(getattr(s, "citations", []) or []),
                                "key_insights": list(getattr(s, "key_insights", []) or []),
                            }
                        )
                    except Exception:
                        continue

                # Citations (answer_obj.citations is a dict[str, Citation])
                raw_citations = getattr(answer_obj, "citations", {}) or {}
                if isinstance(raw_citations, dict):
                    primary_attr = getattr(cls, "primary_paradigm", None)
                    primary_host = primary_attr if isinstance(primary_attr, HostParadigm) else HostParadigm.BERNARD
                    primary_paradigm = HOST_TO_MAIN_PARADIGM.get(primary_host, Paradigm.BERNARD).value
                    # Build domain lookup from orchestrator results for category/explanation enrichment
                    domain_map = {}
                    try:
                        raw_results = (orch_resp.get("results", []) or [])
                        for raw in raw_results:
                            try:
                                r = ResultAdapter(raw)
                                d = (r.domain or (r.metadata or {}).get("domain") or "")
                                d = (d or "").lower()
                                if d and d not in domain_map:
                                    md = r.metadata or {}
                                    domain_map[d] = {
                                        "source_category": md.get("source_category"),
                                        "credibility_explanation": md.get("credibility_explanation"),
                                        "credibility_score": float(r.credibility_score if r.credibility_score is not None else 0.5),
                                    }
                            except Exception:
                                continue
                    except Exception:
                        domain_map = {}
                    for c in raw_citations.values():
                        try:
                            dom = (getattr(c, "domain", "") or "").lower()
                            cat = None
                            expl = None
                            cred_val = float(getattr(c, "credibility_score", 0.5) or 0.5)
                            if dom and dom in domain_map:
                                cat = domain_map[dom].get("source_category")
                                expl = domain_map[dom].get("credibility_explanation")
                                # Prefer backend credibility score if better populated
                                cred_val = float(domain_map[dom].get("credibility_score") or cred_val)
                            citations_payload.append(
                                {
                                    "id": getattr(c, "id", ""),
                                    "source": getattr(c, "domain", ""),
                                    "title": getattr(c, "source_title", ""),
                                    "url": getattr(c, "source_url", ""),
                                    "credibility_score": cred_val,
                                    "source_category": cat,
                                    "credibility_explanation": expl,
                                    "paradigm_alignment": primary_paradigm,
                                }
                            )
                        except Exception:
                            continue
                # Action items: ensure required FE fields with sensible defaults
                try:
                    raw_actions = list(getattr(answer_obj, "action_items", []) or [])
                    if not isinstance(raw_actions, list):
                        raw_actions = []
                    # Reuse primary paradigm for defaulting
                    primary_attr = getattr(cls, "primary_paradigm", None)
                    primary_host = primary_attr if isinstance(primary_attr, HostParadigm) else HostParadigm.BERNARD
                    primary_paradigm = HOST_TO_MAIN_PARADIGM.get(primary_host, Paradigm.BERNARD).value
                    for it in raw_actions:
                        if not isinstance(it, dict):
                            continue
                        item = dict(it)
                        # Map backend 'timeline' -> FE 'timeframe' if missing
                        if "timeframe" not in item and item.get("timeline"):
                            item["timeframe"] = item.get("timeline")
                        # Default paradigm when unspecified (for UI styling)
                        if not item.get("paradigm"):
                            item["paradigm"] = primary_paradigm
                        action_items_payload.append(item)
                except Exception:
                    action_items_payload = []
            except Exception:
                pass

        answer_payload = {
            "summary": summary_text,
            "sections": sections_payload,
            "action_items": action_items_payload,
            "citations": citations_payload,
            "metadata": getattr(answer_obj, "metadata", {}) or {},
        }

        # If orchestrator signaled insufficient data and no summary exists, add a user-facing note
        try:
            md_insuff = (orch_resp.get("metadata") or {}).get("insufficient_data")
            if (not answer_payload.get("summary")) and isinstance(md_insuff, dict):
                reason = md_insuff.get("reason", "insufficient_data")
                cnt = md_insuff.get("results_count")
                answer_payload["summary"] = (
                    f"Insufficient high-quality evidence to synthesize an answer. "
                    f"Reason: {reason.replace('_', ' ')}"
                    + (f"; results={cnt}" if cnt is not None else "")
                )
        except Exception:
            pass

        # Convert sources from orchestrator response results (already compressed/normalized)
        sources_payload = []
        for raw in orch_resp.get("results", []) or []:
            try:
                r = ResultAdapter(raw)
                # Extra defensive cleanup in case any provider markup slipped through
                title = _re.sub(r"<[^>]+>", " ", _html.unescape(r.title or "")).strip()
                snippet = _re.sub(r"<[^>]+>", " ", _html.unescape(r.snippet or "")).strip()
                md = r.metadata or {}
                domain = (r.domain or md.get("domain") or "")
                sources_payload.append(
                    {
                        "title": title,
                        "url": r.url or md.get("url", "") or "",
                        "snippet": snippet,
                        "domain": domain,
                        "credibility_score": float(r.credibility_score if r.credibility_score is not None else 0.5),
                        "published_date": md.get("published_date"),
                        "source_type": md.get("result_type", "web"),
                        "source_category": md.get("source_category"),
                        "credibility_explanation": md.get("credibility_explanation"),
                    }
                )
            except Exception:
                continue

        # Build paradigm analysis summary for UI
        paradigm_analysis = {
            "primary": {
                "paradigm": HOST_TO_MAIN_PARADIGM.get(cast(HostParadigm, getattr(cls, "primary_paradigm", HostParadigm.BERNARD)), Paradigm.BERNARD).value,
                "confidence": float(getattr(cls, "confidence", 0.75) or 0.75),
                "approach": getattr(ce.write_output, "documentation_focus", "")[:140] if hasattr(ce, "write_output") else "",
                "focus": ", ".join((getattr(ce.write_output, "key_themes", []) or [])[:3]) if hasattr(ce, "write_output") else "",
            }
        }

        # Optional: include secondary paradigm summary in analysis
        if getattr(cls, "secondary_paradigm", None):
            paradigm_analysis["secondary"] = {
                "paradigm": HOST_TO_MAIN_PARADIGM.get(cast(HostParadigm, getattr(cls, "secondary_paradigm", HostParadigm.BERNARD)), Paradigm.BERNARD).value,
                "confidence": float(getattr(cls, "confidence", 0.6) or 0.6) * 0.8,
                "approach": getattr(ce.write_output, "documentation_focus", "")[:120] if hasattr(ce, "write_output") else "",
                "focus": ", ".join((getattr(ce.write_output, "key_themes", []) or [])[:2]) if hasattr(ce, "write_output") else "",
            }

        # Aggregate metadata
        exec_meta = orch_resp.get("metadata", {}) or {}
        # Compose SSOTA context_layers summary
        # Compose context layers with rewrite/optimize details
        context_layers = {
            "write_focus": getattr(ce.write_output, "documentation_focus", "") if hasattr(ce, "write_output") else "",
            "compression_ratio": float(getattr(ce.compress_output, "compression_ratio", 0.0) or 0.0) if hasattr(ce, "compress_output") else 0.0,
            "token_budget": int(getattr(ce.compress_output, "token_budget", 0) or 0) if hasattr(ce, "compress_output") else 0,
            "isolation_strategy": getattr(ce.isolate_output, "isolation_strategy", "") if hasattr(ce, "isolate_output") else "",
            "search_queries_count": len(getattr(ce.select_output, "search_queries", []) or []) if hasattr(ce, "select_output") else 0,
            "layer_times": getattr(ce, "layer_durations", {}) if hasattr(ce, "layer_durations") else {},
            "budget_plan": getattr(getattr(ce, "compress_output", object()), "budget_plan", {}) if hasattr(ce, "compress_output") else {},
            # New rewrite/optimize details
            "rewrite_primary": (getattr(ce, "rewrite_output", {}) or {}).get("rewritten", ""),
            "rewrite_alternatives": len(((getattr(ce, "rewrite_output", {}) or {}).get("alternatives") or [])),
            "optimize_primary": (getattr(ce, "optimize_output", {}) or {}).get("primary_query", ""),
            "optimize_variations_count": len(((getattr(ce, "optimize_output", {}) or {}).get("variations") or {})),
            "refined_queries_count": len(getattr(ce, "refined_queries", []) or []),
            # Isolation layer findings summary (lightweight)
            "isolated_findings": {
                "focus_areas": list(getattr(getattr(ce, "isolate_output", object()), "focus_areas", []) or []),
                "patterns": len(getattr(getattr(ce, "isolate_output", object()), "extraction_patterns", []) or []),
            },
        }

        # Attach agent trace from orchestrator if present
        agent_trace = list(exec_meta.get("agent_trace", []) or []) if isinstance(exec_meta, dict) else []

        # Compute quality metrics: actionable content and bias check
        try:
            # Actionable content heuristic
            ai_count = len(answer_payload.get("action_items", []) or [])
            sec_key_insights = 0
            try:
                for s in (answer_payload.get("sections", []) or []):
                    sec_key_insights += len(s.get("key_insights", []) or [])
            except Exception:
                pass
            actionable_count = ai_count + sec_key_insights
            structural_items = len(answer_payload.get("sections", []) or [])
            total_signals = max(1, actionable_count + structural_items)
            actionable_ratio = actionable_count / float(total_signals)

            # Bias/diversity heuristic over sources
            domains = [s.get("domain", "").lower() for s in sources_payload if s.get("domain")]
            total_sources = len(sources_payload)
            unique_domains = len(set(domains)) if domains else 0
            domain_diversity = (unique_domains / total_sources) if total_sources else 0.0
            # dominant domain share
            dominant_share = 0.0
            dominant_domain = None
            if domains:
                from collections import Counter
                ctr = Counter(domains)
                dominant_domain, dom_count = ctr.most_common(1)[0]
                dominant_share = dom_count / float(total_sources)
            # source type diversity
            types = [s.get("source_type", "web") for s in sources_payload]
            unique_types = len(set(types)) if types else 0
            bias_ok = (domain_diversity >= 0.6 and dominant_share <= 0.4 and unique_types >= 2)
        except Exception as e:
            actionable_ratio = 0.0
            bias_ok = False
            domain_diversity = 0.0
            dominant_domain = None
            dominant_share = 0.0
            unique_types = 0
            log_exception("metrics.compute_failed", e, research_id=research_id)
            _warnings.append({"code": "metrics.compute_failed", "message": str(e)})

        # Compute paradigm fit (confidence and margin)
        try:
            dist = getattr(cls, "distribution", {}) or {}
            # Distribution may map HostParadigm -> float; convert to floats list
            vals = list(dist.values())
            if len(vals) >= 2:
                top = sorted(vals, reverse=True)[:2]
                margin = float(top[0] - top[1])
            else:
                margin = float(getattr(cls, "confidence", 0.0) or 0.0)
        except Exception as e:
            margin = 0.0
            log_exception("metrics.paradigm_margin", e, research_id=research_id)

        metadata = {
            "total_sources_analyzed": len(sources_payload),
            "high_quality_sources": sum(1 for s in sources_payload if s.get("credibility_score", 0) >= 0.7),
            "search_queries_executed": context_layers["search_queries_count"],
            "processing_time_seconds": float(exec_meta.get("processing_time", 0.0) or 0.0),
            "synthesis_quality": getattr(answer_obj, "synthesis_quality", None),
            "paradigms_used": [paradigm_analysis["primary"]["paradigm"]],
            "research_depth": research.options.depth.value if hasattr(research, "options") else "standard",
            "context_layers": context_layers,
            "agent_trace": agent_trace,
            "contradictions": exec_meta.get("contradictions", {"count": 0, "examples": []}),
            "credibility_summary": exec_meta.get("credibility_summary"),
            "category_distribution": exec_meta.get("category_distribution"),
            "bias_distribution": exec_meta.get("bias_distribution"),
            # Quality checks
            "actionable_content_ratio": actionable_ratio,
            "bias_check": {
                "balanced": bias_ok,
                "domain_diversity": domain_diversity,
                "dominant_domain": dominant_domain,
                "dominant_share": dominant_share,
                "unique_types": unique_types,
            },
            # Paradigm fit metric for UI
            "paradigm_fit": {
                "primary": paradigm_analysis["primary"]["paradigm"],
                "confidence": float(getattr(cls, "confidence", 0.0) or 0.0),
                "margin": margin,
            },
        }

        # Record override info in metadata for transparency (if any)
        try:
            override_ui = getattr(getattr(research, "options", object()), "paradigm_override", None)
            if override_ui:
                metadata["override"] = {
                    "applied": True,
                    "paradigm": getattr(override_ui, "value", str(override_ui)),
                }
        except Exception:
            pass

        # Add detailed classification breakdown for UI/analytics
        try:
            metadata["classification_details"] = _build_classification_details_for_ui(cls)
        except Exception:
            pass

        # Stream a compact agent trace snapshot for visibility
        try:
            if progress_tracker:
                trace = list(exec_meta.get("agent_trace", []) or [])
                for entry in trace[:5]:
                    if not isinstance(entry, dict):
                        continue
                    action = entry.get("action") or entry.get("name") or "step"
                    duration = entry.get("duration_ms") or entry.get("duration")
                    message = f"Agent: {action}"
                    if duration:
                        try:
                            ms = int(duration)
                            message += f" ({ms} ms)"
                        except Exception:
                            pass
                    await progress_tracker.update_progress(
                        research_id,
                        None,
                        None,
                        custom_data={"message": message},
                    )
        except Exception:
            pass

        # If suitable, generate an integrated synthesis (e.g., Maeve + Dolores)
        integrated_synthesis = None
        try:
            primary_ui = str(paradigm_analysis["primary"]["paradigm"])
            secondary_ui = paradigm_analysis.get("secondary", {}).get("paradigm")
            secondary_ui = str(secondary_ui) if secondary_ui is not None else None
            if primary_ui == "maeve" and secondary_ui == "dolores":
                # Extract list of results for answer generation
                flat_results = orch_resp.get("results", []) or []
                from services.answer_generator import answer_orchestrator as _ans
                # Get method references safely and ensure proper typing
                model_dump_method = getattr(ce, "model_dump", None)
                dict_method = getattr(ce, "dict", None)

                # Use callable methods or fallback to empty dict, with explicit type casting
                context_engineering: dict[str, Any] = {}
                if model_dump_method and callable(model_dump_method):
                    result = model_dump_method()
                    context_engineering = result if isinstance(result, dict) else {}
                elif dict_method and callable(dict_method):
                    result = dict_method()
                    context_engineering = result if isinstance(result, dict) else {}

                # Generate combined answers
                combo = await _ans.generate_multi_paradigm_answer(
                    primary_paradigm=primary_ui,
                    secondary_paradigm=secondary_ui,
                    query=research.query,
                    search_results=flat_results,
                    context_engineering=context_engineering,
                    options={"research_id": research_id, "max_length": SYNTHESIS_MAX_LENGTH_DEFAULT},
                )
                prim = combo.get("primary_paradigm", {}).get("answer")
                sec = combo.get("secondary_paradigm", {}).get("answer")
                # Transform to frontend-friendly minimal shape
                def _sections(ans):
                    out = []
                    for s in getattr(ans, "sections", []) or []:
                        try:
                            out.append({
                                "title": getattr(s, "title", ""),
                                "paradigm": primary_ui,
                                "content": getattr(s, "content", ""),
                                "confidence": float(getattr(s, "confidence", 0.8) or 0.8),
                                "sources_count": len(getattr(s, "citations", []) or []),
                                "citations": list(getattr(s, "citations", []) or []),
                                "key_insights": list(getattr(s, "key_insights", []) or []),
                            })
                        except Exception:
                            continue
                    return out
                prim_sections = _sections(prim) if prim else []
                sec_sections = _sections(sec) if sec else []

                # Build Immediate Opportunities (top 3 actions from primary)
                top_actions = []
                for a in (getattr(prim, "action_items", []) or [])[:3]:
                    try:
                        top_actions.append({
                            "priority": a.get("priority", "high"),
                            "action": a.get("action", ""),
                            "timeframe": a.get("timeline", ""),
                            "paradigm": primary_ui,
                        })
                    except Exception:
                        continue

                # Pick a systemic context section from secondary (fallback to its summary)
                systemic_section = None
                if sec_sections:
                    systemic = sec_sections[0]
                    systemic_section = {
                        "title": "Systemic Context (Dolores)",
                        "paradigm": secondary_ui,
                        "content": systemic.get("content", ""),
                        "confidence": systemic.get("confidence", 0.8),
                        "sources_count": systemic.get("sources_count", 0),
                        "citations": systemic.get("citations", []),
                        "key_insights": systemic.get("key_insights", []),
                    }
                else:
                    sec_summary = getattr(sec, "summary", "") if sec else ""
                    systemic_section = {
                        "title": "Systemic Context (Dolores)",
                        "paradigm": secondary_ui,
                        "content": sec_summary,
                        "confidence": 0.75,
                        "sources_count": 0,
                        "citations": [],
                        "key_insights": [],
                    }

                integrated_synthesis = {
                    "primary_answer": {
                        "summary": getattr(prim, "summary", ""),
                        "sections": prim_sections,
                        "action_items": top_actions,
                        "citations": [],
                    },
                    "secondary_perspective": systemic_section,
                    "conflicts_identified": [],
                    "synergies": [
                        "Local advantages align with policy momentum",
                        "Community trust amplifies strategic differentiation"
                    ],
                    "integrated_summary": (
                        f"Strategic plan with immediate opportunities and systemic context integrated. "
                        f"Primary: {primary_ui}, Secondary: {secondary_ui}."
                    ),
                    "confidence_score": 0.86,
                }

                # Ensure metadata includes both paradigms
                try:
                    if secondary_ui is not None:
                        paradigms_used = metadata.get("paradigms_used")
                        if isinstance(paradigms_used, list) and secondary_ui not in paradigms_used:
                            paradigms_used.append(secondary_ui)
                except Exception:
                    pass
        except Exception as e:
            # Use module-level logger; avoid overshadowing it within function scope
            logger.warning("Integrated synthesis assembly skipped: %s", e)
            _warnings.append({"code": "integrated_synthesis.failed", "message": str(e)})

        # Mesh network negotiation (optional)
        mesh_synthesis = None
        if ENABLE_MESH_NETWORK:
            try:
                if await mesh_negotiator.should_negotiate(cls):
                    raw_evidence = orch_resp.get("results", []) or []
                    # Convert SearchResult objects to dicts for mesh negotiator
                    evidence_pool = []
                    for item in raw_evidence:
                        if hasattr(item, 'model_dump'):
                            evidence_pool.append(item.model_dump())
                        elif hasattr(item, 'dict'):
                            evidence_pool.append(item.dict())
                        elif isinstance(item, dict):
                            evidence_pool.append(item)
                        else:
                            # Skip non-dict items
                            continue
                    mesh = await mesh_negotiator.negotiate(
                        classification=cls,
                        evidence_pool=evidence_pool,
                        context={"query": research.query, "research_id": research_id},
                    )
                    mesh_synthesis = {
                        "integrated": mesh.integrated_recommendation,
                        "synthesis": mesh.primary_synthesis,
                        "stances": [
                            {
                                "paradigm": (s.paradigm.value if hasattr(s.paradigm, "value") else str(s.paradigm)),
                                "perspective": s.perspective,
                                "key_points": list(s.key_points or []),
                            }
                            for s in mesh.paradigm_stances
                        ],
                        "synergies": list(mesh.synergies or []),
                        "tensions": list(mesh.tensions or []),
                    }
            except Exception as e:
                logger.warning("Mesh negotiation failed: %s", e)
                _warnings.append({"code": "mesh.negotiation_failed", "message": str(e)})

        # Attach accumulated warnings/degraded to metadata
        if _warnings:
            try:
                metadata.setdefault("warnings", []).extend(_warnings)
                metadata["degraded"] = True
            except Exception:
                pass

        final_result = {
            "research_id": research_id,
            "query": research.query,
            "status": ResearchStatus.COMPLETED,
            "paradigm_analysis": paradigm_analysis,
            "answer": answer_payload,
            "integrated_synthesis": integrated_synthesis,
            "mesh_synthesis": mesh_synthesis,
            "sources": sources_payload,
            "metadata": metadata,
            "cost_info": orch_resp.get("cost_info", {}),
            "export_formats": {
                "pdf": f"/v1/research/{research_id}/export/pdf",
                "markdown": f"/v1/research/{research_id}/export/markdown",
                "json": f"/v1/research/{research_id}/export/json",
            },
        }

        # Validate final result against API contract; log and mark degraded on failure but do not block
        try:
            validated = ResearchFinalResult(**final_result)
            final_result = validated.dict()
        except Exception as e:
            log_exception("final_result.validation_failed", e, research_id=research_id)
            try:
                metadata.setdefault("warnings", []).append({
                    "code": "contract.validation_failed",
                    "message": str(e),
                })
                metadata["degraded"] = True
                final_result["metadata"] = metadata
            except Exception:
                pass

        # Persist results and status together to avoid race conditions
        await research_store.update_fields(
            research_id,
            {"results": final_result, "status": ResearchStatus.COMPLETED},
        )

        await persist_completion(
            research_id=research_id,
            user_id=user_id,
            final_result=final_result,
        )

        if progress_tracker:
            # Broadcast completion after storage; WS will finalize progress to 100
            await progress_tracker.complete_research(research_id, {"summary": answer_payload.get("summary", "")[:200]})

        # Trigger webhook for completion (if available)
        try:
            if webhook_manager is not None:
                await webhook_manager.trigger_event(
                    WebhookEvent.RESEARCH_COMPLETED,
                    {
                        "research_id": research_id,
                        "user_id": user_id,
                        "query": research.query,
                        "paradigm": paradigm_analysis.get("primary", {}).get("paradigm", "unknown"),
                        "sources_analyzed": metadata.get("total_sources_analyzed"),
                        "processing_time_seconds": metadata.get("processing_time_seconds"),
                    },
                )
        except Exception:
            pass

    except Exception as e:
        # Log full traceback for easier debugging and include exception type
        logger.exception("execute_real_research failed for %s: %s", research_id, str(e))
        # Persist failure status and error message atomically
        await research_store.update_fields(
            research_id,
            {
                "status": ResearchStatus.FAILED,
                "error": f"{e.__class__.__name__}: {str(e)}",
            },
        )
        await persist_failure(
            research_id=research_id,
            user_id=user_id,
            error=f"{e.__class__.__name__}: {str(e)}",
        )
        if progress_tracker:
            await progress_tracker.fail_research(research_id, str(e))

        # Trigger webhook for failure
        try:
            if webhook_manager is not None:
                await webhook_manager.trigger_event(
                    WebhookEvent.RESEARCH_FAILED,
                    {
                        "research_id": research_id,
                        "user_id": user_id,
                        "query": getattr(research, "query", ""),
                        "error": str(e),
                    },
                )
        except Exception:
            pass
    finally:
        # Always decrement concurrent counter when using limiter
        if limiter is not None and limiter_identifier:
            try:
                await limiter.decrement_concurrent(limiter_identifier)
            except Exception:
                pass


@router.post("/query")
async def submit_research(
    research: ResearchQuery,
    background_tasks: BackgroundTasks,
    request: Request,
    current_user=Depends(get_current_user),
    x_experiment: str | None = Header(None, alias="X-Experiment"),
):
    """Submit a research query for paradigm-based analysis"""
    # Check role requirements for research depth
    if research.options.depth in [ResearchDepth.DEEP, ResearchDepth.DEEP_RESEARCH]:
        # Deep research requires at least PRO role
        if current_user.role not in [
            UserRole.PRO, UserRole.ENTERPRISE, UserRole.ADMIN
        ]:
            raise HTTPException(
                status_code=403,
                detail="Deep research requires PRO subscription or higher",
            )

    # Per-user API rate-limit and concurrency enforcement (U-014)
    limiter = getattr(request.app.state, "rate_limiter", None)
    identifier = f"user:{current_user.user_id}"
    if limiter is not None:
        allowed, info = await limiter.check_rate_limit(identifier, current_user.role, "api")
        if not allowed and info:
            raise RateLimitExceeded(
                retry_after=info.get("retry_after", 60),
                limit_type=info.get("limit_type", "requests_per_minute"),
                limit=info.get("limit", 0),
            )

    research_id = f"res_{uuid.uuid4().hex[:12]}"

    try:
        # Classify the query using the new classification engine
        # Honor per-request AI classification toggle (LLM on/off)
        _prev_use_llm = None
        try:
            _prev_use_llm = getattr(classification_engine.classifier, "use_llm", None)
            desired = bool(getattr(getattr(research, "options", object()), "enable_ai_classification", _prev_use_llm if _prev_use_llm is not None else True))
            if _prev_use_llm is not None:
                classification_engine.classifier.use_llm = desired  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            classification_result = await classification_engine.classify_query(
                research.query
            )
        finally:
            try:
                if _prev_use_llm is not None:
                    classification_engine.classifier.use_llm = _prev_use_llm  # type: ignore[attr-defined]
            except Exception:
                pass

        # Convert to the old format for compatibility
        classification = ParadigmClassification(
            primary=HOST_TO_MAIN_PARADIGM[classification_result.primary_paradigm],
            secondary=(
                HOST_TO_MAIN_PARADIGM.get(cast(HostParadigm, classification_result.secondary_paradigm))
                if classification_result.secondary_paradigm
                else None
            ),
            distribution={
                HOST_TO_MAIN_PARADIGM[p].value: v
                for p, v in classification_result.distribution.items()
            },
            confidence=classification_result.confidence,
            explanation={
                HOST_TO_MAIN_PARADIGM[p].value: "; ".join(r)
                for p, r in classification_result.reasoning.items()
            },
        )

        # Apply user override ONCE at submission (single source of truth)
        try:
            override_ui = getattr(getattr(research, "options", object()), "paradigm_override", None)
            if override_ui and override_ui != classification.primary:
                prev = classification.primary
                classification.primary = override_ui
                # Preserve engine suggestion as secondary if not already set
                if not classification.secondary or classification.secondary == override_ui:
                    classification.secondary = prev
                logger.info(
                    "classification.override_applied",
                    extra={
                        "research_id": research_id,
                        "user_id": str(getattr(current_user, "user_id", "unknown")),
                        "from": getattr(prev, "value", str(prev)),
                        "to": getattr(override_ui, "value", str(override_ui)),
                    },
                )
        except Exception:
            pass

        # Store research request
        # Optional experiment override via header (e.g., X-Experiment: v2)
        exp_override = None
        if x_experiment and isinstance(x_experiment, str):
            val = x_experiment.strip().lower()
            if val in {"v1", "v2"}:
                exp_override = {"prompt_variant": val, "raw": x_experiment}

        research_data = {
            "id": research_id,
            "user_id": str(current_user.user_id),
            "query": research.query,
            "options": research.options.dict(),
            "status": ResearchStatus.PROCESSING,
            "paradigm_classification": classification.dict(),
            "created_at": datetime.utcnow().isoformat(),
            "results": None,
            **({"experiment": exp_override} if exp_override else {}),
        }
        await research_store.set(research_id, research_data)

        await record_submission(
            research_id=research_id,
            user_id=str(current_user.user_id),
            query_text=research.query,
            options=research.options.dict(),
            classification=classification.dict(),
        )

        # Execute real research
        background_tasks.add_task(
            execute_real_research,
            research_id,
            research,
            str(current_user.user_id),
            current_user.role,
            limiter,
            identifier,
        )

        # Track in WebSocket (if available)
        if progress_tracker:
            await progress_tracker.start_research(
                research_id,
                str(current_user.user_id),
                research.query,
                classification.primary.value,
                research.options.depth.value,
            )

        # Trigger webhook (if available)
        if webhook_manager is not None:
            await webhook_manager.trigger_event(
                WebhookEvent.RESEARCH_STARTED,
                {
                    "research_id": research_id,
                    "user_id": str(current_user.user_id),
                    "query": research.query,
                    "paradigm": classification.primary.value,
                },
            )

        # Estimate duration based on depth
        depth_name = research.options.depth.value if hasattr(research, 'options') else 'standard'
        if depth_name in ('quick',):
            eta_minutes = 1
        elif depth_name in ('standard',):
            eta_minutes = 4
        elif depth_name in ('deep',):
            eta_minutes = 8
        else:  # deep_research
            eta_minutes = 12

        return {
            "research_id": research_id,
            "status": ResearchStatus.PROCESSING,
            "paradigm_classification": classification.dict(),
            "estimated_completion": (datetime.utcnow() + timedelta(minutes=eta_minutes)).isoformat(),
            "websocket_url": f"/ws/research/{research_id}",
        }

    except Exception as e:
        logger.error("Research submission error: %s", str(e))
        raise HTTPException(
            status_code=500, detail=f"Research submission failed: {str(e)}"
        )


@router.get("/status/{research_id}")
async def get_research_status(
    research_id: str, current_user=Depends(get_current_user)
):
    """Get the status of a research query"""
    research = await research_store.get(research_id)
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")

    # Verify ownership (use canonical user_id; fall back to id for compatibility)
    requester_id = str(getattr(current_user, "user_id", getattr(current_user, "id", None)))
    if (
        research["user_id"] != requester_id
        and current_user.role != UserRole.ADMIN
    ):
        raise HTTPException(status_code=403, detail="Access denied")

    status_response = {
        "research_id": research_id,
        "status": research["status"],
        "paradigm": research["paradigm_classification"]["primary"],
        "started_at": research["created_at"],
        "progress": research.get("progress", {}),
        "cost_info": research.get("cost_info"),
    }

    # Add status-specific information
    if research["status"] == ResearchStatus.FAILED:
        status_response["error"] = research.get("error", "Research failed")
        status_response["can_retry"] = True
        status_response["message"] = "Research failed. You can retry."
    elif research["status"] == ResearchStatus.CANCELLED:
        status_response["cancelled_at"] = research.get("cancelled_at")
        status_response["cancelled_by"] = research.get("cancelled_by")
        status_response["can_retry"] = True
        status_response["message"] = "Research was cancelled by user."
    elif research["status"] == ResearchStatus.COMPLETED:
        status_response["message"] = "Research completed successfully."
    elif research["status"] in [
        ResearchStatus.PROCESSING, ResearchStatus.IN_PROGRESS
    ]:
        status_response["can_cancel"] = True
        status_response["message"] = (
            f"Research is {research['status']}. "
            "Please wait for completion or cancel if needed."
        )

    return status_response


@router.get("/results/{research_id}")
async def get_research_results(
    research_id: str, current_user=Depends(get_current_user)
):
    """Get completed research results"""
    research = await research_store.get(research_id)
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")

    # Verify ownership (use canonical user_id; fall back to id for compatibility)
    requester_id = str(getattr(current_user, "user_id", getattr(current_user, "id", None)))
    if (
        research["user_id"] != requester_id
        and current_user.role != UserRole.ADMIN
    ):
        raise HTTPException(status_code=403, detail="Access denied")

    if research["status"] != ResearchStatus.COMPLETED:
        if research["status"] == ResearchStatus.FAILED:
            # Return detailed error information for failed research
            error_detail = research.get("error", "Research execution failed")
            return {
                "status": "failed",
                "error": error_detail,
                "research_id": research_id,
                "message": "Research failed. Please try submitting a new query.",
                "can_retry": True
            }
        elif research["status"] == ResearchStatus.CANCELLED:
            # Return cancellation information
            return {
                "status": "cancelled",
                "research_id": research_id,
                "message": "Research was cancelled by user.",
                "cancelled_at": research.get("cancelled_at"),
                "cancelled_by": research.get("cancelled_by"),
                "can_retry": True
            }
        elif research["status"] in [
            ResearchStatus.PROCESSING, ResearchStatus.IN_PROGRESS
        ]:
            # Return progress information for ongoing research
            return {
                "status": research["status"],
                "research_id": research_id,
                "message": (
                    f"Research is still {research['status']}. "
                    "Please wait for completion or cancel if needed."
                ),
                "progress": research.get("progress", {}),
                "estimated_completion": research.get("estimated_completion"),
                "can_cancel": True,
                "can_retry": False
            }
        else:
            # Handle other statuses (QUEUED, etc.)
            return {
                "status": research["status"],
                "research_id": research_id,
                "message": f"Research is {research['status']}",
                "can_retry": research["status"] != ResearchStatus.PROCESSING,
                "can_cancel": research["status"] in [ResearchStatus.QUEUED]
            }

    # Ensure results have the expected structure for frontend
    results = research["results"]

    # The frontend expects 'answer' to exist for display
    # If answer is missing or incomplete, try to reconstruct it
    if results:
        # Check if answer exists and has required fields
        answer = results.get("answer")
        if not answer or not isinstance(answer, dict):
            # Try to get it from integrated_synthesis
            integrated = results.get("integrated_synthesis")
            if integrated and isinstance(integrated, dict):
                primary = integrated.get("primary_answer")
                if primary and isinstance(primary, dict):
                    results["answer"] = primary
                    logger.info(f"Populated answer from integrated_synthesis for research_id: {research_id}")

        # Ensure answer has minimum required fields for display
        if results.get("answer") and isinstance(results["answer"], dict):
            answer = results["answer"]
            # Ensure required fields exist with defaults
            if "summary" not in answer:
                answer["summary"] = "Research completed but no summary available."
            if "sections" not in answer:
                answer["sections"] = []
            if "action_items" not in answer:
                answer["action_items"] = []
            if "citations" not in answer:
                answer["citations"] = []
            if "metadata" not in answer:
                answer["metadata"] = {}

        # Ensure status field is included for frontend consistency
        if "status" not in results:
            results["status"] = "completed"

    return results


# ---------------------------------------------------------------------------
# Deep Research Endpoints (compatibility wrapper for frontend API methods)
# ---------------------------------------------------------------------------


@router.post("/deep")
async def submit_deep_research(
    payload: ResearchDeepQuery,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
):
    """Submit a deep research request (alias for /research/query with depth=deep_research).

    Accepts a slightly different payload shape used by the frontend convenience
    method and translates it into the canonical ResearchQuery + ResearchOptions
    used by the main pipeline.
    """
    # Enforce subscription requirement (PRO+)
    if current_user.role not in [UserRole.PRO, UserRole.ENTERPRISE, UserRole.ADMIN]:
        raise HTTPException(
            status_code=403,
            detail="Deep research requires PRO subscription or higher",
        )

    # Build canonical request
    options = ResearchOptions(
        depth=ResearchDepth.DEEP_RESEARCH,
        paradigm_override=payload.paradigm,
        enable_real_search=True,
        search_context_size=payload.search_context_size,
        user_location=payload.user_location,
    )
    research = ResearchQuery(query=payload.query, options=options)

    # Generate a research id and reuse the normal submission flow parts
    research_id = f"res_{uuid.uuid4().hex[:12]}"

    try:
        # Classify query
        classification_result = await classification_engine.classify_query(
            research.query
        )

        classification = ParadigmClassification(
            primary=HOST_TO_MAIN_PARADIGM[classification_result.primary_paradigm],
            secondary=(
                HOST_TO_MAIN_PARADIGM.get(cast(HostParadigm, classification_result.secondary_paradigm))
                if classification_result.secondary_paradigm
                else None
            ),
            distribution={
                HOST_TO_MAIN_PARADIGM[p].value: v
                for p, v in classification_result.distribution.items()
            },
            confidence=classification_result.confidence,
            explanation={
                HOST_TO_MAIN_PARADIGM[p].value: "; ".join(r)
                for p, r in classification_result.reasoning.items()
            },
        )

        # Apply user override ONCE at submission
        try:
            override_ui = research.options.paradigm_override
            if override_ui and override_ui != classification.primary:
                prev = classification.primary
                classification.primary = override_ui
                if not classification.secondary or classification.secondary == override_ui:
                    classification.secondary = prev
                logger.info(
                    "classification.override_applied",
                    extra={
                        "research_id": research_id,
                        "user_id": str(getattr(current_user, "user_id", "unknown")),
                        "from": getattr(prev, "value", str(prev)),
                        "to": getattr(override_ui, "value", str(override_ui)),
                    },
                )
        except Exception:
            pass

        # Store request
        research_data = {
            "id": research_id,
            "user_id": str(current_user.user_id),
            "query": research.query,
            "options": research.options.dict(),
            "status": ResearchStatus.PROCESSING,
            "paradigm_classification": classification.dict(),
            "created_at": datetime.utcnow().isoformat(),
            "results": None,
        }
        await research_store.set(research_id, research_data)

        # Execute in background
        background_tasks.add_task(
            execute_real_research,
            research_id,
            research,
            str(current_user.user_id),
            current_user.role,
        )

        # Start WS tracking
        if progress_tracker:
            await progress_tracker.start_research(
                research_id,
                str(current_user.user_id),
                research.query,
                classification.primary.value,
                ResearchDepth.DEEP_RESEARCH.value,
            )

        # Webhook
        if webhook_manager is not None:
            await webhook_manager.trigger_event(
                WebhookEvent.RESEARCH_STARTED,
                {
                    "research_id": research_id,
                    "user_id": str(current_user.user_id),
                    "query": research.query,
                    "paradigm": classification.primary.value,
                },
            )

        eta_minutes = 12
        return {
            "research_id": research_id,
            "status": ResearchStatus.PROCESSING,
            "paradigm_classification": classification.dict(),
            "estimated_completion": (datetime.utcnow() + timedelta(minutes=eta_minutes)).isoformat(),
            "websocket_url": f"/ws/research/{research_id}",
        }
    except Exception as e:
        logger.error("Deep research submission error: %s", str(e))
        raise HTTPException(
            status_code=500, detail=f"Deep research submission failed: {str(e)}"
        )


@router.get("/deep/status")
async def get_deep_research_status(current_user=Depends(get_current_user)):
    """Return a lightweight status summary for the user's deep research jobs."""
    try:
        records = await research_store.get_user_research(str(current_user.user_id), limit=200)
        deep_records = [
            r for r in records
            if (r.get("options") or {}).get("depth") in (ResearchDepth.DEEP_RESEARCH, ResearchDepth.DEEP)
        ]

        active = [
            r for r in deep_records
            if r.get("status") in (ResearchStatus.PROCESSING, ResearchStatus.IN_PROGRESS)
        ]
        recent = sorted(deep_records, key=lambda x: x.get("created_at", ""), reverse=True)[:5]
        return {
            "total_deep_jobs": len(deep_records),
            "active_deep_jobs": len(active),
            "recent": [
                {
                    "research_id": r.get("id"),
                    "status": r.get("status"),
                    "query": r.get("query"),
                    "created_at": r.get("created_at"),
                }
                for r in recent
            ],
        }
    except Exception as e:
        logger.error("Failed to get deep research status: %s", e)
        return {"total_deep_jobs": 0, "active_deep_jobs": 0, "recent": []}


@router.post("/deep/{research_id}/resume")
async def resume_deep_research(
    research_id: str,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
):
    """Resume a deep research job by re-running the pipeline with the stored query/options.

    This uses the original research_id so any open WebSocket client can continue
    receiving progress events without re-subscribing.
    """
    research = await research_store.get(research_id)
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")

    # Verify ownership
    if (
        research.get("user_id") != str(current_user.user_id)
        and current_user.role != UserRole.ADMIN
    ):
        raise HTTPException(status_code=403, detail="Access denied")

    # Completed jobs aren't resumable
    if research.get("status") == ResearchStatus.COMPLETED:
        return {
            "research_id": research_id,
            "status": ResearchStatus.COMPLETED,
            "message": "Research is already completed",
            "can_retry": True,
        }

    # Reconstruct canonical request
    try:
        q = research.get("query") or ""
        opts_dict = research.get("options") or {}
        # Force deep_research depth for this endpoint
        opts_dict["depth"] = ResearchDepth.DEEP_RESEARCH
        options = ResearchOptions.parse_obj(opts_dict)
        rq = ResearchQuery(query=q, options=options)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid stored request: {e}")

    # Attempt to resume an existing background LLM task first (U-007 resumability)
    try:
        bg_task_id = research.get("llm_bg_task_id")
        if bg_task_id and background_llm_manager is not None:
            resumed = await background_llm_manager.resume_background_task(bg_task_id, research_id)
            if resumed:
                if progress_tracker:
                    try:
                        await progress_tracker.update_progress(
                            research_id,
                            phase="synthesis",
                            message="Resumed background LLM task",
                            custom_data={"bg_task_id": bg_task_id, "resumed": True},
                        )
                    except Exception:
                        pass
                return {
                    "research_id": research_id,
                    "status": ResearchStatus.PROCESSING,
                    "message": "Resumed ongoing background LLM task",
                    "websocket_url": f"/ws/research/{research_id}",
                }
    except Exception:
        # Fall back to full rerun if resume isn't possible
        pass

    # Update state and launch background execution
    await research_store.update_field(research_id, "status", ResearchStatus.PROCESSING)
    await research_store.update_field(research_id, "estimated_completion", (datetime.utcnow() + timedelta(minutes=12)).isoformat())

    # Re-classify for transparency and attach to record
    try:
        classification_result = await classification_engine.classify_query(rq.query)
        classification = ParadigmClassification(
            primary=HOST_TO_MAIN_PARADIGM[classification_result.primary_paradigm],
            secondary=(
                HOST_TO_MAIN_PARADIGM.get(cast(HostParadigm, classification_result.secondary_paradigm))
                if classification_result.secondary_paradigm
                else None
            ),
            distribution={
                HOST_TO_MAIN_PARADIGM[p].value: v
                for p, v in classification_result.distribution.items()
            },
            confidence=classification_result.confidence,
            explanation={
                HOST_TO_MAIN_PARADIGM[p].value: "; ".join(r)
                for p, r in classification_result.reasoning.items()
            },
        )
        # Respect stored override once when resuming
        try:
            override_ui = getattr(getattr(rq, "options", object()), "paradigm_override", None)
            if override_ui and override_ui != classification.primary:
                prev = classification.primary
                classification.primary = override_ui
                if not classification.secondary or classification.secondary == override_ui:
                    classification.secondary = prev
                logger.info(
                    "classification.override_applied",
                    extra={
                        "research_id": research_id,
                        "user_id": str(getattr(current_user, "user_id", "unknown")),
                        "from": getattr(prev, "value", str(prev)),
                        "to": getattr(override_ui, "value", str(override_ui)),
                    },
                )
        except Exception:
            pass
        await research_store.update_field(research_id, "paradigm_classification", classification.dict())
    except Exception:
        classification = None

    background_tasks.add_task(
        execute_real_research,
        research_id,
        rq,
        str(current_user.user_id),
        current_user.role,
    )

    # WebSocket event
    if progress_tracker:
        try:
            await progress_tracker.start_research(
                research_id,
                str(current_user.user_id),
                rq.query,
                (classification.primary.value if classification else research.get("paradigm_classification", {}).get("primary", "unknown")),
                ResearchDepth.DEEP_RESEARCH.value,
            )
        except Exception:
            pass

    return {
        "research_id": research_id,
        "status": ResearchStatus.PROCESSING,
        "paradigm_classification": (classification.dict() if classification else research.get("paradigm_classification")),
        "estimated_completion": (datetime.utcnow() + timedelta(minutes=12)).isoformat(),
        "websocket_url": f"/ws/research/{research_id}",
    }


@router.post("/cancel/{research_id}")
async def cancel_research(
    research_id: str, current_user=Depends(get_current_user)
):
    """Cancel an ongoing research query"""
    research = await research_store.get(research_id)
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")

    # Verify ownership (use canonical user_id; fall back to id for compatibility)
    requester_id = str(getattr(current_user, "user_id", getattr(current_user, "id", None)))
    if (
        research["user_id"] != requester_id
        and current_user.role != UserRole.ADMIN
    ):
        raise HTTPException(status_code=403, detail="Access denied")

    # Check if research can be cancelled
    current_status = research["status"]
    if current_status in [
        ResearchStatus.COMPLETED,
        ResearchStatus.FAILED,
        ResearchStatus.CANCELLED
    ]:
        return {
            "research_id": research_id,
            "status": current_status,
            "message": (
                f"Research is already {current_status} "
                "and cannot be cancelled"
            ),
            "cancelled": False
        }

    try:
        # Update status to cancelled
        await research_store.update_field(
            research_id, "status", ResearchStatus.CANCELLED
        )
        await research_store.update_field(
            research_id, "cancelled_at", datetime.utcnow().isoformat()
        )
        await research_store.update_field(
            research_id, "cancelled_by", str(current_user.user_id)
        )

        logger.info(
            "Research %s cancelled by user %s",
            research_id, current_user.user_id
        )

        # Broadcast cancellation over WebSocket for live UIs
        if progress_tracker:
            try:
                await progress_tracker.connection_manager.broadcast_to_research(
                    research_id,
                    WSMessage(
                        type=WSEventType.RESEARCH_CANCELLED,
                        data={
                            "research_id": research_id,
                            "status": "cancelled",
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    ),
                )
            except Exception:
                pass

        # Trigger webhook for cancellation
        try:
            if webhook_manager is not None:
                await webhook_manager.trigger_event(
                    WebhookEvent.RESEARCH_CANCELLED,
                    {
                        "research_id": research_id,
                        "user_id": str(current_user.user_id),
                        "cancelled_at": datetime.utcnow().isoformat(),
                    },
                )
        except Exception:
            pass

        return {
            "research_id": research_id,
            "status": "cancelled",
            "message": "Research has been successfully cancelled",
            "cancelled": True,
            "cancelled_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error("Failed to cancel research %s: %s", research_id, e)
        raise HTTPException(status_code=500, detail="Failed to cancel research")


@router.get("/history")
async def get_research_history(
    current_user=Depends(get_current_user),
    limit: int = 10,
    offset: int = 0
):
    """Get user's research history"""
    try:
        # Get research history for user
        user_research = await research_store.get_user_research(
            str(current_user.user_id), limit + offset
        )

        # Sort by creation date (newest first)
        user_research.sort(key=lambda x: x["created_at"], reverse=True)

        # Apply pagination
        total = len(user_research)
        paginated = user_research[offset: offset + limit]

        # Format the response
        history = []
        for research in paginated:
            history_item = {
                "research_id": research["id"],
                "query": research["query"],
                "status": research["status"],
                "paradigm": research.get("paradigm_classification", {}).get(
                    "primary", "unknown"
                ),
                "created_at": research["created_at"],
                "options": research["options"],
            }

            # Include results summary if completed
            if (research["status"] == ResearchStatus.COMPLETED and
                research.get("results")):
                results = research["results"]
                content_preview = ""
                if results.get("answer", {}).get("sections"):
                    content_preview = (
                        results["answer"]["sections"][0].get("content", "")[:200]
                        + "..."
                    )
                history_item["summary"] = {
                    "answer_preview": content_preview,
                    "source_count": len(results.get("sources", [])),
                    "total_cost": results.get("cost_info", {}).get("total_cost", 0),
                }

            history.append(history_item)

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "history": history
        }
    except Exception as e:
        logger.error("Error fetching research history: %s", str(e))
        raise HTTPException(
            status_code=500, detail="Failed to fetch research history"
        )


@router.post("/feedback/{research_id}")
async def submit_research_feedback(
    research_id: str,
    satisfaction_score: float,
    paradigm_feedback: Optional[str] = None,
    current_user=Depends(get_current_user),
):
    """Submit feedback for a research query to improve the system"""
    research = await research_store.get(research_id)
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")

    # Verify ownership
    if (research["user_id"] != str(current_user.id) and
        current_user.role != UserRole.ADMIN):
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        # Store feedback in research data
        await research_store.update_field(research_id, "user_feedback", {
            "satisfaction_score": satisfaction_score,
            "paradigm_feedback": paradigm_feedback,
            "submitted_at": datetime.utcnow().isoformat()
        })

        return {
            "success": True,
            "message": "Feedback recorded successfully",
            "research_id": research_id,
            "satisfaction_score": satisfaction_score
        }
    except Exception as e:
        logger.error("Failed to record feedback: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to record feedback")
