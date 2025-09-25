"""
Paradigm-related routes for the Four Hosts Research API
"""

import structlog
from typing import Dict, Any, Optional, Union

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends

from models.research import (
    ClassifyRequest,
    ParadigmOverrideRequest
)
from models.base import (
    Paradigm,
    UserRole,
    HOST_TO_MAIN_PARADIGM,
    ParadigmClassification
)
from core.dependencies import get_current_user, get_current_user_optional
from core.config import PARADIGM_EXPLANATIONS
from models.paradigms import normalize_to_internal_code
from services.research_store import research_store
from services.enhanced_integration import (
    enhanced_classification_engine as classification_engine
)

from utils.date_utils import get_current_utc
logger = structlog.get_logger(__name__)

# Create router
router = APIRouter(prefix="/paradigms", tags=["paradigms"])


async def _fallback_classification(query: str):
    """Fallback classification when the main engine fails"""
    from services.classification_engine import ClassificationResult, HostParadigm, QueryFeatures

    logger.info("Using fallback classification", query=query[:50])

    # Simple keyword-based fallback
    query_lower = query.lower()

    # Basic keyword mapping for fallback
    if any(word in query_lower for word in ['revolution', 'protest', 'activism', 'resistance', 'oppression']):
        primary = HostParadigm.DOLORES
    elif any(word in query_lower for word in ['community', 'support', 'help', 'care', 'empathy']):
        primary = HostParadigm.TEDDY
    elif any(word in query_lower for word in ['research', 'data', 'analysis', 'study', 'evidence']):
        primary = HostParadigm.BERNARD
    else:
        primary = HostParadigm.MAEVE  # Default to strategic

    # Create minimal features for fallback
    features = QueryFeatures(
        text=query,
        tokens=query_lower.split(),
        entities=[],
        intent_signals=[],
        domain=None,
        urgency_score=0.0,
        complexity_score=0.0,
        emotional_valence=0.0
    )

    return ClassificationResult(
        query=query,
        primary_paradigm=primary,
        secondary_paradigm=None,
        distribution={primary: 1.0},
        confidence=0.1,  # Low confidence for fallback
        features=features,
        reasoning={primary: ["Fallback classification - limited analysis"]},
        signals={}
    )


def get_paradigm_approach_suggestion(paradigm: Union[Paradigm, str]) -> str:
    """Get approach suggestion for a paradigm (robust to str or Enum)."""
    raw = getattr(paradigm, "value", paradigm)
    try:
        key = normalize_to_internal_code(str(raw))
    except Exception:
        key = str(raw).strip().lower()
    cfg = PARADIGM_EXPLANATIONS.get(key) or {}
    defaults_by_code = {
        "dolores": "Focus on exposing systemic issues and empowering resistance",
        "teddy": "Prioritize community support and protective measures",
        "bernard": "Emphasize empirical research and data-driven analysis",
        "maeve": "Develop strategic frameworks and actionable plans",
    }
    return cfg.get("approach_suggestion") or defaults_by_code.get(key, defaults_by_code["bernard"])


@router.post("/classify")
async def classify_paradigm(
    payload: ClassifyRequest,
    current_user=Depends(get_current_user_optional)
):
    """Classify a query into paradigms with enhanced error handling and fallback"""
    import asyncio

    try:
        query = payload.query
        uid = str(getattr(current_user, "user_id", "anonymous"))
        logger.info("Starting classification", query=query[:50], user_id=uid)

        # Add timeout protection for the classification engine
        try:
            classification_result = await asyncio.wait_for(
                classification_engine.classify_query(query),
                timeout=30.0  # 30 second timeout
            )
            logger.info("Classification completed successfully",
                       confidence=classification_result.confidence,
                       primary_paradigm=classification_result.primary_paradigm.value)
        except asyncio.TimeoutError:
            logger.error("Classification engine timeout", query=query[:50])
            # Fallback to simple classification
            classification_result = await _fallback_classification(query)
        except Exception as engine_error:
            logger.error("Classification engine failed",
                        error=str(engine_error),
                        error_type=type(engine_error).__name__)
            # Use fallback instead of failing completely
            classification_result = await _fallback_classification(query)

        # Convert to the old format for compatibility
        classification = ParadigmClassification(
            primary=HOST_TO_MAIN_PARADIGM[classification_result.primary_paradigm],
            secondary=(
                HOST_TO_MAIN_PARADIGM.get(classification_result.secondary_paradigm)
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

        # Normalize structured signals (keywords, intents) to frontend paradigm codes
        signals_out: Dict[str, Dict[str, Any]] = {}
        try:
            # Map internal HostParadigm (revolutionary/devotion/analytical/strategic) to UI names
            for p, details in (classification_result.signals or {}).items():
                ui_key = None
                if hasattr(p, "name"):
                    mapped = HOST_TO_MAIN_PARADIGM.get(p)
                    ui_key = mapped.value if mapped else None
                else:
                    ui_key = str(p)
                if ui_key:
                    # keep only compact fields expected by UI
                    signals_out[ui_key] = {
                        "keywords": list(details.get("keywords", []) or [])[:5],
                        "intent_signals": list(details.get("intent_signals", []) or [])[:5],
                    }
        except Exception as signals_error:
            logger.warning("Failed to process signals", error=str(signals_error))
            signals_out = {}

        response: Dict[str, Any] = {
            "query": query,
            "classification": classification.dict(),
            "signals": signals_out,
            "suggested_approach": get_paradigm_approach_suggestion(
                classification.primary
            ),
            "user_id": uid,
        }

        # Add fallback indicator if confidence is very low (indicating fallback)
        if classification_result.confidence < 0.2:
            response["fallback_used"] = True

        _pv = classification.primary.value if hasattr(classification.primary, "value") else str(classification.primary)
        logger.info("Classification response prepared",
                   paradigm=_pv,
                   confidence=classification.confidence,
                   fallback_used=classification_result.confidence < 0.2)

        return response

    except Exception as e:
        logger.error("Classification error",
                    error=str(e),
                    error_type=type(e).__name__,
                    user_id=str(getattr(current_user, "user_id", "anonymous")),
                    query=payload.query[:50] if hasattr(payload, 'query') else 'unknown')

        # Return structured error instead of generic 500
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Classification failed",
                "type": type(e).__name__,
                "message": str(e),
                "user_id": str(current_user.user_id),
                "timestamp": get_current_utc().isoformat()
            }
        ) from e


@router.post("/override")
async def override_paradigm(
    payload: ParadigmOverrideRequest,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
):
    """
    Force a specific paradigm for an existing research request
    and restart processing.
    """
    research = await research_store.get(payload.research_id)
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")

    # Ownership / permission check
    if ((research["user_id"] != str(current_user.user_id)) and
        current_user.role != UserRole.ADMIN):
        raise HTTPException(status_code=403, detail="Access denied")

    # Update stored record: set override + reset status and results
    try:
        # Persist override for traceability (allow clearing via explicit null)
        override_value: Optional[str] = (
            payload.paradigm.value if payload.paradigm else None
        )

        await research_store.update_field(
            payload.research_id, "override_paradigm", override_value
        )
        if payload.reason is not None:
            reason_value = payload.reason.strip() or None
            await research_store.update_field(
                payload.research_id, "override_reason", reason_value
            )
        elif not payload.paradigm:
            # Clearing the override should also remove stale reason text
            await research_store.update_field(
                payload.research_id, "override_reason", None
            )

        # Update nested options.paradigm_override for the executor path
        try:
            options = dict(research.get("options") or {})
            options["paradigm_override"] = override_value
            await research_store.update_field(payload.research_id, "options", options)
        except Exception:
            # Non-fatal; executor also reads top-level override
            pass

        # Reset execution state for re-processing
        await research_store.update_field(payload.research_id, "status", "processing")
        await research_store.update_field(payload.research_id, "results", None)
        await research_store.update_field(payload.research_id, "error", None)
        await research_store.update_field(payload.research_id, "requeued_at", get_current_utc().isoformat())
    except Exception as e:
        logger.error("Failed to update research for override: %s", e)
        raise HTTPException(status_code=500, detail="Failed to persist override")

    # Re-queue the job to run with the override applied
    try:
        from models.research import ResearchQuery, ResearchOptions
        from routes.research import execute_real_research

        # Reconstruct canonical ResearchQuery using stored options + new override
        opts = ResearchOptions(**(research.get("options") or {}))
        opts.paradigm_override = payload.paradigm
        rq = ResearchQuery(query=research.get("query", ""), options=opts)

        # Use the original job's user_id to attribute execution; role uses caller's role
        background_tasks.add_task(
            execute_real_research,
            payload.research_id,
            rq,
            research.get("user_id", str(current_user.user_id)),
            current_user.role,
        )
    except Exception as e:
        logger.error("Failed to enqueue override reprocessing: %s", e)
        raise HTTPException(status_code=500, detail="Failed to re-queue research")

    logger.info(
        "Paradigm override enqueued for research %s: -> %s",
        payload.research_id,
        override_value or "cleared",
    )

    return {
        "success": True,
        "research_id": payload.research_id,
        "new_paradigm": override_value,
        "override_cleared": override_value is None,
        "status": "in_progress",
    }


@router.get("/explanation/{paradigm}")
async def get_paradigm_explanation(paradigm: Paradigm):
    """Return a detailed explanation of the selected paradigm"""
    explanation = PARADIGM_EXPLANATIONS.get(paradigm.value)
    if not explanation:
        raise HTTPException(status_code=404, detail="Paradigm not found")
    return explanation
