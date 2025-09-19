"""
Paradigm-related routes for the Four Hosts Research API
"""

import logging
import structlog
from typing import Dict, Any

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
from core.dependencies import get_current_user
from core.config import PARADIGM_EXPLANATIONS
from models.paradigms import normalize_to_internal_code
from services.research_store import research_store
from services.enhanced_integration import (
    enhanced_classification_engine as classification_engine
)

logger = structlog.get_logger(__name__)

# Create router
router = APIRouter(prefix="/paradigms", tags=["paradigms"])


def get_paradigm_approach_suggestion(paradigm: Paradigm) -> str:
    """Get approach suggestion for a paradigm"""
    try:
        key = normalize_to_internal_code(paradigm.value)
    except Exception:
        key = str(paradigm.value)
    cfg = PARADIGM_EXPLANATIONS.get(key) or {}
    return cfg.get("approach_suggestion") or {
        Paradigm.DOLORES: "Focus on exposing systemic issues and empowering resistance",
        Paradigm.TEDDY: "Prioritize community support and protective measures",
        Paradigm.BERNARD: "Emphasize empirical research and data-driven analysis",
        Paradigm.MAEVE: "Develop strategic frameworks and actionable plans",
    }[paradigm]


@router.post("/classify")
async def classify_paradigm(
    payload: ClassifyRequest,
    current_user=Depends(get_current_user)
):
    """Classify a query into paradigms"""
    try:
        query = payload.query

        # Use the new classification engine
        classification_result = await classification_engine.classify_query(query)

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
                ui_key = HOST_TO_MAIN_PARADIGM.get(p).value if hasattr(p, 'name') else str(p)
                if ui_key:
                    # keep only compact fields expected by UI
                    signals_out[ui_key] = {
                        "keywords": list(details.get("keywords", []) or [])[:5],
                        "intent_signals": list(details.get("intent_signals", []) or [])[:5],
                    }
        except Exception:
            signals_out = {}

        return {
            "query": query,
            "classification": classification.dict(),
            "signals": signals_out,
            "suggested_approach": get_paradigm_approach_suggestion(
                classification.primary
            ),
            "user_id": str(current_user.user_id),
        }
    except Exception as e:
        logger.error("Classification error: %s", str(e))
        raise HTTPException(
            status_code=500, detail=f"Classification failed: {str(e)}"
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
        # Persist override for traceability
        await research_store.update_field(
            payload.research_id, "override_paradigm", payload.paradigm.value
        )
        if payload.reason:
            await research_store.update_field(
                payload.research_id, "override_reason", payload.reason
            )

        # Update nested options.paradigm_override for the executor path
        try:
            options = dict(research.get("options") or {})
            options["paradigm_override"] = payload.paradigm.value
            await research_store.update_field(payload.research_id, "options", options)
        except Exception:
            # Non-fatal; executor also reads top-level override
            pass

        # Reset execution state for re-processing
        await research_store.update_field(payload.research_id, "status", "processing")
        await research_store.update_field(payload.research_id, "results", None)
        await research_store.update_field(payload.research_id, "error", None)
        await research_store.update_field(payload.research_id, "requeued_at", __import__("datetime").datetime.utcnow().isoformat())
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
        payload.paradigm.value,
    )

    return {
        "success": True,
        "research_id": payload.research_id,
        "new_paradigm": payload.paradigm.value,
        "status": "in_progress",
    }


@router.get("/explanation/{paradigm}")
async def get_paradigm_explanation(paradigm: Paradigm):
    """Return a detailed explanation of the selected paradigm"""
    explanation = PARADIGM_EXPLANATIONS.get(paradigm.value)
    if not explanation:
        raise HTTPException(status_code=404, detail="Paradigm not found")
    return explanation
