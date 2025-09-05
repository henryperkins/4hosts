"""
Paradigm-related routes for the Four Hosts Research API
"""

import logging
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
from services.research_store import research_store
from services.enhanced_integration import (
    enhanced_classification_engine as classification_engine
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/paradigms", tags=["paradigms"])


def get_paradigm_approach_suggestion(paradigm: Paradigm) -> str:
    """Get approach suggestion for a paradigm"""
    suggestions = {
        Paradigm.DOLORES: "Focus on exposing systemic issues and empowering resistance",
        Paradigm.TEDDY: "Prioritize community support and protective measures",
        Paradigm.BERNARD: "Emphasize empirical research and data-driven analysis",
        Paradigm.MAEVE: "Develop strategic frameworks and actionable plans",
    }
    return suggestions[paradigm]


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

    # Update paradigm and reset status
    await research_store.update_field(
        payload.research_id, "override_paradigm", payload.paradigm.value
    )
    await research_store.update_field(
        payload.research_id, "status", "processing"
    )

    # Note: Re-queuing logic would go here in the actual implementation
    # For now, just log the override
    logger.info(
        "Paradigm override requested for research %s: %s -> %s",
        payload.research_id,
        research.get("paradigm_classification", {}).get("primary", "unknown"),
        payload.paradigm.value
    )

    return {
        "success": True,
        "research_id": payload.research_id,
        "new_paradigm": payload.paradigm.value,
        "status": "re-processing",
    }


@router.get("/explanation/{paradigm}")
async def get_paradigm_explanation(paradigm: Paradigm):
    """Return a detailed explanation of the selected paradigm"""
    explanation = PARADIGM_EXPLANATIONS.get(paradigm.value)
    if not explanation:
        raise HTTPException(status_code=404, detail="Paradigm not found")
    return explanation
