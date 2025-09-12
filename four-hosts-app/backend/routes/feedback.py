"""
Feedback routes for Four Hosts backend.

Exposes two endpoints:

    POST /v1/feedback/classification
    POST /v1/feedback/answer

The file is imported by core.app.  If import fails the router is skipped,
so keeping this module lightweight ensures the API is available when the
backend boots even if optional deps (Redis, Postgres) are not configured.

Data is persisted in the ResearchStore when `research_id` is provided,
otherwise the event is logged only (e.g. classification feedback collected
before the user submits the full research request).

Events are shaped using models.feedback.* schemas.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

# Pydantic request / event models
from models.feedback import (
    ClassificationFeedbackRequest,
    AnswerFeedbackRequest,
    FeedbackEvent,
)

# Auth dependency
from backend.core.dependencies import get_current_user
# Research store (Redis + in-mem fallback)
from backend.services.research_store import research_store
# Enhanced integration helpers – self-healing & ML pipeline hooks
from backend.services.enhanced_integration import record_user_feedback

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/feedback", tags=["feedback"])

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
EVENT_CLASSIFICATION = "classification"
EVENT_ANSWER = "answer"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
async def _persist_feedback_event(
    *,
    event_type: str,
    payload: Dict[str, Any],
    user_id: str,
    research_id: str | None = None,
) -> None:
    """
    Persist a feedback event in the research store.

    1. If `research_id` is supplied the event is appended to that research
       document under the `feedback_events` array.
    2. Otherwise the event is stored under a user-specific key
       `feedback:{user_id}` to be reconciled later.

    Any exception is logged and re-raised to surface unexpected errors during
    development while preventing silent data loss.
    """
    evt: Dict[str, Any] = FeedbackEvent(
        id=str(uuid.uuid4()),
        user_id=user_id,
        type=event_type,
        payload=payload,
        timestamp=datetime.utcnow(),
    ).dict()

    try:
        if research_id:
            # Attach to an existing research run
            rec = await research_store.get(research_id) or {}
            rec.setdefault("feedback_events", []).append(evt)
            await research_store.update_field(
                research_id, "feedback_events", rec["feedback_events"]
            )
        else:
            # Fallback bucket per user
            key = f"feedback:{user_id}"
            blob = await research_store.get(key) or {"events": []}
            blob["events"].append(evt)
            await research_store.set(key, blob)
    except Exception as exc:  # pragma: no cover
        logger.error("Persisting feedback failed: %s", exc, exc_info=True)
        raise


# --------------------------------------------------------------------------- #
# Routes – Classification feedback
# --------------------------------------------------------------------------- #
@router.post("/classification", status_code=201)
async def submit_classification_feedback(
    feedback: ClassificationFeedbackRequest,
    current_user=Depends(get_current_user),
):
    """
    Capture user feedback on paradigm classification.

    Body: models.feedback.ClassificationFeedbackRequest
    """
    # Sanity-check optional correction
    if feedback.user_correction:
        correction = feedback.user_correction.strip().lower()
        if correction not in {"dolores", "teddy", "bernard", "maeve"}:
            raise HTTPException(status_code=400, detail="Invalid paradigm correction")

    await _persist_feedback_event(
        event_type=EVENT_CLASSIFICATION,
        payload=feedback.dict(),
        user_id=str(current_user.user_id),
        research_id=feedback.research_id,
    )

    return JSONResponse({"success": True}, status_code=201)


# --------------------------------------------------------------------------- #
# Routes – Answer feedback
# --------------------------------------------------------------------------- #
@router.post("/answer", status_code=201)
async def submit_answer_feedback(
    feedback: AnswerFeedbackRequest,
    current_user=Depends(get_current_user),
):
    """
    Capture satisfaction ratings & free-text tips for generated answers.

    Body: models.feedback.AnswerFeedbackRequest
    """
    # Persist first to avoid losing the event if downstream hooks fail
    await _persist_feedback_event(
        event_type=EVENT_ANSWER,
        payload=feedback.dict(),
        user_id=str(current_user.user_id),
        research_id=feedback.research_id,
    )

    # Best-effort self-healing / ML pipeline hook
    try:
        await record_user_feedback(
            query_id=feedback.research_id,
            satisfaction_score=float(feedback.rating),
            paradigm_feedback=None,
        )
    except Exception as exc:  # pragma: no cover
        logger.debug("record_user_feedback failed (non-fatal): %s", exc)

    return JSONResponse(
        {"success": True, "normalized_rating": float(feedback.rating)}, status_code=201
    )
