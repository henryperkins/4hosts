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

from fastapi import APIRouter, Depends, HTTPException, Request
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
# Feature flags and rate limiting
from backend.core.config import ENABLE_FEEDBACK_RATE_LIMIT
from backend.services.rate_limiter import RateLimitExceeded

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
            # Attach to an existing research run (create record if missing)
            rec = await research_store.get(research_id)
            if not rec:
                rec = {"feedback_events": []}
                # Create a new record so subsequent update_field works reliably
                await research_store.set(research_id, rec)
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
    request: Request,
    current_user=Depends(get_current_user),
):
    """
    Capture user feedback on paradigm classification.

    Body: models.feedback.ClassificationFeedbackRequest
    """
    # Optional per-user rate limiting
    if ENABLE_FEEDBACK_RATE_LIMIT:
        try:
            limiter = getattr(request.app.state, "rate_limiter", None)
            if limiter is not None:
                allowed, info = await limiter.check_rate_limit(
                    f"user:{current_user.user_id}", current_user.role, "api"
                )
                if not allowed and info:
                    raise RateLimitExceeded(
                        retry_after=info.get("retry_after", 60),
                        limit_type=info.get("limit_type", "requests_per_minute"),
                        limit=info.get("limit", 0),
                    )
        except Exception as rl_exc:
            # Non-fatal: log and continue to preserve feedback path robustness
            logger.debug("Rate limit check error (non-fatal): %s", rl_exc)
    try:
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

        # Metrics: count classification feedback submissions
        try:  # best-effort
            from backend.services.metrics import metrics  # type: ignore
            metrics.increment("feedback_classification_submitted")
        except Exception:
            pass

        return JSONResponse({"success": True}, status_code=201)
    except HTTPException:
        # Already contains a clear detail; re-raise
        logger.warning("Classification feedback rejected: %s", feedback.dict())
        raise
    except Exception as exc:
        # Provide a structured 400 for easier debugging in tests/dev
        logger.error("Classification feedback failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=400, detail=f"Classification feedback error: {exc}")


# --------------------------------------------------------------------------- #
# Routes – Answer feedback
# --------------------------------------------------------------------------- #
@router.post("/answer", status_code=201)
async def submit_answer_feedback(
    feedback: AnswerFeedbackRequest,
    request: Request,
    current_user=Depends(get_current_user),
):
    """
    Capture satisfaction ratings & free-text tips for generated answers.

    Body: models.feedback.AnswerFeedbackRequest
    """
    # Optional per-user rate limiting
    if ENABLE_FEEDBACK_RATE_LIMIT:
        try:
            limiter = getattr(request.app.state, "rate_limiter", None)
            if limiter is not None:
                allowed, info = await limiter.check_rate_limit(
                    f"user:{current_user.user_id}", current_user.role, "api"
                )
                if not allowed and info:
                    raise RateLimitExceeded(
                        retry_after=info.get("retry_after", 60),
                        limit_type=info.get("limit_type", "requests_per_minute"),
                        limit=info.get("limit", 0),
                    )
        except Exception as rl_exc:
            # Non-fatal: log and continue to preserve feedback path robustness
            logger.debug("Rate limit check error (non-fatal): %s", rl_exc)
    try:
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

        # Metrics: count answer feedback and bucketed rating/helpful
        try:  # best-effort
            from backend.services.metrics import metrics  # type: ignore
            metrics.increment("feedback_answer_submitted")
            # bucketed rating (0-1 normalized) to 5-star bins
            try:
                r = float(feedback.rating)
            except Exception:
                r = 0.0
            stars = 1 + int(round((max(0.0, min(1.0, r)) * 4)))
            metrics.increment("feedback_answer_rating", str(stars))
            if feedback.helpful is True:
                metrics.increment("feedback_answer_helpful", "yes")
            elif feedback.helpful is False:
                metrics.increment("feedback_answer_helpful", "no")
            else:
                metrics.increment("feedback_answer_helpful", "unset")
        except Exception:
            pass

        return JSONResponse(
            {"success": True, "normalized_rating": float(feedback.rating)}, status_code=201
        )
    except HTTPException:
        logger.warning("Answer feedback rejected: %s", feedback.dict())
        raise
    except Exception as exc:
        logger.error("Answer feedback failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=400, detail=f"Answer feedback error: {exc}")
