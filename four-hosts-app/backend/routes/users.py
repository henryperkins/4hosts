"""
SSOTA User routes: preferences and history
"""

from typing import Any, Dict
import logging

from fastapi import APIRouter, HTTPException, Depends
import uuid

from core.dependencies import get_current_user
from services.user_management import user_profile_service
from services.research_store import research_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["users"])


@router.post("/preferences")
async def set_preferences(payload: Dict[str, Any], current_user=Depends(get_current_user)):
    try:
        prefs = (payload or {}).get("preferences") or {}
        ok = await user_profile_service.update_user_preferences(uuid.UUID(str(current_user.user_id)), prefs)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to update preferences")
        profile = await user_profile_service.get_user_profile(uuid.UUID(str(current_user.user_id)))
        return {"preferences": (profile or {}).get("preferences", {})}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("set_preferences failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to update preferences")


@router.get("/history")
async def user_history(limit: int = 10, offset: int = 0, current_user=Depends(get_current_user)):
    try:
        items = await research_store.get_user_research(str(current_user.user_id), limit + offset)
        items.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        page = items[offset: offset + limit]
        history = [
            {
                "research_id": r.get("id"),
                "query": r.get("query"),
                "status": r.get("status"),
                "paradigm": (r.get("paradigm_classification") or {}).get("primary", "unknown"),
                "created_at": r.get("created_at"),
                "options": r.get("options", {}),
            }
            for r in page
        ]
        return {"total": len(items), "limit": limit, "offset": offset, "history": history}
    except Exception as e:
        logger.error("user_history failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to fetch history")
