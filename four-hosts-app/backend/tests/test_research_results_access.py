import pytest
from fastapi import HTTPException

from routes.research import get_research_results
from services.research_store import research_store
from models.base import ResearchStatus, UserRole


def _minimal_results_payload(rid: str) -> dict:
    # Minimal shape sufficient for route return and basic FE expectations
    return {
        "research_id": rid,
        "query": "unit test query",
        "status": "ok",
        "paradigm_analysis": {
            "primary": {"paradigm": "bernard", "confidence": 0.9}
        },
        "answer": {"summary": "", "sections": [], "action_items": [], "citations": [], "metadata": {}},
        "sources": [],
        "metadata": {
            "processing_time_seconds": 0.0,
            "total_sources_analyzed": 0,
            "high_quality_sources": 0,
            "credibility_summary": {"average_score": 0.0, "score_distribution": {"high": 0, "medium": 0, "low": 0}},
            "category_distribution": {},
            "bias_distribution": {},
        },
        "cost_info": {},
        "export_formats": {},
    }


class _User:
    # Helper that can provide either user_id or id for compatibility tests
    def __init__(self, user_id: str | None = None, id_: str | None = None, role: UserRole = UserRole.PRO):
        if user_id is not None:
            self.user_id = user_id
        if id_ is not None:
            self.id = id_
        self.role = role


@pytest.mark.asyncio
async def test_get_research_results_owner_with_user_id():
    rid = "res_unit_1"
    owner_id = "owner-uid-1"
    await research_store.set(rid, {
        "id": rid,
        "user_id": owner_id,
        "query": "x",
        "options": {},
        "status": ResearchStatus.COMPLETED,
        "created_at": "2025-01-01T00:00:00Z",
        "results": _minimal_results_payload(rid),
    })
    user = _User(user_id=owner_id, role=UserRole.PRO)
    result = await get_research_results(rid, current_user=user)
    assert isinstance(result, dict)
    assert result.get("research_id") == rid


@pytest.mark.asyncio
async def test_get_research_results_owner_with_id_compat():
    rid = "res_unit_2"
    owner_id = "owner-uid-2"
    await research_store.set(rid, {
        "id": rid,
        "user_id": owner_id,
        "query": "x",
        "options": {},
        "status": ResearchStatus.COMPLETED,
        "created_at": "2025-01-01T00:00:00Z",
        "results": _minimal_results_payload(rid),
    })
    # Only id attribute present (legacy shape)
    user = _User(id_=owner_id, role=UserRole.PRO)
    result = await get_research_results(rid, current_user=user)
    assert isinstance(result, dict)
    assert result.get("research_id") == rid


@pytest.mark.asyncio
async def test_get_research_results_access_denied_for_other_user():
    rid = "res_unit_3"
    owner_id = "owner-uid-3"
    await research_store.set(rid, {
        "id": rid,
        "user_id": owner_id,
        "query": "x",
        "options": {},
        "status": ResearchStatus.COMPLETED,
        "created_at": "2025-01-01T00:00:00Z",
        "results": _minimal_results_payload(rid),
    })
    other = _User(user_id="someone-else", role=UserRole.PRO)
    with pytest.raises(HTTPException) as ei:
        await get_research_results(rid, current_user=other)
    assert ei.value.status_code == 403