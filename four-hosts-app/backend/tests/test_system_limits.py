import pytest

from core.limits import API_RATE_LIMITS, WS_RATE_LIMITS
from routes.system import get_limits


@pytest.mark.asyncio
async def test_limits_route_reflects_core_definitions():
    payload = await get_limits()

    plans = payload.get("plans", {})
    realtime = payload.get("realtime", {})

    for role, limits in API_RATE_LIMITS.items():
        key = role.value.lower()
        assert key in plans
        plan = plans[key]
        for field in [
            "requests_per_minute",
            "requests_per_hour",
            "requests_per_day",
            "concurrent_requests",
            "max_query_length",
            "max_sources",
        ]:
            assert plan[field] == limits[field]

    for role, limits in WS_RATE_LIMITS.items():
        key = role.value.lower()
        assert key in realtime
        assert realtime[key] == limits
