import pytest
from fastapi.testclient import TestClient

# Assuming FastAPI app is exposed as app in main_new or main
try:
    from main_new import app  # noqa
except Exception:
    try:
        from main import app  # type: ignore # noqa
    except Exception:  # pragma: no cover
        app = None  # type: ignore

pytestmark = pytest.mark.asyncio


@pytest.mark.skipif(app is None, reason="App not available for metrics test")
def test_extended_stats_shape():
    client = TestClient(app)

    # Warm up: attempt lightweight route to seed metrics (best-effort)
    # Optional; if routes missing, ignore failures.
    try:
        client.get("/system/llm-ping")
    except Exception:
        pass

    resp = client.get("/system/extended-stats")
    assert resp.status_code == 200
    data = resp.json()

    # Required top-level keys
    for key in [
        "latency",
        "fallback_rates",
        "llm_usage",
        "paradigm_distribution",
        "quality",
        "counters",
        "timestamp",
    ]:
        assert key in data, f"Missing key {key} in extended stats"

    # Latency percentiles structure if present
    if data["latency"]:
        for stage, dist in data["latency"].items():
            for p in ["p50", "p95", "p99"]:
                assert p in dist, f"Missing percentile {p} for stage {stage}"
                assert isinstance(dist[p], (int, float))

    # Fallback rates values between 0 and 1
    for rate in data.get("fallback_rates", {}).values():
        assert 0.0 <= rate <= 1.0

    # Timestamp basic format check (ends with Z)
    assert data["timestamp"].endswith("Z")
