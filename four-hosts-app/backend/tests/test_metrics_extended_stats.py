import asyncio
from datetime import datetime, timezone

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


@pytest.mark.skipif(app is None, reason="App not available for metrics test")
def test_search_metrics_endpoint_records():
    from services.telemetry_pipeline import telemetry_pipeline

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "paradigm": "bernard",
        "depth": "standard",
        "total_queries": 3,
        "total_results": 12,
        "deduplication_rate": 0.25,
        "apis_used": ["brave", "google"],
        "provider_costs": {"brave": 0.01, "google": 0.02},
        "processing_time_seconds": 42.0,
        "stage_breakdown": {"rule_based": 2, "paradigm": 1},
    }

    asyncio.run(telemetry_pipeline.record_search_run(event))

    client = TestClient(app)
    resp = client.get("/system/search-metrics?window_minutes=1440")
    assert resp.status_code == 200
    data = resp.json()

    assert data["total_queries"] >= 3
    assert data["runs"] >= 1
    assert "timeline" in data and isinstance(data["timeline"], list)
    assert data.get("provider_usage", {}).get("brave", 0) >= 1
    assert data.get("total_cost_usd", 0) >= 0.03 - 1e-6
