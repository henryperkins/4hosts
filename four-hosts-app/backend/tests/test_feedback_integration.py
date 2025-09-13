import pytest
from fastapi.testclient import TestClient

try:
    from main_new import app  # noqa
except Exception:
    try:
        from main import app  # type: ignore # noqa
    except Exception:  # pragma: no cover
        app = None  # type: ignore


pytestmark = pytest.mark.asyncio


def _fake_user():
    class U:
        user_id = "test-user-123"
        id = user_id
        email = "t@example.com"
        role = "admin"
    return U()


@pytest.mark.skipif(app is None, reason="App not available")
def test_feedback_endpoints_happy_path(monkeypatch):
    # Override auth dependency
    from backend.core.dependencies import get_current_user
    app.dependency_overrides[get_current_user] = lambda: _fake_user()

    client = TestClient(app)

    # Classification feedback
    payload_c = {
        "research_id": "research-abc",
        "query": "why did my cluster cost spike?",
        "original": {
            "primary": "bernard",
            "secondary": None,
            "distribution": {"bernard": 0.8, "maeve": 0.2},
            "confidence": 0.8
        },
        "user_correction": "maeve",
        "rationale": "Seeking strategy guidance"
    }
    r1 = client.post("/v1/feedback/classification", json=payload_c)
    assert r1.status_code == 201

    # Answer feedback
    payload_a = {
        "research_id": "research-abc",
        "rating": 0.75,
        "helpful": True,
        "improvements": ["add cost breakdown", "cite kubecost report"],
        "reason": "good synthesis"
    }
    r2 = client.post("/v1/feedback/answer", json=payload_a)
    assert r2.status_code == 201
    data = r2.json()
    assert pytest.approx(data.get("normalized_rating", 0), 0.01) == 0.75

    # Metrics counters should reflect submissions
    r3 = client.get("/v1/system/extended-stats")
    assert r3.status_code == 200
    counters = r3.json().get("counters", {})
    # flat keys are "" for no-label counters
    assert "" in counters.get("feedback_classification_submitted", {})
    assert "" in counters.get("feedback_answer_submitted", {})
    # star bucket was computed from 0.75 => 1 + round(0.75*4) = 4
    assert counters.get("feedback_answer_rating", {}).get("4", 0) >= 1

