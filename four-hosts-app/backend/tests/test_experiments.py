import os
import importlib


def test_variant_assignment_deterministic(tmp_path, monkeypatch):
    from backend.services import experiments

    # Force specific weights for predictability
    monkeypatch.setenv("ENABLE_PROMPT_AB", "1")
    monkeypatch.setenv("PROMPT_AB_WEIGHTS", '{"A":0.3,"B":0.7}')

    # Reload to pick up env in the same process (weights read at call time too)
    importlib.reload(experiments)

    # Deterministic assignment for the same unit_id
    u = "user-1234"
    v1 = experiments.assign("trial", u)
    v2 = experiments.assign("trial", u)
    assert v1 == v2

    # Different unit_id should change the bucket sometimes
    # This is probabilistic; just assert variant is one of the declared keys.
    v_other = experiments.assign("trial", "someone-else")
    assert v_other in {"A", "B"}


def test_variant_or_default_disabled(monkeypatch):
    from backend.services import experiments
    monkeypatch.delenv("ENABLE_PROMPT_AB", raising=False)
    v = experiments.variant_or_default("trial", unit_id="u", default="control")
    assert v == "control"

