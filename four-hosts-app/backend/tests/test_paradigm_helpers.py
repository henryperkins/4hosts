import pytest

from routes.search import compute_paradigm_alignment, extract_key_insights_from_result


def test_compute_paradigm_alignment_uses_keywords():
    text = "This analysis provides statistical evidence and data-driven research outcomes."
    score = compute_paradigm_alignment("bernard", text)
    assert score > 0
    assert score <= 1.0


def test_compute_paradigm_alignment_unknown_paradigm():
    assert compute_paradigm_alignment("unknown", "any text") == 0.0


def test_extract_key_insights_prefers_provider_highlights():
    data = {
        "raw_data": {
            "exa": {"highlights": ["First highlight", "Second highlight"]},
        }
    }
    insights = extract_key_insights_from_result(data)
    assert insights[:2] == ["First highlight", "Second highlight"]


def test_extract_key_insights_falls_back_to_sentences():
    snippet = (
        "This resource offers support and community programs for caregivers. "
        "It also outlines additional steps families can take to access help."
    )
    data = {"snippet": snippet}
    insights = extract_key_insights_from_result(data)
    assert insights
    assert all(isinstance(item, str) and item for item in insights)
    assert len(insights) <= 3
