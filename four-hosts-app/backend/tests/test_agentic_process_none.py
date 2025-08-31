import pytest

from services.agentic_process import propose_queries_from_missing


def test_propose_queries_from_missing_handles_none_inputs():
    # Should not raise when original_query is None-like and terms include None/empty
    out = propose_queries_from_missing(None, "bernard", [None, " ", "evidence"], max_new=3)  # type: ignore[arg-type]
    assert isinstance(out, list)
    # Ensure non-empty proposals are produced for valid term
    assert any("evidence" in q for q in out)

