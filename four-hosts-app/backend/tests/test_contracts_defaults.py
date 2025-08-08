"""Contracts default behavior tests

Ensure list/dict defaults are isolated per-instance and never shared.
"""

from backend.contracts import Source, GeneratedAnswer, ResearchStatus


def test_source_metadata_default_is_not_shared():
    a = Source(url="https://a.example", title="A")
    b = Source(url="https://b.example", title="B")

    a.metadata["x"] = 1
    assert "x" not in b.metadata


def test_generated_answer_defaults_are_not_shared():
    g1 = GeneratedAnswer(status=ResearchStatus.OK, content_md="")
    g2 = GeneratedAnswer(status=ResearchStatus.OK, content_md="")

    g1.citations.append(Source(url="https://c.example", title="C"))
    g1.diagnostics["trace"] = "t1"

    assert len(g2.citations) == 0
    assert "trace" not in g2.diagnostics

