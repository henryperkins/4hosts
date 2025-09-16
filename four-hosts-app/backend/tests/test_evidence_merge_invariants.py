import pytest

from services.deep_research_service import convert_citations_to_evidence_quotes


class DummyCitation:
    def __init__(self, title: str = "", url: str = "", start_index: int = 0, end_index: int = 0):
        self.title = title
        self.url = url
        self.start_index = start_index
        self.end_index = end_index


def test_convert_citations_preserves_unlinked_without_synthetic_urls():
    content = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    c1 = DummyCitation(title="Linked", url="https://example.com/x", start_index=0, end_index=20)
    c2 = DummyCitation(title="Unlinked", url="", start_index=21, end_index=40)

    quotes = convert_citations_to_evidence_quotes([c1, c2], content)
    assert len(quotes) == 2

    # Linked quote preserves URL and domain
    q1 = quotes[0]
    assert q1.url.startswith("http")
    assert q1.domain in ("example.com", "https://example.com/x")

    # Unlinked quote should NOT synthesize about:blank URLs
    q2 = quotes[1]
    assert isinstance(q2.url, str)
    assert q2.url == ""
    # Domain may be blank when url is blank
    assert q2.domain in ("",)

