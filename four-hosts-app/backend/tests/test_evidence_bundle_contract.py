import pytest

from models.evidence import EvidenceBundle, EvidenceQuote, EvidenceMatch


def test_evidence_bundle_model_roundtrip():
    q = EvidenceQuote(
        id="q001",
        url="https://example.com/post",
        title="Example",
        domain="example.com",
        quote="A relevant sentence used as evidence.",
        credibility_score=0.82,
        doc_summary="A brief summary",
    )
    m = EvidenceMatch(domain="example.com", fragments=["fragment A", "fragment B"])
    eb = EvidenceBundle(quotes=[q], matches=[m], by_domain={"example.com": 1}, focus_areas=["theme"])

    data = eb.model_dump()
    assert data["quotes"][0]["id"] == "q001"
    assert data["by_domain"]["example.com"] == 1
    # Validate via Pydantic
    eb2 = EvidenceBundle.model_validate(data)
    assert eb2.quotes[0].domain == "example.com"

