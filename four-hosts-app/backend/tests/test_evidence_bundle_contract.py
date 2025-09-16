import pytest

from models.evidence import EvidenceBundle, EvidenceQuote, EvidenceMatch, EvidenceDocument


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
    doc = EvidenceDocument(
        id="d001",
        url="https://example.com/report",
        title="Depth Report",
        domain="example.com",
        content="Full document content...",
        token_count=1200,
        word_count=4500,
        truncated=True,
    )
    eb = EvidenceBundle(
        quotes=[q],
        matches=[m],
        by_domain={"example.com": 1},
        focus_areas=["theme"],
        documents=[doc],
        documents_token_count=doc.token_count,
    )

    data = eb.model_dump()
    assert data["quotes"][0]["id"] == "q001"
    assert data["by_domain"]["example.com"] == 1
    # Validate via Pydantic
    eb2 = EvidenceBundle.model_validate(data)
    assert eb2.quotes[0].domain == "example.com"
