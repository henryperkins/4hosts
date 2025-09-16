import os


def test_safe_evidence_block_handles_missing_quotes():
    # Lazy import to avoid heavy imports at module load
    from services.answer_generator import BernardAnswerGenerator, SynthesisContext

    # Minimal context: evidence_bundle is None
    ctx = SynthesisContext(
        query="test",
        paradigm="bernard",
        search_results=[],
        context_engineering={},
        metadata={},
        evidence_bundle=None,
    )

    gen = BernardAnswerGenerator()

    out = gen._safe_evidence_block(ctx)
    assert isinstance(out, str)
    assert "evidence quotes" in out.lower()


def test_safe_evidence_block_handles_empty_quotes():
    from services.answer_generator import BernardAnswerGenerator, SynthesisContext
    from models.evidence import EvidenceBundle

    # EvidenceBundle with empty quotes list
    eb = EvidenceBundle(quotes=[], matches=[], by_domain={}, focus_areas=[])
    ctx = SynthesisContext(
        query="test",
        paradigm="bernard",
        search_results=[],
        context_engineering={},
        metadata={},
        evidence_bundle=eb,
    )

    gen = BernardAnswerGenerator()
    out = gen._safe_evidence_block(ctx)
    assert isinstance(out, str)
    assert "evidence quotes" in out.lower()

