import asyncio
import types

import pytest

from backend.services.classification_engine import (
    ParadigmClassifier,
    QueryAnalyzer,
    HostParadigm,
    ParadigmScore,
)


def _mk_scores(mapping):
    out = {}
    for p, s in mapping.items():
        out[p] = ParadigmScore(paradigm=p, score=s, confidence=min(s/10.0, 1.0), reasoning=[], keyword_matches=[])
    return out


@pytest.mark.asyncio
async def test_secondary_threshold_above(monkeypatch):
    analyzer = QueryAnalyzer()
    clf = ParadigmClassifier(analyzer, use_llm=False)

    # Monkeypatch rule-based + analyzer to return fixed scores/features
    async def fake_classify(self, query: str, research_id=None, progress_tracker=None):
        rule_scores = _mk_scores({
            HostParadigm.BERNARD: 10.0,
            HostParadigm.MAEVE: 5.0,
            HostParadigm.DOLORES: 1.0,
            HostParadigm.TEDDY: 1.0,
        })
        final = self._normalize_scores(rule_scores)
        # Emulate the final selection logic from classify()
        sorted_paradigms = sorted(final.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_paradigms[0][0]
        secondary = (sorted_paradigms[1][0] if sorted_paradigms[1][1] > 0.2 else None)
        assert primary == HostParadigm.BERNARD
        assert secondary == HostParadigm.MAEVE  # 5/(10+5+1+1) ~= 0.294 > 0.2
    # Run the inline assertion as the test
    await fake_classify(clf, "query")


@pytest.mark.asyncio
async def test_secondary_threshold_below(monkeypatch):
    analyzer = QueryAnalyzer()
    clf = ParadigmClassifier(analyzer, use_llm=False)

    rule_scores = _mk_scores({
        HostParadigm.BERNARD: 10.0,
        HostParadigm.MAEVE: 1.0,
        HostParadigm.DOLORES: 1.0,
        HostParadigm.TEDDY: 1.0,
    })
    final = clf._normalize_scores(rule_scores)
    sorted_paradigms = sorted(final.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_paradigms[0][0]
    secondary = (sorted_paradigms[1][0] if sorted_paradigms[1][1] > 0.2 else None)
    assert primary == HostParadigm.BERNARD
    assert secondary is None  # 1/13 ~= 0.0769


def test_confidence_calculation():
    analyzer = QueryAnalyzer()
    clf = ParadigmClassifier(analyzer, use_llm=False)
    scores = _mk_scores({
        HostParadigm.BERNARD: 8.0,
        HostParadigm.MAEVE: 4.0,
        HostParadigm.DOLORES: 2.0,
        HostParadigm.TEDDY: 1.0,
    })
    distribution = clf._normalize_scores(scores)
    conf = clf._calculate_confidence(distribution, scores)
    # spread = p1 - p2 = (8/15) - (4/15) = 4/15 ~= 0.2667
    # top_score_confidence = 8/10 = 0.8
    # final = 0.5*0.2667 + 0.5*0.8 = ~= 0.53335 (capped at 0.95)
    assert 0.52 < conf < 0.55

