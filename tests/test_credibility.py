import asyncio
import sys, os, pathlib
# Ensure project root is on path so `import source_credibility` succeeds when
# tests are run from a sub-directory.
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from datetime import timedelta

import pytest

# date util lives under backend utils; load it dynamically
import importlib.util as _iu
import pathlib as _pl

_date_path = _pl.Path(__file__).resolve().parent.parent / "four-hosts-app" / "backend" / "utils" / "date_utils.py"
_spec_date = _iu.spec_from_file_location("_date_utils", _date_path)
_date_mod = _iu.module_from_spec(_spec_date)
_spec_date.loader.exec_module(_date_mod)  # type: ignore
get_current_utc = _date_mod.get_current_utc  # type: ignore

# The top-level wrapper exposes the public API used in downstream code and tests.
from source_credibility import (
    DomainAuthorityChecker,
    CredibilityScore,
    ControversyDetector,
    SourceReputationDatabase,
    analyze_source_credibility_batch,
    get_source_credibility,
)


@pytest.mark.asyncio
async def test_da_heuristic_tld():
    """`.gov` and `.edu` domains should receive high heuristic DA."""
    dac = DomainAuthorityChecker()
    assert await dac.get_domain_authority("whitehouse.gov") >= 80
    assert await dac.get_domain_authority("mit.edu") >= 75


def test_card_strengths_concerns_bucketting():
    """Verify strengths/concerns are bucketed based on whitelists."""
    c = CredibilityScore(
        domain="example.com",
        overall_score=0.7,
        reputation_factors=[
            "High Domain Authority",
            "Highly Controversial",
            "Low Source Agreement",
        ],
        controversy_indicators=["Contains conflicting viewpoints"],
    )
    card = c.generate_credibility_card()
    assert "High Domain Authority" in card["strengths"]
    assert "Highly Controversial" in card["concerns"]
    assert any("conflicting" in s for s in card["concerns"])


def test_controversy_word_boundaries():
    """Keyword matching must respect word boundaries to avoid false positives."""
    cd = ControversyDetector()
    score1, _ = cd.calculate_controversy_score(
        "d", content="We discuss transfers and transformers", search_terms=None
    )
    score2, _ = cd.calculate_controversy_score(
        "d", content="trans rights debate", search_terms=None
    )
    assert score2 > score1  # boundary avoids false positives


@pytest.mark.asyncio
async def test_bias_heuristics_used_for_unknown():
    """Unknown domains fall back to heuristic bias scoring."""
    srd = SourceReputationDatabase()
    cred = await srd.calculate_credibility_score("example-blog.foobar")
    assert cred.bias_rating in {"center", "left", "right", "mixed"}
    assert cred.source_category in {
        "general",
        "blog",
        "news",
        "tech",
        "reference",
        "academic",
        "government",
        "social",
        "video",
        "pdf",
    }


@pytest.mark.asyncio
async def test_batch_parallel_empty_and_nonempty():
    # Empty list should return zeros rather than raising.
    out = await analyze_source_credibility_batch([])
    assert out["total_sources"] == 0

    # Non-empty list should still work (smoke test).
    out2 = await analyze_source_credibility_batch(
        [{"domain": "reuters.com"}, {"domain": "apnews.com"}]
    )
    assert out2["total_sources"] == 2


@pytest.mark.asyncio
async def test_smoke_get_source_credibility():
    c = await get_source_credibility(
        "reuters.com",
        paradigm="bernard",
        published_date=get_current_utc() - timedelta(days=2),
    )
    assert 0 <= c.overall_score <= 1
