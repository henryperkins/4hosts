import os
import sys
import pytest
from datetime import datetime, timedelta

# Ensure backend root is on sys.path for 'services' imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.credibility import get_source_credibility


@pytest.mark.asyncio
async def test_exa_highlights_feed_credibility_terms(monkeypatch):
    # Highlights containing a controversial term should increase controversy score
    domain = "example.com"
    recent = datetime.now() - timedelta(days=5)
    old = datetime.now() - timedelta(days=365 * 5)

    # Recent article with controversial terms
    cred_recent = await get_source_credibility(
        domain=domain,
        published_date=recent,
        search_terms=["abortion", "policy debate"],
    )
    assert 0.0 <= cred_recent.overall_score <= 1.0
    assert 0.0 <= cred_recent.recency_score <= 1.0
    assert cred_recent.controversy_score > 0.0  # reacts to controversial term

    # Older article: recency should be lower than recent
    cred_old = await get_source_credibility(
        domain=domain,
        published_date=old,
        search_terms=["neutral"],
    )
    assert cred_old.recency_score < cred_recent.recency_score
