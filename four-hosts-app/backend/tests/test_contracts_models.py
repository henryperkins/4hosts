"""Unit tests for newly introduced `backend.contracts` package.

These tests are intentionally lightweight – they only verify that the
contract models can be imported and instantiated without raising, ensuring
that PR0 does **not** break the runtime.
"""

from datetime import datetime

import pytest


def test_contracts_import_and_basic_instantiation():
    """Smoke-test that all public contract objects round-trip basic data."""

    from backend.contracts import (
        GeneratedAnswer,
        ResearchBundle,
        ResearchStatus,
        SearchResult,
        Source,
    )

    src = Source(
        url="https://example.com/article",
        title="Example Article",
        snippet="Lorem ipsum dolor sit amet.",
        score=0.88,
        metadata={"fetched_at": datetime.utcnow().isoformat()},
    )

    search = SearchResult(query="foo bar", sources=[src])

    bundle = ResearchBundle(query="foo bar", sources=search.sources)

    answer = GeneratedAnswer(
        status=ResearchStatus.OK,
        content_md="# Answer\nBody",
        citations=[src],
        quality_score=0.95,
    )

    assert answer.status is ResearchStatus.OK
    assert bundle.sources[0].url == src.url
