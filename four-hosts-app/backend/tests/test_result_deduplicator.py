"""
Comprehensive tests for ResultDeduplicator simhash algorithm correctness.
Tests cover deduplication accuracy, edge cases, and performance characteristics.
"""

import pytest
import asyncio
from typing import List, Dict, Any
from unittest.mock import Mock, patch
from services.query_planning.result_deduplicator import ResultDeduplicator
from services.search_apis import SearchResult


class TestResultDeduplicator:
    """Test suite for ResultDeduplicator simhash-based deduplication."""

    @pytest.fixture
    def deduplicator(self) -> ResultDeduplicator:
        """Create a fresh deduplicator instance for each test."""
        return ResultDeduplicator()

    @pytest.fixture
    def sample_results(self) -> List[SearchResult]:
        """Create sample search results for testing."""
        return [
            SearchResult(
                title="Python Programming Guide",
                url="https://example.com/python-guide",
                snippet="Learn Python programming basics",
                source="test",
                domain="example.com",
                content="Complete guide to Python programming with examples"
            ),
            SearchResult(
                title="Python Programming Tutorial",
                url="https://example.com/python-tutorial",
                snippet="Step by step Python tutorial",
                source="test",
                domain="example.com",
                content="Tutorial covering Python basics and advanced topics"
            ),
            SearchResult(
                title="JavaScript Guide",
                url="https://example.com/js-guide",
                snippet="Learn JavaScript programming",
                source="test",
                domain="example.com",
                content="JavaScript programming guide"
            ),
            SearchResult(
                title="Python Programming Guide",  # Exact duplicate
                url="https://example.com/python-guide",
                snippet="Learn Python programming basics",
                source="test",
                domain="example.com",
                content="Complete guide to Python programming with examples"
            ),
            SearchResult(
                title="Python Programming Guide",  # Near duplicate
                url="https://example.com/python-guide-copy",
                snippet="Learn Python programming basics and fundamentals",
                source="test",
                domain="example.com",
                content="Complete guide to Python programming with examples and exercises"
            )
        ]

    def test_simhash_algorithm_correctness(self, deduplicator: ResultDeduplicator, sample_results: List[SearchResult]):
        """Test that simhash algorithm correctly identifies duplicates."""
        result = asyncio.run(deduplicator.deduplicate_results(sample_results))

        # Should identify duplicates correctly
        assert "unique_results" in result
        assert "duplicates_removed" in result
        assert "duplicate_groups" in result

        unique_results = result["unique_results"]
        duplicates_removed = result["duplicates_removed"]

        # Should have removed at least the exact duplicate
        assert duplicates_removed >= 1
        assert len(unique_results) < len(sample_results)

    def test_exact_duplicate_detection(self, deduplicator: ResultDeduplicator):
        """Test detection of exact duplicate results."""
        exact_duplicates = [
            SearchResult(
                title="Test Article",
                url="https://example.com/article",
                snippet="Test snippet",
                source="test",
                domain="example.com",
                content="Test content"
            ),
            SearchResult(
                title="Test Article",
                url="https://example.com/article",
                snippet="Test snippet",
                source="test",
                domain="example.com",
                content="Test content"
            )
        ]

        result = asyncio.run(deduplicator.deduplicate_results(exact_duplicates))

        assert result["duplicates_removed"] == 1
        assert len(result["unique_results"]) == 1

    def test_near_duplicate_detection(self, deduplicator: ResultDeduplicator):
        """Test detection of near-duplicate results with minor variations."""
        near_duplicates = [
            SearchResult(
                title="Machine Learning Basics",
                url="https://example.com/ml-basics",
                snippet="Introduction to machine learning",
                source="test",
                domain="example.com",
                content="Basic concepts of machine learning algorithms"
            ),
            SearchResult(
                title="Machine Learning Fundamentals",
                url="https://example.com/ml-fundamentals",
                snippet="Introduction to ML concepts",
                source="test",
                domain="example.com",
                content="Basic concepts of machine learning algorithms and techniques"
            ),
            SearchResult(
                title="Deep Learning Guide",
                url="https://example.com/dl-guide",
                snippet="Advanced neural networks",
                source="test",
                domain="example.com",
                content="Deep learning with neural networks"
            )
        ]

        result = asyncio.run(deduplicator.deduplicate_results(near_duplicates))

        # Should detect the two ML articles as near duplicates
        assert result["duplicates_removed"] >= 1
        assert len(result["unique_results"]) <= 2

    def test_different_domains_preserved(self, deduplicator: ResultDeduplicator):
        """Test that results from different domains are preserved."""
        multi_domain_results = [
            SearchResult(
                title="AI Research",
                url="https://stanford.edu/ai-paper",
                snippet="Stanford AI research paper",
                source="stanford",
                domain="stanford.edu",
                content="Research on artificial intelligence"
            ),
            SearchResult(
                title="AI Research",
                url="https://mit.edu/ai-paper",
                snippet="MIT AI research paper",
                source="mit",
                domain="mit.edu",
                content="Research on artificial intelligence"
            ),
            SearchResult(
                title="AI Research",
                url="https://berkeley.edu/ai-paper",
                snippet="Berkeley AI research paper",
                source="berkeley",
                domain="berkeley.edu",
                content="Research on artificial intelligence"
            )
        ]

        result = asyncio.run(deduplicator.deduplicate_results(multi_domain_results))

        # Should preserve all results from different domains
        assert result["duplicates_removed"] == 0
        assert len(result["unique_results"]) == 3

    def test_empty_content_handling(self, deduplicator: ResultDeduplicator):
        """Test handling of results with empty or minimal content."""
        empty_content_results = [
            SearchResult(
                title="Empty Result 1",
                url="https://example.com/empty1",
                snippet="",
                source="test",
                domain="example.com",
                content=""
            ),
            SearchResult(
                title="Empty Result 2",
                url="https://example.com/empty2",
                snippet="",
                source="test",
                domain="example.com",
                content=""
            )
        ]

        result = asyncio.run(deduplicator.deduplicate_results(empty_content_results))

        # Should handle empty content gracefully
        assert "unique_results" in result
        assert len(result["unique_results"]) <= 2

    def test_large_scale_deduplication(self, deduplicator: ResultDeduplicator):
        """Test deduplication performance with large result sets."""
        # Create a large set of results with some duplicates
        large_results = []
        for i in range(100):
            large_results.append(SearchResult(
                title=f"Article {i % 10}",  # Creates duplicates every 10 items
                url=f"https://example.com/article{i}",
                snippet=f"Content of article {i}",
                source="test",
                domain="example.com",
                content=f"Full content of article {i}"
            ))

        result = asyncio.run(deduplicator.deduplicate_results(large_results))

        # Should identify duplicates and reduce the set
        assert result["duplicates_removed"] > 0
        assert len(result["unique_results"]) < len(large_results)

    def test_malformed_results_handling(self, deduplicator: ResultDeduplicator):
        """Test handling of malformed or incomplete results."""
        malformed_results = [
            SearchResult(
                title="Valid Result",
                url="https://example.com/valid",
                snippet="Valid snippet",
                source="test",
                domain="example.com",
                content="Valid content"
            ),
            SearchResult(
                title="",  # Empty title
                url="https://example.com/empty-title",
                snippet="",
                source="test",
                domain="example.com",
                content=""
            ),
            SearchResult(
                title="No URL Result",
                url="",  # Empty URL
                snippet="No URL snippet",
                source="test",
                domain="",
                content="No URL content"
            )
        ]

        result = asyncio.run(deduplicator.deduplicate_results(malformed_results))

        # Should handle malformed results gracefully
        assert "unique_results" in result
        assert len(result["unique_results"]) >= 1

    def test_deduplication_stats_accuracy(self, deduplicator: ResultDeduplicator, sample_results: List[SearchResult]):
        """Test that deduplication statistics are accurate."""
        result = asyncio.run(deduplicator.deduplicate_results(sample_results))

        original_count = len(sample_results)
        final_count = len(result["unique_results"])
        duplicates_removed = result["duplicates_removed"]

        # Stats should be mathematically consistent
        assert final_count == original_count - duplicates_removed
        assert duplicates_removed >= 0
        assert final_count >= 1

    def test_duplicate_groups_structure(self, deduplicator: ResultDeduplicator, sample_results: List[SearchResult]):
        """Test that duplicate groups are properly structured."""
        result = asyncio.run(deduplicator.deduplicate_results(sample_results))

        duplicate_groups = result.get("duplicate_groups", {})

        # If there are duplicate groups, they should be properly structured
        for group_id, group in duplicate_groups.items():
            assert isinstance(group_id, str)
            assert isinstance(group, list)
            assert len(group) >= 2  # Groups should have at least 2 items

    def test_deduplication_persistence(self, deduplicator: ResultDeduplicator):
        """Test that deduplication results are consistent across multiple calls."""
        test_results = [
            SearchResult(
                title="Test Article",
                url="https://example.com/test1",
                snippet="Test content",
                source="test",
                domain="example.com",
                content="Test content"
            ),
            SearchResult(
                title="Test Article",
                url="https://example.com/test2",
                snippet="Test content",
                source="test",
                domain="example.com",
                content="Test content"
            )
        ]

        # Run deduplication multiple times
        result1 = asyncio.run(deduplicator.deduplicate_results(test_results))
        result2 = asyncio.run(deduplicator.deduplicate_results(test_results))

        # Results should be consistent
        assert result1["duplicates_removed"] == result2["duplicates_removed"]
        assert len(result1["unique_results"]) == len(result2["unique_results"])

    def test_memory_efficiency(self, deduplicator: ResultDeduplicator):
        """Test that deduplication doesn't consume excessive memory."""
        import psutil
        import os

        # Create a moderately large dataset
        large_results = []
        for i in range(1000):
            large_results.append(SearchResult(
                title=f"Article {i}",
                url=f"https://example.com/article{i}",
                snippet=f"Snippet {i}",
                source="test",
                domain="example.com",
                content=f"Content {i}"
            ))

        # Monitor memory usage
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss

        result = asyncio.run(deduplicator.deduplicate_results(large_results))

        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before

        # Memory increase should be reasonable (less than 100MB for 1000 results)
        assert memory_increase < 100 * 1024 * 1024  # 100MB limit

        # Should still produce valid results
        assert len(result["unique_results"]) > 0
        assert "duplicates_removed" in result
