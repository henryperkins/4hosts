"""
Comprehensive tests for EarlyRelevanceFilter spam detection accuracy.
Tests cover relevance scoring, spam detection, and filtering accuracy.
"""

import pytest
from typing import List
from services.query_planning.relevance_filter import EarlyRelevanceFilter
from services.search_apis import SearchResult


class TestEarlyRelevanceFilter:
    """Test suite covering spam detection and paradigm relevance scoring."""

    @pytest.fixture
    def relevance_filter(self) -> EarlyRelevanceFilter:
        """Create a fresh relevance filter instance for each test."""
        return EarlyRelevanceFilter()

    @pytest.fixture
    def sample_query(self) -> str:
        """Sample query for testing."""
        return "machine learning algorithms"

    @pytest.fixture
    def relevant_results(self, sample_query: str) -> List[SearchResult]:
        """Create sample relevant search results."""
        return [
            SearchResult(
                title="Machine Learning Algorithms: A Comprehensive Guide",
                url="https://example.com/ml-guide",
                snippet=(
                    "This guide covers various machine learning algorithms "
                    "including supervised and unsupervised learning methods."
                ),
                source="test",
                domain="example.com",
                content=(
                    "Detailed explanation of machine learning algorithms with "
                    "examples and code samples."
                ),
            ),
            SearchResult(
                title="Deep Learning Neural Networks",
                url="https://example.com/deep-learning",
                snippet=(
                    "Introduction to neural networks and deep learning "
                    "architectures used in modern AI systems."
                ),
                source="test",
                domain="example.com",
                content=(
                    "Comprehensive overview of deep learning techniques and "
                    "neural network architectures."
                ),
            ),
            SearchResult(
                title="Supervised vs Unsupervised Learning",
                url="https://example.com/supervised-learning",
                snippet=(
                    "Comparison of supervised and unsupervised learning "
                    "approaches in machine learning."
                ),
                source="test",
                domain="example.com",
                content=(
                    "Detailed comparison with examples of when to use each "
                    "approach."
                ),
            )
        ]

    @pytest.fixture
    def spam_results(self, sample_query: str) -> List[SearchResult]:
        """Create sample spam/irrelevant search results."""
        return [
            SearchResult(
                title="Buy Cheap Viagra Online - Best Prices!",
                url="https://spam-site.com/viagra",
                snippet=(
                    "Get the best deals on prescription medications. Fast "
                    "shipping worldwide."
                ),
                source="spam",
                domain="spam-site.com",
                content="Pharmacy spam content with drug advertisements.",
            ),
            SearchResult(
                title="Click Here to Win iPhone - Free Giveaway!",
                url="https://scam-site.com/iphone-giveaway",
                snippet=(
                    "Enter our contest to win a free iPhone 15. "
                    "No purchase necessary!"
                ),
                source="scam",
                domain="scam-site.com",
                content="Scam content with fake giveaway promotions.",
            ),
            SearchResult(
                title="Lose Weight Fast - Miracle Diet Pills",
                url="https://miracle-diet.com/pills",
                snippet=(
                    "Amazing weight loss results in just 7 days. "
                    "Buy now and save 50%!"
                ),
                source="spam",
                domain="miracle-diet.com",
                content="Spam content promoting weight loss supplements.",
            ),
            SearchResult(
                title="Work from Home - Make $5000/Week",
                url="https://get-rich-quick.com/work-home",
                snippet=(
                    "Earn money online with our proven system. "
                    "Start today!"
                ),
                source="scam",
                domain="get-rich-quick.com",
                content="Get rich quick scheme promotion.",
            )
        ]

    @pytest.fixture
    def mixed_results(self, relevant_results: List[SearchResult], spam_results: List[SearchResult]) -> List[SearchResult]:
        """Create a mixed set of relevant and spam results."""
        return relevant_results + spam_results

    def test_relevance_scoring_accuracy(self, relevance_filter: EarlyRelevanceFilter, sample_query: str, relevant_results: List[SearchResult]):
        """Test that relevant results are marked as relevant."""
        for result in relevant_results:
            is_relevant = relevance_filter.is_relevant(result, sample_query, "bernard")

            # Relevant results should be marked as relevant
            assert is_relevant == True

    def test_spam_detection_accuracy(self, relevance_filter: EarlyRelevanceFilter, sample_query: str, spam_results: List[SearchResult]):
        """Test that spam results are marked as irrelevant."""
        for result in spam_results:
            is_relevant = relevance_filter.is_relevant(result, sample_query, "bernard")

            # Spam results should be marked as irrelevant
            assert is_relevant == False

    def test_spam_filtering_effectiveness(self, relevance_filter: EarlyRelevanceFilter, sample_query: str, mixed_results: List[SearchResult]):
        """Test that the filter effectively removes spam while preserving relevant results."""
        # Filter results using is_relevant method
        filtered_results = [r for r in mixed_results if relevance_filter.is_relevant(r, sample_query, "bernard")]

        # Should preserve most relevant results
        relevant_preserved = sum(1 for r in filtered_results if r.source == "test")
        spam_removed = sum(1 for r in mixed_results if r.source == "spam") - sum(1 for r in filtered_results if r.source == "spam")

        # Should preserve at least 80% of relevant results
        assert relevant_preserved >= 2  # At least 2 out of 3 relevant results

        # Should remove most spam results
        assert spam_removed >= 3  # At least 3 out of 4 spam results

    def test_spam_keyword_detection(self, relevance_filter: EarlyRelevanceFilter):
        """Test that spam keywords are properly detected."""
        query = "machine learning"

        # Test various spam indicators
        spam_indicators = ["viagra", "casino", "poker", "weight loss", "get rich quick"]

        for spam_word in spam_indicators:
            spam_result = SearchResult(
                title=f"Amazing {spam_word} deals",
                url=f"https://spam-site.com/{spam_word}",
                snippet=f"Get the best {spam_word} offers now!",
                source="spam",
                domain="spam-site.com",
                content=f"Spam content about {spam_word}",
            )

            is_relevant = relevance_filter.is_relevant(spam_result, query, "bernard")
            assert is_relevant == False, f"Spam word '{spam_word}' should be detected"

    def test_domain_blocking(self, relevance_filter: EarlyRelevanceFilter):
        """Test that low-quality domains are blocked."""
        query = "test query"

        # Test blocked domains
        blocked_domains = ["ezinearticles.com", "articlesbase.com", "hubpages.com"]

        for domain in blocked_domains:
            blocked_result = SearchResult(
                title="Test Article",
                url=f"https://{domain}/test-article",
                snippet="Test snippet",
                source="test",
                domain=domain,
                content="Test content",
            )

            is_relevant = relevance_filter.is_relevant(blocked_result, query, "bernard")
            assert is_relevant == False, f"Domain '{domain}' should be blocked"

    def test_minimum_content_length(self, relevance_filter: EarlyRelevanceFilter):
        """Test that results with insufficient content are filtered out."""
        query = "test query"

        # Result with very short title
        short_title_result = SearchResult(
            title="Hi",  # Less than 10 characters
            url="https://example.com/short",
            snippet="This is a test snippet with sufficient length",
            source="test",
            domain="example.com",
            content="Test content"
        )

        # Result with very short snippet
        short_snippet_result = SearchResult(
            title="Test Article",
            url="https://example.com/short-snippet",
            snippet="Hi",  # Less than 20 characters
            source="test",
            domain="example.com",
            content="Test content"
        )

        is_relevant_short_title = relevance_filter.is_relevant(short_title_result, query, "bernard")
        is_relevant_short_snippet = relevance_filter.is_relevant(short_snippet_result, query, "bernard")

        # Should be filtered out due to insufficient content length
        assert is_relevant_short_title == False
        assert is_relevant_short_snippet == False

    def test_paradigm_specific_filtering(self, relevance_filter: EarlyRelevanceFilter):
        """Test that filtering behavior varies by paradigm."""
        query = "artificial intelligence research"

        academic_result = SearchResult(
            title="AI Research Paper",
            url="https://stanford.edu/ai-research",
            snippet=(
                "Scientific study on artificial intelligence with "
                "peer-reviewed methodology."
            ),
            source="stanford",
            domain="stanford.edu",
            content="Academic research content",
        )

        commercial_result = SearchResult(
            title="AI Marketing Tool",
            url="https://marketing-site.com/ai-tool",
            snippet=(
                "Use AI for better marketing campaigns and boost ROI quickly."
            ),
            source="commercial",
            domain="marketing-site.com",
            content="Commercial marketing content",
        )

        bernard_academic = relevance_filter.is_relevant(academic_result, query, "bernard")
        bernard_commercial = relevance_filter.is_relevant(commercial_result, query, "bernard")
        maeve_commercial = relevance_filter.is_relevant(commercial_result, query, "maeve")

        assert bernard_academic is True
        assert bernard_commercial is False
        assert maeve_commercial is True

    def test_paradigm_alignment_bonus(self, relevance_filter: EarlyRelevanceFilter):
        """Preferred domains and paradigm keywords should boost alignment."""
        query = "expose corruption at multinational corporations"
        dolores_aligned = SearchResult(
            title="Power Investigation uncovers systemic corruption",
            url="https://www.propublica.org/article/systemic-power-investigation",
            snippet=(
                "Whistleblower evidence shows systemic power abuse and "
                "corruption cover-up."
            ),
            source="propublica",
            domain="propublica.org",
            content="Detailed investigative report",
        )
        neutral_result = SearchResult(
            title="Company press release about compliance",
            url="https://example.com/compliance-report",
            snippet=(
                "Official statement outlining compliance improvements "
                "and policies."
            ),
            source="company",
            domain="example.com",
            content="Corporate content",
        )

        assert relevance_filter.is_relevant(dolores_aligned, query, "dolores") is True
        assert relevance_filter.is_relevant(neutral_result, query, "dolores") is False

    def test_duplicate_site_detection(self, relevance_filter: EarlyRelevanceFilter):
        """Test that duplicate/mirror sites are detected."""
        query = "test query"

        # Test duplicate site patterns
        duplicate_patterns = [
            "example-mirror.com",
            "example-cache.net",
            "example-proxy.org",
            "webcache.googleusercontent.com",
            "cached.example.com"
        ]

        for domain in duplicate_patterns:
            duplicate_result = SearchResult(
                title="Test Article",
                url=f"https://{domain}/test-article",
                snippet="Test snippet",
                source="test",
                domain=domain,
                content="Test content",
            )

            is_relevant = relevance_filter.is_relevant(duplicate_result, query, "bernard")
            assert is_relevant == False, f"Duplicate site pattern '{domain}' should be detected"

    def test_query_compression_integration(self, relevance_filter: EarlyRelevanceFilter):
        """Test that the filter integrates properly with query compression."""
        # This test verifies that the filter can handle compressed queries
        # without throwing exceptions
        query = "machine learning artificial intelligence deep neural networks"

        result = SearchResult(
            title="ML Research",
            url="https://example.com/ml-research",
            snippet="Research on machine learning",
            source="test",
            domain="example.com",
            content="Research content",
        )

        # Should not crash and should return a boolean
        is_relevant = relevance_filter.is_relevant(result, query, "bernard")
        assert isinstance(is_relevant, bool)

    def test_filter_consistency(self, relevance_filter: EarlyRelevanceFilter):
        """Test that filtering is consistent across multiple calls."""
        query = "test query"
        result = SearchResult(
            title="Test Result",
            url="https://example.com/test",
            snippet="Test snippet",
            source="test",
            domain="example.com",
            content="Test content",
        )

        # Call multiple times
        result1 = relevance_filter.is_relevant(result, query, "bernard")
        result2 = relevance_filter.is_relevant(result, query, "bernard")
        result3 = relevance_filter.is_relevant(result, query, "bernard")

        # Results should be identical
        assert result1 == result2 == result3

    def test_malformed_input_handling(self, relevance_filter: EarlyRelevanceFilter):
        """Test that the filter handles malformed input gracefully."""
        query = "test query"

        # Test with None values
        malformed_result = SearchResult(
            title=None,
            url=None,
            snippet=None,
            source="test",
            domain=None,
            content=None,
        )

        # Should not crash and should return a boolean
        is_relevant = relevance_filter.is_relevant(malformed_result, query, "bernard")
        assert isinstance(is_relevant, bool)
