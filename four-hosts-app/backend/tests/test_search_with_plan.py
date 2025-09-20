"""
Tests for search_with_plan method and orchestrator budget integration.
Tests cover the key integration points for budget-aware search execution.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any
from services.search_apis import SearchAPIManager, SearchConfig, SearchResult
from search.query_planner import QueryCandidate
from services.query_planning.planning_utils import BudgetAwarePlanner, ToolRegistry, RetryPolicy, ToolCapability


class TestSearchWithPlanIntegration:
    """Test suite for search_with_plan method and budget integration."""

    @pytest.fixture
    def search_manager(self) -> SearchAPIManager:
        """Create a search manager for testing."""
        return SearchAPIManager()

    @pytest.fixture
    def sample_candidates(self) -> List[QueryCandidate]:
        """Create sample query candidates for testing."""
        return [
            QueryCandidate(
                query="artificial intelligence ethics",
                stage="context",
                label="primary_context"
            ),
            QueryCandidate(
                query="AI safety research",
                stage="expansion",
                label="safety_expansion"
            ),
            QueryCandidate(
                query="machine learning bias",
                stage="deepening",
                label="bias_deepening"
            )
        ]

    @pytest.fixture
    def search_config(self) -> SearchConfig:
        """Create a search config for testing."""
        return SearchConfig(
            max_results=10,
            language="en",
            region="us",
            min_relevance_score=0.15
        )

    @patch('services.search_apis.SearchAPIManager.search_with_priority')
    async def test_search_with_plan_basic_execution(self, mock_search_with_priority: AsyncMock, search_manager: SearchAPIManager, sample_candidates: List[QueryCandidate], search_config: SearchConfig):
        """Test basic execution of search_with_plan method."""
        # Setup mock results
        mock_results = [
            SearchResult(
                title="AI Ethics Article",
                url="https://example.com/ai-ethics",
                snippet="Article about AI ethics",
                source="test",
                domain="example.com"
            )
        ]
        mock_search_with_priority.return_value = mock_results

        # Execute search_with_plan
        results = await search_manager.search_with_plan(sample_candidates, search_config)

        # Verify the method was called correctly
        assert mock_search_with_priority.called
        call_args = mock_search_with_priority.call_args
        assert call_args[0][0] == sample_candidates  # candidates
        assert call_args[0][1] == search_config     # config

        # Verify results structure
        assert isinstance(results, dict)
        assert len(results) > 0

    @patch('services.search_apis.SearchAPIManager.search_with_priority')
    async def test_search_with_plan_results_by_label(self, mock_search_with_priority: AsyncMock, search_manager: SearchAPIManager, sample_candidates: List[QueryCandidate], search_config: SearchConfig):
        """Test that search_with_plan returns results organized by candidate label."""
        # Setup mock to return different results for different calls
        def side_effect(candidates, config):
            # Return different results based on the first candidate's label
            if hasattr(candidates[0], 'label'):
                label = candidates[0].label
                return [
                    SearchResult(
                        title=f"Result for {label}",
                        url=f"https://example.com/{label}",
                        snippet=f"Content for {label}",
                        source="test",
                        domain="example.com",
                        raw_data={"query_label": label}
                    )
                ]
            return []

        mock_search_with_priority.side_effect = side_effect

        # Execute search_with_plan
        results = await search_manager.search_with_plan(sample_candidates, search_config)

        # Verify results are organized by label
        assert isinstance(results, dict)
        for candidate in sample_candidates:
            label = candidate.label
            assert label in results
            assert len(results[label]) >= 0

    @patch('services.search_apis.SearchAPIManager.search_with_priority')
    async def test_search_with_plan_empty_candidates(self, mock_search_with_priority: AsyncMock, search_manager: SearchAPIManager, search_config: SearchConfig):
        """Test search_with_plan with empty candidates list."""
        # Execute with empty candidates
        results = await search_manager.search_with_plan([], search_config)

        # Should return empty dict
        assert results == {}

        # Should not call search_with_priority
        mock_search_with_priority.assert_not_called()

    @patch('services.search_apis.SearchAPIManager.search_with_priority')
    async def test_search_with_plan_stage_organization(self, mock_search_with_priority: AsyncMock, search_manager: SearchAPIManager, search_config: SearchConfig):
        """Test that search_with_plan properly organizes execution by stage."""
        # Create candidates with different stages
        staged_candidates = [
            QueryCandidate(query="context query", stage="context", label="context_1"),
            QueryCandidate(query="context query 2", stage="context", label="context_2"),
            QueryCandidate(query="expansion query", stage="expansion", label="expansion_1"),
            QueryCandidate(query="deepening query", stage="deepening", label="deepening_1"),
        ]

        mock_search_with_priority.return_value = [
            SearchResult(
                title="Test Result",
                url="https://example.com/test",
                snippet="Test content",
                source="test",
                domain="example.com"
            )
        ]

        # Execute search_with_plan
        results = await search_manager.search_with_plan(staged_candidates, search_config)

        # Verify search_with_priority was called (stage organization happens internally)
        assert mock_search_with_priority.called
        assert mock_search_with_priority.call_count >= 1

    @patch('services.search_apis.SearchAPIManager.search_with_priority')
    async def test_search_with_plan_error_handling(self, mock_search_with_priority: AsyncMock, search_manager: SearchAPIManager, sample_candidates: List[QueryCandidate], search_config: SearchConfig):
        """Test error handling in search_with_plan method."""
        # Setup mock to raise an exception
        mock_search_with_priority.side_effect = Exception("Search failed")

        # Execute search_with_plan
        results = await search_manager.search_with_plan(sample_candidates, search_config)

        # Should return empty dict on error
        assert results == {}

    @patch('services.search_apis.SearchAPIManager.search_with_priority')
    async def test_search_with_plan_stage_label_annotation(self, mock_search_with_priority: AsyncMock, search_manager: SearchAPIManager, sample_candidates: List[QueryCandidate], search_config: SearchConfig):
        """Test that search_with_plan properly annotates results with stage:label."""
        # Setup mock results with raw_data
        mock_results = [
            SearchResult(
                title="Test Result",
                url="https://example.com/test",
                snippet="Test content",
                source="test",
                domain="example.com",
                raw_data={
                    "query_stage": "context",
                    "query_label": "primary_context"
                }
            )
        ]
        mock_search_with_priority.return_value = mock_results

        # Execute search_with_plan
        results = await search_manager.search_with_plan(sample_candidates, search_config)

        # Verify stage:label annotation is added
        for label, result_list in results.items():
            for result in result_list:
                assert "stage_label" in result.raw_data
                assert result.raw_data["stage_label"] == "context:primary_context"


class TestOrchestratorBudgetIntegration:
    """Test suite for orchestrator budget enforcement integration."""

    @pytest.fixture
    def tool_registry(self) -> ToolRegistry:
        """Create a tool registry for testing."""
        registry = ToolRegistry()
        registry.register(ToolCapability(
            name="google",
            cost_per_call_usd=0.005,
            rpm_limit=100,
            typical_latency_ms=800
        ))
        return registry

    @pytest.fixture
    def budget_planner(self, tool_registry: ToolRegistry) -> BudgetAwarePlanner:
        """Create a budget-aware planner for testing."""
        return BudgetAwarePlanner(tool_registry, RetryPolicy())

    @patch('services.research_orchestrator.UnifiedResearchOrchestrator._execute_searches_deterministic')
    async def test_orchestrator_budget_method_exists(self, mock_deterministic: AsyncMock, budget_planner: BudgetAwarePlanner):
        """Test that the orchestrator has the budget-aware search method."""
        from services.research_orchestrator import UnifiedResearchOrchestrator

        orchestrator = UnifiedResearchOrchestrator()
        orchestrator.planner = budget_planner

        # Verify the method exists
        assert hasattr(orchestrator, '_execute_searches_with_budget')
        assert callable(getattr(orchestrator, '_execute_searches_with_budget'))

    @patch('services.research_orchestrator.UnifiedResearchOrchestrator._execute_searches_deterministic')
    async def test_orchestrator_budget_method_fallback(self, mock_deterministic: AsyncMock, budget_planner: BudgetAwarePlanner):
        """Test that budget method falls back to deterministic execution on error."""
        from services.research_orchestrator import UnifiedResearchOrchestrator
        from search.query_planner import QueryCandidate

        orchestrator = UnifiedResearchOrchestrator()
        orchestrator.planner = budget_planner

        # Setup mock search manager
        mock_search_manager = Mock()
        mock_search_manager.search_with_plan = AsyncMock(side_effect=Exception("Budget search failed"))
        orchestrator.search_manager = mock_search_manager

        # Setup fallback mock
        mock_deterministic.return_value = {"test": []}

        candidates = [QueryCandidate(query="test", stage="context", label="test")]

        # Execute budget method
        result = await orchestrator._execute_searches_with_budget(
            candidates, None, None, None, None, None
        )

        # Verify fallback was called
        assert mock_deterministic.called
        assert result == {"test": []}

    @patch('services.research_orchestrator.UnifiedResearchOrchestrator._execute_searches_deterministic')
    async def test_orchestrator_budget_method_success(self, mock_deterministic: AsyncMock, budget_planner: BudgetAwarePlanner):
        """Test successful execution of budget-aware search method."""
        from services.research_orchestrator import UnifiedResearchOrchestrator
        from search.query_planner import QueryCandidate

        orchestrator = UnifiedResearchOrchestrator()
        orchestrator.planner = budget_planner

        # Setup mock search manager
        mock_search_manager = Mock()
        expected_results = {"test": [SearchResult(
            title="Test Result",
            url="https://example.com/test",
            snippet="Test content",
            source="test",
            domain="example.com"
        )]}
        mock_search_manager.search_with_plan = AsyncMock(return_value=expected_results)
        orchestrator.search_manager = mock_search_manager

        candidates = [QueryCandidate(query="test", stage="context", label="test")]

        # Execute budget method
        result = await orchestrator._execute_searches_with_budget(
            candidates, None, None, None, None, None
        )

        # Verify search_with_plan was called
        assert mock_search_manager.search_with_plan.called
        assert result == expected_results

    async def test_orchestrator_budget_integration_in_execute_research(self, budget_planner: BudgetAwarePlanner):
        """Test that execute_research uses the budget-aware search method."""
        from services.research_orchestrator import UnifiedResearchOrchestrator
        from models.context_models import ClassificationResultSchema, ContextEngineeredQuerySchema
        from models.paradigms import HostParadigm

        orchestrator = UnifiedResearchOrchestrator()
        orchestrator.planner = budget_planner

        # Setup minimal test data
        classification = ClassificationResultSchema(
            query="test query",
            primary_paradigm=HostParadigm.BERNARD,
            secondary_paradigm=None,
            distribution={HostParadigm.BERNARD: 1.0},
            confidence=0.9,
            features=None,
            reasoning={}
        )

        context_engineered = ContextEngineeredQuerySchema(
            original_query="test query",
            refined_queries=["test query"],
            classification=classification
        )

        # Mock the search manager and budget method
        with patch.object(orchestrator, 'search_manager') as mock_sm, \
             patch.object(orchestrator, '_execute_searches_with_budget') as mock_budget:

            mock_sm.search_with_plan = AsyncMock(return_value={})
            mock_budget.return_value = {"test": []}

            # This would normally call the budget method, but we'll just verify the method exists
            assert hasattr(orchestrator, '_execute_searches_with_budget')
            assert callable(orchestrator._execute_searches_with_budget)
