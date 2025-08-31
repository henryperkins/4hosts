import pytest
from unittest.mock import Mock, AsyncMock, patch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.research_orchestrator import ResearchOrchestrator
from schemas.research import SearchResult, HostParadigm

@pytest.mark.asyncio
async def test_execute_paradigm_research_handles_none_content():
    """Test that the orchestrator properly handles results with None content."""
    
    orchestrator = ResearchOrchestrator()
    
    # Create mock results with various content scenarios
    mock_results = {
        "query1": [
            SearchResult(
                title="Valid Result",
                content="This is valid content",
                url="http://example.com/1",
                relevance_score=0.9
            ),
            SearchResult(
                title="None Content Result",
                content=None,  # This should be handled gracefully
                url="http://example.com/2",
                relevance_score=0.8
            ),
            SearchResult(
                title="Empty Content Result",
                content="",  # This should also be filtered out
                url="http://example.com/3",
                relevance_score=0.7
            ),
            SearchResult(
                title="Whitespace Content Result",
                content="   ",  # This should also be filtered out
                url="http://example.com/4",
                relevance_score=0.6
            )
        ]
    }
    
    # Mock the search execution to return our test results
    with patch.object(orchestrator, '_execute_search_batch', new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_results
        
        # Mock the answer generator to avoid actual API calls
        with patch.object(orchestrator.answer_generator, 'generate_answer', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = Mock(
                content="Generated answer",
                confidence_score=0.85,
                paradigm_alignment=0.9,
                sources_used=1
            )
            
            # Execute the research
            result = await orchestrator.execute_paradigm_research(
                query="test query",
                paradigm=HostParadigm.BERNARD,
                search_queries=["query1"],
                context={}
            )
            
            # Verify the result is successful
            assert result is not None
            assert result.content == "Generated answer"
            
            # Verify that only the valid result was passed to the answer generator
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args[1]
            assert 'search_results' in call_args
            assert len(call_args['search_results']) == 1
            assert call_args['search_results'][0].content == "This is valid content"

@pytest.mark.asyncio
async def test_execute_paradigm_research_all_none_content():
    """Test that the orchestrator handles case where all results have None content."""
    
    orchestrator = ResearchOrchestrator()
    
    # Create mock results with all None or empty content
    mock_results = {
        "query1": [
            SearchResult(
                title="None Content 1",
                content=None,
                url="http://example.com/1",
                relevance_score=0.9
            ),
            SearchResult(
                title="None Content 2",
                content=None,
                url="http://example.com/2",
                relevance_score=0.8
            )
        ]
    }
    
    # Mock the search execution
    with patch.object(orchestrator, '_execute_search_batch', new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_results
        
        # Mock the answer generator
        with patch.object(orchestrator.answer_generator, 'generate_answer', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = Mock(
                content="No valid results found",
                confidence_score=0.0,
                paradigm_alignment=0.0,
                sources_used=0
            )
            
            # Execute the research
            result = await orchestrator.execute_paradigm_research(
                query="test query",
                paradigm=HostParadigm.BERNARD,
                search_queries=["query1"],
                context={}
            )
            
            # Verify the result
            assert result is not None
            # The answer generator should be called with empty results
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args[1]
            assert 'search_results' in call_args
            assert len(call_args['search_results']) == 0