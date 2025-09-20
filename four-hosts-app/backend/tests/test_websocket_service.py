"""
Tests for WebSocket service counter synchronization.
Focuses on the report_search_completed method enhancements.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from services.websocket_service import ResearchProgressTracker, WSMessage


class TestResearchProgressTracker:
    """Test the ResearchProgressTracker class, particularly counter synchronization."""

    @pytest.fixture
    def mock_connection_manager(self):
        """Create a mock connection manager."""
        manager = MagicMock()
        manager.broadcast_to_research = AsyncMock()
        return manager

    @pytest.fixture
    def tracker(self, mock_connection_manager):
        """Create a ResearchProgressTracker instance with mocked dependencies."""
        tracker = ResearchProgressTracker(mock_connection_manager)
        return tracker

    @pytest.mark.asyncio
    async def test_report_search_completed_with_existing_progress(self, tracker):
        """Test that search completion correctly updates counters when progress exists."""
        research_id = "test-123"
        query = "test query"
        results_count = 10

        # Set up initial progress state
        tracker.research_progress[research_id] = {
            "searches_completed": 2,
            "total_searches": 5,
            "status": "in_progress"
        }

        # Call the method
        await tracker.report_search_completed(research_id, query, results_count)

        # Verify broadcast was called with correct counters
        broadcast_call = tracker.connection_manager.broadcast_to_research.call_args
        assert broadcast_call is not None

        ws_message = broadcast_call[0][1]  # Second argument is the WSMessage
        assert ws_message.type.value == "search.completed"
        assert ws_message.data["searches_completed"] == 3  # Incremented from 2
        assert ws_message.data["total_searches"] == 5  # Remains same
        assert ws_message.data["results_count"] == results_count
        assert ws_message.data["query"] == query

        # Verify progress was updated
        assert tracker.research_progress[research_id]["searches_completed"] == 3

    @pytest.mark.asyncio
    async def test_report_search_completed_adjusts_total_if_needed(self, tracker):
        """Test that total_searches is adjusted if completed exceeds it."""
        research_id = "test-456"

        # Set up progress where we're at the limit
        tracker.research_progress[research_id] = {
            "searches_completed": 5,
            "total_searches": 5,
            "status": "in_progress"
        }

        await tracker.report_search_completed(research_id, "query", 15)

        # Verify total was adjusted to match completed
        broadcast_call = tracker.connection_manager.broadcast_to_research.call_args
        ws_message = broadcast_call[0][1]
        assert ws_message.data["searches_completed"] == 6
        assert ws_message.data["total_searches"] == 6  # Adjusted up

    @pytest.mark.asyncio
    async def test_report_search_completed_no_existing_progress(self, tracker):
        """Test behavior when no progress exists for the research."""
        research_id = "test-789"

        await tracker.report_search_completed(research_id, "new query", 20)

        # Should broadcast with None values for counters
        broadcast_call = tracker.connection_manager.broadcast_to_research.call_args
        ws_message = broadcast_call[0][1]
        assert ws_message.data["searches_completed"] is None
        assert ws_message.data["total_searches"] is None
        assert ws_message.data["results_count"] == 20

    @pytest.mark.asyncio
    async def test_report_search_completed_handles_update_failure(self, tracker):
        """Test that broadcast still happens even if update_progress fails."""
        research_id = "test-fail"

        tracker.research_progress[research_id] = {
            "searches_completed": 1,
            "total_searches": 3
        }

        # Mock update_progress to fail
        with patch.object(tracker, 'update_progress', side_effect=Exception("DB Error")):
            # Should not raise exception
            await tracker.report_search_completed(research_id, "query", 5)

        # Verify broadcast was still called
        assert tracker.connection_manager.broadcast_to_research.called

    @pytest.mark.asyncio
    async def test_report_search_completed_truncates_long_query(self, tracker):
        """Test that long queries are truncated in the broadcast."""
        research_id = "test-long"
        long_query = "x" * 100  # 100 character query

        await tracker.report_search_completed(research_id, long_query, 5)

        broadcast_call = tracker.connection_manager.broadcast_to_research.call_args
        ws_message = broadcast_call[0][1]

        # Should be truncated to 50 chars + "..."
        assert len(ws_message.data["query"]) == 53
        assert ws_message.data["query"].endswith("...")

    @pytest.mark.asyncio
    async def test_counter_synchronization_consistency(self, tracker):
        """Test that counter values remain consistent across multiple searches."""
        research_id = "test-consistency"

        # Initialize with some progress
        tracker.research_progress[research_id] = {
            "searches_completed": 0,
            "total_searches": 10
        }

        # Simulate multiple search completions
        for i in range(5):
            await tracker.report_search_completed(research_id, f"query-{i}", i * 10)

        # Final state should show 5 completed
        assert tracker.research_progress[research_id]["searches_completed"] == 5
        assert tracker.research_progress[research_id]["total_searches"] == 10

        # Last broadcast should have correct values
        last_call = tracker.connection_manager.broadcast_to_research.call_args
        ws_message = last_call[0][1]
        assert ws_message.data["searches_completed"] == 5
        assert ws_message.data["total_searches"] == 10

    @pytest.mark.asyncio
    async def test_report_search_completed_with_invalid_progress_data(self, tracker):
        """Test handling of invalid progress data gracefully."""
        research_id = "test-invalid"

        # Set up invalid progress data (non-integer values)
        tracker.research_progress[research_id] = {
            "searches_completed": "not-a-number",
            "total_searches": None
        }

        # Should handle gracefully and broadcast with None values
        await tracker.report_search_completed(research_id, "query", 10)

        broadcast_call = tracker.connection_manager.broadcast_to_research.call_args
        ws_message = broadcast_call[0][1]
        assert ws_message.data["searches_completed"] is None
        assert ws_message.data["total_searches"] is None

    @pytest.mark.asyncio
    async def test_timestamp_format_in_broadcast(self, tracker):
        """Test that timestamps are properly formatted in ISO format."""
        research_id = "test-timestamp"

        await tracker.report_search_completed(research_id, "query", 10)

        broadcast_call = tracker.connection_manager.broadcast_to_research.call_args
        ws_message = broadcast_call[0][1]

        # Verify timestamp is present and in ISO format
        timestamp = ws_message.data["timestamp"]
        assert timestamp is not None
        # Should parse without error
        parsed = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        assert parsed.tzinfo is not None  # Should have timezone info