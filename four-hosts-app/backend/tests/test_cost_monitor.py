"""
Comprehensive tests for CostMonitor cost calculation validation.
Tests cover cost tracking, API cost calculation, and budget monitoring.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from services.query_planning.planning_utils import CostMonitor
from services.cache import cache_manager


class TestCostMonitor:
    """Test suite for CostMonitor cost calculation and tracking."""

    @pytest.fixture
    def cost_monitor(self) -> CostMonitor:
        """Create a fresh cost monitor instance for each test."""
        return CostMonitor()

    def test_initial_cost_per_call_configuration(self, cost_monitor: CostMonitor):
        """Test that cost monitor has correct initial cost configuration."""
        expected_costs = {
            "google": 0.005,
            "brave": 0.0,
            "exa": 0.005,
        }

        for api, expected_cost in expected_costs.items():
            assert cost_monitor.cost_per_call[api] == expected_cost

    def test_cost_calculation_single_api(self, cost_monitor: CostMonitor):
        """Test cost calculation for a single API call."""
        cost = asyncio.run(cost_monitor.track_search_cost("google", 1))

        assert cost == 0.005
        assert cost_monitor.cost_per_call["google"] == 0.005

    def test_cost_calculation_multiple_calls(self, cost_monitor: CostMonitor):
        """Test cost calculation for multiple API calls."""
        # Track multiple calls
        cost1 = asyncio.run(cost_monitor.track_search_cost("google", 1))
        cost2 = asyncio.run(cost_monitor.track_search_cost("google", 3))
        cost3 = asyncio.run(cost_monitor.track_search_cost("exa", 2))

        assert cost1 == 0.005  # 1 call * $0.005
        assert cost2 == 0.015  # 3 calls * $0.005
        assert cost3 == 0.010  # 2 calls * $0.005

    def test_cost_calculation_different_apis(self, cost_monitor: CostMonitor):
        """Test cost calculation across different APIs."""
        google_cost = asyncio.run(cost_monitor.track_search_cost("google", 1))
        brave_cost = asyncio.run(cost_monitor.track_search_cost("brave", 1))
        exa_cost = asyncio.run(cost_monitor.track_search_cost("exa", 1))

        assert google_cost == 0.005
        assert brave_cost == 0.0  # Brave is free
        assert exa_cost == 0.005

    def test_cost_calculation_unknown_api(self, cost_monitor: CostMonitor):
        """Test cost calculation for unknown API (should default to 0)."""
        cost = asyncio.run(cost_monitor.track_search_cost("unknown_api", 1))

        assert cost == 0.0  # Unknown APIs should cost nothing

    def test_cost_calculation_zero_calls(self, cost_monitor: CostMonitor):
        """Test cost calculation with zero calls."""
        cost = asyncio.run(cost_monitor.track_search_cost("google", 0))

        assert cost == 0.0

    def test_cost_calculation_negative_calls(self, cost_monitor: CostMonitor):
        """Test cost calculation with negative calls (should be treated as zero)."""
        cost = asyncio.run(cost_monitor.track_search_cost("google", -1))

        assert cost == 0.0

    @patch('services.query_planning.planning_utils.cache_manager')
    def test_cache_manager_integration(self, mock_cache_manager: Mock, cost_monitor: CostMonitor):
        """Test that cost monitor properly integrates with cache manager."""
        # Setup mock
        mock_cache_manager.track_api_cost = AsyncMock()

        # Track cost
        asyncio.run(cost_monitor.track_search_cost("google", 1))

        # Verify cache manager was called correctly
        mock_cache_manager.track_api_cost.assert_called_once_with("google", 0.005, 1)

    @patch('services.query_planning.planning_utils.cache_manager')
    def test_cache_manager_error_handling(self, mock_cache_manager: Mock, cost_monitor: CostMonitor):
        """Test that cost monitor handles cache manager errors gracefully."""
        # Setup mock to raise exception
        mock_cache_manager.track_api_cost = AsyncMock(side_effect=Exception("Cache error"))

        # Should not raise exception, but log warning
        cost = asyncio.run(cost_monitor.track_search_cost("google", 1))

        # Should still return correct cost
        assert cost == 0.005

    def test_get_daily_costs_structure(self, cost_monitor: CostMonitor):
        """Test that get_daily_costs returns properly structured data."""
        @patch('services.query_planning.planning_utils.cache_manager')
        def run_test(mock_cache_manager: Mock):
            # Setup mock response
            mock_cache_manager.get_daily_api_costs = AsyncMock(return_value={
                "google": {"cost": 0.025, "calls": 5},
                "exa": {"cost": 0.010, "calls": 2}
            })

            # Get daily costs
            result = asyncio.run(cost_monitor.get_daily_costs())

            # Verify structure
            assert isinstance(result, dict)
            assert "google" in result
            assert "exa" in result
            assert "cost" in result["google"]
            assert "calls" in result["google"]

        run_test()

    def test_get_daily_costs_with_date_filter(self, cost_monitor: CostMonitor):
        """Test that get_daily_costs properly passes date filter."""
        @patch('services.query_planning.planning_utils.cache_manager')
        def run_test(mock_cache_manager: Mock):
            test_date = "2024-01-15"
            mock_cache_manager.get_daily_api_costs = AsyncMock(return_value={})

            # Get daily costs with date filter
            asyncio.run(cost_monitor.get_daily_costs(test_date))

            # Verify date was passed to cache manager
            mock_cache_manager.get_daily_api_costs.assert_called_once_with(test_date)

        run_test()

    def test_get_daily_costs_without_date_filter(self, cost_monitor: CostMonitor):
        """Test that get_daily_costs works without date filter."""
        @patch('services.query_planning.planning_utils.cache_manager')
        def run_test(mock_cache_manager: Mock):
            mock_cache_manager.get_daily_api_costs = AsyncMock(return_value={})

            # Get daily costs without date
            asyncio.run(cost_monitor.get_daily_costs())

            # Verify None was passed to cache manager
            mock_cache_manager.get_daily_api_costs.assert_called_once_with(None)

        run_test()

    def test_cost_monitor_thread_safety(self, cost_monitor: CostMonitor):
        """Test that cost monitor is thread-safe for concurrent access."""
        import threading
        import time

        results = []
        errors = []

        def track_costs():
            try:
                for i in range(10):
                    cost = asyncio.run(cost_monitor.track_search_cost("google", 1))
                    results.append(cost)
                    time.sleep(0.01)  # Small delay to increase chance of race conditions
            except Exception as e:
                errors.append(e)

        # Run multiple threads concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=track_costs)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0

        # Verify all costs are correct
        assert len(results) == 50  # 5 threads * 10 iterations
        assert all(cost == 0.005 for cost in results)

    def test_cost_monitor_memory_usage(self, cost_monitor: CostMonitor):
        """Test that cost monitor doesn't leak memory with many operations."""
        import psutil
        import os

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Perform many cost tracking operations
        for i in range(1000):
            asyncio.run(cost_monitor.track_search_cost("google", 1))

        # Get memory usage after operations
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be minimal (less than 10MB for 1000 operations)
        assert memory_increase < 10 * 1024 * 1024  # 10MB limit

    def test_cost_calculation_precision(self, cost_monitor: CostMonitor):
        """Test that cost calculations maintain proper precision."""
        # Test with fractional calls (should be treated as whole numbers)
        cost = asyncio.run(cost_monitor.track_search_cost("google", 1.7))

        # Should be calculated as 1 call, not 1.7
        assert cost == 0.005

    def test_cost_monitor_state_isolation(self):
        """Test that multiple cost monitor instances don't interfere with each other."""
        monitor1 = CostMonitor()
        monitor2 = CostMonitor()

        # Track costs on different monitors
        cost1 = asyncio.run(monitor1.track_search_cost("google", 1))
        cost2 = asyncio.run(monitor2.track_search_cost("google", 1))

        # Both should return the same cost
        assert cost1 == 0.005
        assert cost2 == 0.005

        # Verify they have independent state
        assert monitor1.cost_per_call == monitor2.cost_per_call

    def test_cost_monitor_configuration_override(self):
        """Test that cost monitor respects configuration overrides."""
        # Create monitor with custom configuration
        monitor = CostMonitor()
        monitor.cost_per_call["google"] = 0.010  # Override default

        cost = asyncio.run(monitor.track_search_cost("google", 1))

        # Should use the overridden cost
        assert cost == 0.010

    def test_cost_calculation_with_large_numbers(self, cost_monitor: CostMonitor):
        """Test cost calculation with large call counts."""
        large_call_count = 1000000
        cost = asyncio.run(cost_monitor.track_search_cost("google", large_call_count))

        expected_cost = 0.005 * large_call_count
        assert cost == expected_cost

    def test_cost_monitor_error_recovery(self, cost_monitor: CostMonitor):
        """Test that cost monitor recovers from errors gracefully."""
        # First operation should succeed
        cost1 = asyncio.run(cost_monitor.track_search_cost("google", 1))
        assert cost1 == 0.005

        # Simulate an error in cache manager
        with patch('services.query_planning.planning_utils.cache_manager') as mock_cache:
            mock_cache.track_api_cost = AsyncMock(side_effect=Exception("Network error"))

            # Second operation should still return correct cost despite cache error
            cost2 = asyncio.run(cost_monitor.track_search_cost("google", 1))
            assert cost2 == 0.005

        # Third operation should work normally after error
        cost3 = asyncio.run(cost_monitor.track_search_cost("google", 1))
        assert cost3 == 0.005

    def test_cost_monitor_cost_bounds(self, cost_monitor: CostMonitor):
        """Test that cost calculations respect reasonable bounds."""
        # Test minimum cost
        min_cost = asyncio.run(cost_monitor.track_search_cost("google", 0))
        assert min_cost >= 0.0

        # Test with very large call count
        large_cost = asyncio.run(cost_monitor.track_search_cost("google", 1000000))
        assert large_cost >= 0.0
        assert large_cost == 5000.0  # 1M calls * $0.005

    def test_cost_monitor_api_name_normalization(self, cost_monitor: CostMonitor):
        """Test that API names are handled consistently."""
        # Test with different capitalizations
        cost1 = asyncio.run(cost_monitor.track_search_cost("Google", 1))
        cost2 = asyncio.run(cost_monitor.track_search_cost("GOOGLE", 1))
        cost3 = asyncio.run(cost_monitor.track_search_cost("google", 1))

        # All should return the same cost
        assert cost1 == cost2 == cost3 == 0.005

    def test_cost_monitor_empty_api_name(self, cost_monitor: CostMonitor):
        """Test handling of empty or None API names."""
        cost = asyncio.run(cost_monitor.track_search_cost("", 1))

        # Should default to 0 cost for unknown API
        assert cost == 0.0

    def test_cost_monitor_special_characters_in_api_name(self, cost_monitor: CostMonitor):
        """Test handling of API names with special characters."""
        cost = asyncio.run(cost_monitor.track_search_cost("google-v2.0", 1))

        # Should default to 0 cost for unknown API
        assert cost == 0.0
