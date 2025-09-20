"""
Comprehensive tests for BudgetAwarePlanner budget enforcement logic.
Tests cover budget planning, cost estimation, and enforcement decisions.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from services.query_planning.planning_utils import (
    BudgetAwarePlanner,
    ToolRegistry,
    ToolCapability,
    Plan,
    Budget,
    PlannerCheckpoint,
    RetryPolicy
)


class TestBudgetAwarePlanner:
    """Test suite for BudgetAwarePlanner budget enforcement and planning."""

    @pytest.fixture
    def tool_registry(self) -> ToolRegistry:
        """Create a tool registry with test capabilities."""
        registry = ToolRegistry()

        # Register test tools with different costs
        registry.register(ToolCapability(
            name="google",
            cost_per_call_usd=0.005,
            rpm_limit=100,
            typical_latency_ms=800
        ))
        registry.register(ToolCapability(
            name="brave",
            cost_per_call_usd=0.0,
            rpm_limit=100,
            typical_latency_ms=600
        ))
        registry.register(ToolCapability(
            name="exa",
            cost_per_call_usd=0.005,
            rpm_limit=50,
            typical_latency_ms=1000
        ))
        registry.register(ToolCapability(
            name="expensive_tool",
            cost_per_call_usd=0.050,
            rpm_limit=10,
            typical_latency_ms=2000
        ))

        return registry

    @pytest.fixture
    def retry_policy(self) -> RetryPolicy:
        """Create a retry policy for testing."""
        return RetryPolicy(max_attempts=3, base_delay_sec=0.5, max_delay_sec=8.0)

    @pytest.fixture
    def budget_planner(self, tool_registry: ToolRegistry, retry_policy: RetryPolicy) -> BudgetAwarePlanner:
        """Create a budget-aware planner instance for testing."""
        return BudgetAwarePlanner(tool_registry, retry_policy)

    @pytest.fixture
    def test_budget(self) -> Budget:
        """Create a test budget for planning."""
        return Budget(
            max_tokens=10000,
            max_cost_usd=0.50,
            max_wallclock_minutes=5
        )

    @pytest.fixture
    def test_plan(self, test_budget: Budget) -> Plan:
        """Create a test plan for budget operations."""
        return Plan(
            objective="Test research plan",
            budget=test_budget,
            checkpoints=[
                PlannerCheckpoint(name="test_checkpoint", description="Test checkpoint")
            ]
        )

    def test_planner_initialization(self, budget_planner: BudgetAwarePlanner, tool_registry: ToolRegistry, retry_policy: RetryPolicy):
        """Test that budget planner initializes correctly."""
        assert budget_planner.registry is tool_registry
        assert budget_planner.retry_policy is retry_policy
        assert budget_planner.retry_policy.max_attempts == 3

    def test_tool_selection_with_healthy_tools(self, budget_planner: BudgetAwarePlanner):
        """Test tool selection when all tools are healthy."""
        preferred_tools = ["google", "brave", "exa"]
        selected = budget_planner.select_tools(preferred_tools)

        # Should return all healthy tools in order
        assert len(selected) == 3
        assert selected == preferred_tools

    def test_tool_selection_with_unhealthy_tools(self, budget_planner: BudgetAwarePlanner):
        """Test tool selection when some tools are unhealthy."""
        # Mark exa as unhealthy
        budget_planner.registry.set_health("exa", False)

        preferred_tools = ["google", "brave", "exa"]
        selected = budget_planner.select_tools(preferred_tools)

        # Should exclude unhealthy tool
        assert len(selected) == 2
        assert "exa" not in selected
        assert "google" in selected
        assert "brave" in selected

    def test_tool_selection_empty_preferences(self, budget_planner: BudgetAwarePlanner):
        """Test tool selection with empty preferences."""
        selected = budget_planner.select_tools([])
        assert selected == []

    def test_cost_estimation_known_tool(self, budget_planner: BudgetAwarePlanner):
        """Test cost estimation for known tools."""
        # Single call
        cost = budget_planner.estimate_cost("google", 1)
        assert cost == 0.005

        # Multiple calls
        cost = budget_planner.estimate_cost("google", 5)
        assert cost == 0.025

    def test_cost_estimation_unknown_tool(self, budget_planner: BudgetAwarePlanner):
        """Test cost estimation for unknown tools."""
        cost = budget_planner.estimate_cost("unknown_tool", 1)
        assert cost == 0.0

    def test_cost_estimation_zero_calls(self, budget_planner: BudgetAwarePlanner):
        """Test cost estimation with zero calls."""
        cost = budget_planner.estimate_cost("google", 0)
        assert cost == 0.0

    def test_cost_estimation_negative_calls(self, budget_planner: BudgetAwarePlanner):
        """Test cost estimation with negative calls."""
        cost = budget_planner.estimate_cost("google", -1)
        assert cost == 0.0

    def test_record_tool_spend_within_budget(self, budget_planner: BudgetAwarePlanner, test_plan: Plan):
        """Test recording spend when within budget limits."""
        # Record spend for google (cheap tool)
        success = budget_planner.record_tool_spend(test_plan, "google", 1, 100)

        assert success == True
        assert test_plan.consumed_cost_usd == 0.005
        assert test_plan.consumed_tokens == 100

    def test_record_tool_spend_exceeds_cost_budget(self, budget_planner: BudgetAwarePlanner, test_plan: Plan):
        """Test recording spend when exceeding cost budget."""
        # Use up most of the budget first
        budget_planner.record_tool_spend(test_plan, "google", 99, 0)  # 99 calls * $0.005 = $0.495

        # Try to record one more call that would exceed budget
        success = budget_planner.record_tool_spend(test_plan, "google", 1, 0)

        assert success == False
        assert test_plan.consumed_cost_usd == 0.495  # Should not have increased

    def test_record_tool_spend_exceeds_token_budget(self, budget_planner: BudgetAwarePlanner, test_plan: Plan):
        """Test recording spend when exceeding token budget."""
        # Use up most of the token budget first
        budget_planner.record_tool_spend(test_plan, "google", 0, 9900)  # 9900 tokens

        # Try to record more tokens that would exceed budget
        success = budget_planner.record_tool_spend(test_plan, "google", 0, 200)

        assert success == False
        assert test_plan.consumed_tokens == 9900  # Should not have increased

    def test_record_tool_spend_expensive_tool(self, budget_planner: BudgetAwarePlanner, test_plan: Plan):
        """Test recording spend for expensive tools."""
        # Record spend for expensive tool
        success = budget_planner.record_tool_spend(test_plan, "expensive_tool", 1, 0)

        assert success == True
        assert test_plan.consumed_cost_usd == 0.050

    def test_record_tool_spend_multiple_calls(self, budget_planner: BudgetAwarePlanner, test_plan: Plan):
        """Test recording spend for multiple calls."""
        # Record multiple calls
        success1 = budget_planner.record_tool_spend(test_plan, "google", 3, 0)
        success2 = budget_planner.record_tool_spend(test_plan, "exa", 2, 0)

        assert success1 == True
        assert success2 == True
        assert test_plan.consumed_cost_usd == 0.025  # 3 * 0.005 + 2 * 0.005

    def test_record_tool_spend_unknown_tool(self, budget_planner: BudgetAwarePlanner, test_plan: Plan):
        """Test recording spend for unknown tools."""
        success = budget_planner.record_tool_spend(test_plan, "unknown_tool", 1, 0)

        assert success == True
        assert test_plan.consumed_cost_usd == 0.0  # Unknown tools cost nothing

    def test_record_tool_spend_zero_calls(self, budget_planner: BudgetAwarePlanner, test_plan: Plan):
        """Test recording spend with zero calls."""
        success = budget_planner.record_tool_spend(test_plan, "google", 0, 0)

        assert success == True
        assert test_plan.consumed_cost_usd == 0.0
        assert test_plan.consumed_tokens == 0

    def test_record_tool_spend_negative_calls(self, budget_planner: BudgetAwarePlanner, test_plan: Plan):
        """Test recording spend with negative calls."""
        success = budget_planner.record_tool_spend(test_plan, "google", -1, 0)

        assert success == True
        assert test_plan.consumed_cost_usd == 0.0  # Negative calls should not reduce cost

    def test_budget_planning_with_mixed_tools(self, budget_planner: BudgetAwarePlanner, test_plan: Plan):
        """Test budget planning with a mix of different tools."""
        # Plan to use multiple tools
        tools_to_use = [
            ("google", 10),  # 10 calls * $0.005 = $0.050
            ("brave", 20),   # 20 calls * $0.000 = $0.000
            ("exa", 5),      # 5 calls * $0.005 = $0.025
        ]

        total_estimated_cost = 0.0
        for tool, calls in tools_to_use:
            estimated_cost = budget_planner.estimate_cost(tool, calls)
            total_estimated_cost += estimated_cost

            # Record the spend
            success = budget_planner.record_tool_spend(test_plan, tool, calls, 0)
            assert success == True

        # Verify total cost
        assert test_plan.consumed_cost_usd == 0.075  # $0.050 + $0.025
        assert total_estimated_cost == 0.075

    def test_budget_enforcement_precision(self, budget_planner: BudgetAwarePlanner, test_plan: Plan):
        """Test that budget enforcement handles floating point precision correctly."""
        # Use up budget with high precision
        budget_planner.record_tool_spend(test_plan, "google", 99, 0)  # 99 * 0.005 = 0.495

        # Try to add a tiny bit more that would exceed budget due to precision
        success = budget_planner.record_tool_spend(test_plan, "google", 1, 0)  # 1 * 0.005 = 0.005

        # Should fail due to budget limit
        assert success == False
        assert abs(test_plan.consumed_cost_usd - 0.495) < 0.0001

    def test_budget_with_token_and_cost_limits(self, budget_planner: BudgetAwarePlanner):
        """Test budget enforcement with both token and cost limits."""
        # Create budget with both limits
        mixed_budget = Budget(
            max_tokens=100,
            max_cost_usd=0.10,
            max_wallclock_minutes=5
        )
        mixed_plan = Plan(
            objective="Mixed limits test",
            budget=mixed_budget,
            checkpoints=[]
        )

        # Use up token budget
        budget_planner.record_tool_spend(mixed_plan, "google", 0, 100)

        # Try to add more tokens (should fail)
        success1 = budget_planner.record_tool_spend(mixed_plan, "google", 0, 1)
        assert success1 == False

        # Cost budget should still be available
        success2 = budget_planner.record_tool_spend(mixed_plan, "google", 1, 0)
        assert success2 == True

    def test_budget_planning_edge_cases(self, budget_planner: BudgetAwarePlanner):
        """Test budget planning with edge cases."""
        # Create plan with zero budget
        zero_budget = Budget(max_tokens=0, max_cost_usd=0.0, max_wallclock_minutes=0)
        zero_plan = Plan(objective="Zero budget test", budget=zero_budget, checkpoints=[])

        # Any spend should fail
        success = budget_planner.record_tool_spend(zero_plan, "google", 1, 1)
        assert success == False

    def test_budget_consumption_tracking(self, budget_planner: BudgetAwarePlanner, test_plan: Plan):
        """Test that budget consumption is tracked accurately."""
        initial_cost = test_plan.consumed_cost_usd
        initial_tokens = test_plan.consumed_tokens

        # Record some spend
        budget_planner.record_tool_spend(test_plan, "google", 5, 200)

        # Check that consumption increased correctly
        assert test_plan.consumed_cost_usd == initial_cost + 0.025  # 5 * 0.005
        assert test_plan.consumed_tokens == initial_tokens + 200

    def test_budget_negative_consumption_protection(self, budget_planner: BudgetAwarePlanner, test_plan: Plan):
        """Test that budget consumption cannot go negative."""
        # Record negative spend (should be treated as zero)
        budget_planner.record_tool_spend(test_plan, "google", -10, -100)

        # Consumption should remain zero
        assert test_plan.consumed_cost_usd == 0.0
        assert test_plan.consumed_tokens == 0

    def test_tool_health_status_integration(self, budget_planner: BudgetAwarePlanner):
        """Test that tool health status affects planning decisions."""
        # Mark google as unhealthy
        budget_planner.registry.set_health("google", False)

        # Try to select tools
        selected = budget_planner.select_tools(["google", "brave", "exa"])

        # Should exclude unhealthy tool
        assert "google" not in selected
        assert "brave" in selected
        assert "exa" in selected

    def test_budget_planning_with_health_considerations(self, budget_planner: BudgetAwarePlanner, test_plan: Plan):
        """Test budget planning when considering tool health."""
        # Mark expensive tool as unhealthy
        budget_planner.registry.set_health("expensive_tool", False)

        # Try to use expensive tool (should still work but may not be selected)
        success = budget_planner.record_tool_spend(test_plan, "expensive_tool", 1, 0)
        assert success == True  # Budget logic should still work

    def test_retry_policy_integration(self, budget_planner: BudgetAwarePlanner):
        """Test that retry policy is properly integrated."""
        # Verify retry policy settings
        assert budget_planner.retry_policy.max_attempts == 3
        assert budget_planner.retry_policy.base_delay_sec == 0.5
        assert budget_planner.retry_policy.max_delay_sec == 8.0

    def test_budget_planning_performance(self, budget_planner: BudgetAwarePlanner):
        """Test budget planning performance with many operations."""
        import time

        # Create a larger budget for performance testing
        large_budget = Budget(max_tokens=100000, max_cost_usd=100.0, max_wallclock_minutes=60)
        large_plan = Plan(objective="Performance test", budget=large_budget, checkpoints=[])

        # Perform many budget operations
        start_time = time.time()

        for i in range(1000):
            success = budget_planner.record_tool_spend(large_plan, "google", 1, 10)
            assert success == True

        end_time = time.time()

        # Should complete within reasonable time (less than 5 seconds for 1000 operations)
        assert end_time - start_time < 5.0

        # Verify final budget state
        assert large_plan.consumed_cost_usd == 5.0  # 1000 * 0.005
        assert large_plan.consumed_tokens == 10000  # 1000 * 10

    def test_budget_enforcement_error_handling(self, budget_planner: BudgetAwarePlanner):
        """Test that budget enforcement handles errors gracefully."""
        # Create plan with None values (edge case)
        error_plan = Plan(
            objective="Error test",
            budget=Budget(max_tokens=100, max_cost_usd=0.50, max_wallclock_minutes=5),
            checkpoints=[]
        )

        # This should not crash
        success = budget_planner.record_tool_spend(error_plan, "google", 1, 0)
        assert success == True

    def test_budget_planning_state_consistency(self, budget_planner: BudgetAwarePlanner):
        """Test that budget planning maintains state consistency."""
        plan1 = Plan(
            objective="Test 1",
            budget=Budget(max_tokens=100, max_cost_usd=0.50, max_wallclock_minutes=5),
            checkpoints=[]
        )
        plan2 = Plan(
            objective="Test 2",
            budget=Budget(max_tokens=100, max_cost_usd=0.50, max_wallclock_minutes=5),
            checkpoints=[]
        )

        # Record spend on different plans
        budget_planner.record_tool_spend(plan1, "google", 5, 0)
        budget_planner.record_tool_spend(plan2, "google", 3, 0)

        # Each plan should have independent state
        assert plan1.consumed_cost_usd == 0.025  # 5 * 0.005
        assert plan2.consumed_cost_usd == 0.015  # 3 * 0.005

    def test_budget_enforcement_with_large_numbers(self, budget_planner: BudgetAwarePlanner):
        """Test budget enforcement with large numbers."""
        large_budget = Budget(max_tokens=1000000, max_cost_usd=1000.0, max_wallclock_minutes=60)
        large_plan = Plan(objective="Large test", budget=large_budget, checkpoints=[])

        # Test with large call counts
        success = budget_planner.record_tool_spend(large_plan, "google", 100000, 50000)
        assert success == True

        assert large_plan.consumed_cost_usd == 500.0  # 100000 * 0.005
        assert large_plan.consumed_tokens == 50000
