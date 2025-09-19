from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from services.cache import cache_manager


class CostMonitor:
    """Monitors and tracks API costs."""

    def __init__(self) -> None:
        self.cost_per_call: Dict[str, float] = {
            "google": 0.005,
            "brave": 0.0,
            "exa": 0.005,
        }

    async def track_search_cost(self, api_name: str, queries_count: int) -> float:
        cost = self.cost_per_call.get(api_name, 0.0) * queries_count
        try:
            await cache_manager.track_api_cost(api_name, cost, queries_count)
        except Exception:
            pass
        return cost

    async def get_daily_costs(self, date: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        return await cache_manager.get_daily_api_costs(date)


class RetryPolicy:
    def __init__(self, max_attempts: int = 3, base_delay_sec: float = 0.5, max_delay_sec: float = 8.0) -> None:
        self.max_attempts = max_attempts
        self.base_delay_sec = base_delay_sec
        self.max_delay_sec = max_delay_sec


@dataclass
class ToolCapability:
    name: str
    cost_per_call_usd: float = 0.0
    rpm_limit: Optional[int] = None
    rpd_limit: Optional[int] = None
    typical_latency_ms: Optional[int] = None
    failure_types: List[str] = field(default_factory=list)
    healthy: bool = True
    last_health_check: Optional[datetime] = None


class ToolRegistry:
    """Registry for tool capabilities, costs, limits, and health status."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolCapability] = {}

    def register(self, capability: ToolCapability) -> None:
        self._tools[capability.name] = capability

    def get(self, name: str) -> Optional[ToolCapability]:
        return self._tools.get(name)

    def list(self) -> List[ToolCapability]:
        return list(self._tools.values())

    def set_health(self, name: str, healthy: bool) -> None:
        cap = self._tools.get(name)
        if cap:
            cap.healthy = healthy
            cap.last_health_check = datetime.now()


@dataclass
class Budget:
    max_tokens: int
    max_cost_usd: float
    max_wallclock_minutes: int


@dataclass
class PlannerCheckpoint:
    name: str
    description: str
    done: bool = False


@dataclass
class Plan:
    objective: str
    checkpoints: List[PlannerCheckpoint]
    budget: Budget
    stop_conditions: Dict[str, Any] = field(default_factory=dict)
    consumed_cost_usd: float = 0.0
    consumed_tokens: int = 0
    started_at: datetime = field(default_factory=datetime.now)

    def can_spend(self, additional_cost_usd: float, additional_tokens: int) -> bool:
        within_cost = (self.consumed_cost_usd + additional_cost_usd) <= self.budget.max_cost_usd
        within_tokens = (self.consumed_tokens + additional_tokens) <= self.budget.max_tokens
        return within_cost and within_tokens

    def spend(self, cost_usd: float, tokens: int) -> None:
        self.consumed_cost_usd += max(0.0, cost_usd)
        self.consumed_tokens += max(0, tokens)


class BudgetAwarePlanner:
    def __init__(self, registry: ToolRegistry, retry_policy: Optional[RetryPolicy] = None) -> None:
        self.registry = registry
        self.retry_policy = retry_policy or RetryPolicy()

    def select_tools(self, preferred: List[str]) -> List[str]:
        tools: List[str] = []
        for name in preferred:
            cap = self.registry.get(name)
            if cap and cap.healthy:
                tools.append(name)
        return tools

    def estimate_cost(self, tool_name: str, calls: int = 1) -> float:
        cap = self.registry.get(tool_name)
        return (cap.cost_per_call_usd * calls) if cap else 0.0

    def record_tool_spend(self, plan: Plan, tool_name: str, calls: int, tokens: int = 0) -> bool:
        cost = self.estimate_cost(tool_name, calls)
        if not plan.can_spend(cost, tokens):
            return False
        plan.spend(cost, tokens)
        return True


class SearchMetrics(TypedDict, total=False):
    total_queries: int
    total_results: int
    apis_used: List[str]
    deduplication_rate: float
    retries_attempted: int
    task_timeouts: int
    exceptions_by_api: Dict[str, int]
    api_call_counts: Dict[str, int]
    dropped_no_url: int
    dropped_invalid_shape: int
    compression_plural_used: int
    compression_singular_used: int

