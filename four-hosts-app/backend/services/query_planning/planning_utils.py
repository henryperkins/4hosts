from __future__ import annotations

from decimal import Decimal
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from services.cache import cache_manager


class CostMonitor:
    """Monitors and tracks API costs with Decimal precision internally."""

    def __init__(self) -> None:
        self.cost_per_call: Dict[str, Decimal] = {
            "google": Decimal("0.005"),
            "brave": Decimal("0.0"),
            "exa": Decimal("0.005"),
        }

    async def track_search_cost(
        self, api_name: str, queries_count: int
    ) -> float:
        """Track cost for a given API and number of calls.

        - API name is normalized to lowercase.
        - Call counts are treated as whole numbers (floored) and
          min-clamped to 0.
        - Unknown APIs default to 0 cost.
        """
        try:
            name = str(api_name or "").strip().lower()
        except Exception:
            name = ""
        try:
            calls = int(queries_count)
        except Exception:
            calls = 0
        calls = max(0, calls)

        unit_cost = self.cost_per_call.get(name, Decimal("0"))
        cost = unit_cost * Decimal(calls)
        cost_float = float(cost)
        try:
            await cache_manager.track_api_cost(name, cost_float, calls)
        except Exception:
            # Cache tracking is best-effort; never fail cost computation
            pass
        return cost_float

    async def get_daily_costs(
        self, date: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        return await cache_manager.get_daily_api_costs(date)


class RetryPolicy:
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay_sec: float = 0.5,
        max_delay_sec: float = 8.0,
    ) -> None:
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

    def can_spend(
        self,
        additional_cost_usd: float,
        additional_tokens: int,
    ) -> bool:
        """Strict budget gating with clamping and precision-safe comparison.

        - Cost: strictly less than max_cost_usd (equality not allowed)
        - Tokens: less-than-or-equal allowed
        - Negative inputs are clamped to zero
        """
        add_cost = max(0.0, float(additional_cost_usd or 0.0))
        add_tokens = max(0, int(additional_tokens or 0))
        # Strictly less for cost to avoid edge-case equality approvals
        within_cost = (
            (self.consumed_cost_usd + add_cost) < self.budget.max_cost_usd
        )
        within_tokens = (
            (self.consumed_tokens + add_tokens) <= self.budget.max_tokens
        )
        return within_cost and within_tokens

    def spend(self, cost_usd: float, tokens: int) -> None:
        # Clamp and round to mitigate floating point accumulation
        self.consumed_cost_usd = round(
            self.consumed_cost_usd + max(0.0, float(cost_usd or 0.0)),
            6,
        )
        self.consumed_tokens += max(0, int(tokens or 0))


class BudgetAwarePlanner:
    def __init__(
        self,
        registry: ToolRegistry,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> None:
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
        """Estimate cost for a given tool; negative/float calls handled
        gracefully.
        """
        cap = self.registry.get(tool_name)
        try:
            n_calls = int(calls)
        except Exception:
            n_calls = 0
        n_calls = max(0, n_calls)
        return round((cap.cost_per_call_usd * n_calls), 6) if cap else 0.0

    def record_tool_spend(
        self,
        plan: Plan,
        tool_name: str,
        calls: int,
        tokens: int = 0,
    ) -> bool:
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

