"""
Cost attribution utilities for search operations
"""

import structlog
from typing import Dict, List, Any, Optional

logger = structlog.get_logger(__name__)


async def attribute_search_costs(
    results: List[Any],
    cost_monitor: Any,
    cost_accumulator: Optional[Dict[str, float]] = None,
    search_manager: Optional[Any] = None
) -> None:
    """
    Attribute costs to search providers based on observed results.

    Args:
        results: List of search results
        cost_monitor: CostMonitor instance for tracking costs
        cost_accumulator: Dictionary to accumulate costs by provider
        search_manager: Search manager instance (optional, for fallback provider list)
    """
    if cost_accumulator is None:
        return

    try:
        # Extract unique providers from results
        seen = {
            (str(getattr(r, "source_api", None) or getattr(r, "source", "")).strip().lower())
            for r in results
        }
        seen.discard("")

        # If no providers found in results, fall back to configured list
        if not seen and search_manager:
            mgr = search_manager
            seen = set(getattr(mgr, "apis", {}).keys()) if mgr else set()

        # Track cost for each provider
        for name in seen:
            cost = await cost_monitor.track_search_cost(name, 1)
            cost_accumulator[name] = float(cost_accumulator.get(name, 0.0)) + float(cost)

    except Exception as e:
        logger.debug(f"Cost attribution failed: {e}", exc_info=True)