"""
MCP Tool Usage Monitoring and Telemetry
Tracks MCP tool invocations, performance metrics, and usage patterns
"""

import time
import structlog
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from collections import defaultdict
from dataclasses import dataclass, field, asdict
import asyncio

logger = structlog.get_logger(__name__)


@dataclass
class MCPToolCall:
    """Record of a single MCP tool invocation"""
    tool_name: str
    server_name: str
    paradigm: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    success: bool = False
    error: Optional[str] = None
    research_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    result_size_bytes: Optional[int] = None

    def complete(self, success: bool = True, error: Optional[str] = None, result_size: Optional[int] = None):
        """Mark the tool call as complete"""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.success = success
        self.error = error
        self.result_size_bytes = result_size

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return asdict(self)


@dataclass
class MCPServerMetrics:
    """Metrics for a single MCP server"""
    server_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0
    tools_used: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    paradigms_used: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    recent_errors: List[str] = field(default_factory=list)

    def update(self, call: MCPToolCall):
        """Update metrics with a completed call"""
        self.total_calls += 1

        if call.success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
            if call.error:
                self.recent_errors.append(call.error)
                # Keep only last 10 errors
                self.recent_errors = self.recent_errors[-10:]

        if call.duration_ms is not None:
            self.total_duration_ms += call.duration_ms
            self.avg_duration_ms = self.total_duration_ms / self.total_calls
            self.min_duration_ms = min(self.min_duration_ms, call.duration_ms)
            self.max_duration_ms = max(self.max_duration_ms, call.duration_ms)

        self.tools_used[call.tool_name] += 1

        if call.paradigm:
            self.paradigms_used[call.paradigm] += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            "server_name": self.server_name,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.successful_calls / self.total_calls if self.total_calls > 0 else 0.0,
            "avg_duration_ms": round(self.avg_duration_ms, 2),
            "min_duration_ms": round(self.min_duration_ms, 2) if self.min_duration_ms != float('inf') else None,
            "max_duration_ms": round(self.max_duration_ms, 2),
            "tools_used": dict(self.tools_used),
            "paradigms_used": dict(self.paradigms_used),
            "recent_errors": self.recent_errors[-5:],  # Last 5 errors
        }


class MCPTelemetry:
    """
    Centralized telemetry collector for MCP tool usage.
    Tracks metrics, performance, and usage patterns across all MCP servers.
    """

    def __init__(self):
        self.enabled = True
        self.metrics: Dict[str, MCPServerMetrics] = {}
        self.active_calls: Dict[str, MCPToolCall] = {}
        self._lock = asyncio.Lock()

    def start_tool_call(
        self,
        tool_name: str,
        server_name: str,
        paradigm: Optional[str] = None,
        research_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Start tracking a tool call.

        Returns:
            Call ID for tracking
        """
        if not self.enabled:
            return ""

        call = MCPToolCall(
            tool_name=tool_name,
            server_name=server_name,
            paradigm=paradigm,
            research_id=research_id,
            parameters=parameters or {},
        )

        call_id = f"{server_name}_{tool_name}_{int(call.start_time * 1000000)}"
        self.active_calls[call_id] = call

        logger.debug(
            "MCP tool call started",
            call_id=call_id,
            tool=tool_name,
            server=server_name,
            paradigm=paradigm,
        )

        return call_id

    def end_tool_call(
        self,
        call_id: str,
        success: bool = True,
        error: Optional[str] = None,
        result_size: Optional[int] = None,
    ):
        """End tracking a tool call and update metrics"""
        if not self.enabled or not call_id:
            return

        call = self.active_calls.pop(call_id, None)
        if not call:
            logger.warning("MCP tool call not found", call_id=call_id)
            return

        call.complete(success=success, error=error, result_size=result_size)

        # Update server metrics
        if call.server_name not in self.metrics:
            self.metrics[call.server_name] = MCPServerMetrics(server_name=call.server_name)

        self.metrics[call.server_name].update(call)

        # Log completion
        logger.info(
            "MCP tool call completed",
            call_id=call_id,
            tool=call.tool_name,
            server=call.server_name,
            paradigm=call.paradigm,
            duration_ms=round(call.duration_ms, 2) if call.duration_ms else None,
            success=success,
            error=error,
        )

    def get_server_metrics(self, server_name: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific server"""
        metrics = self.metrics.get(server_name)
        return metrics.to_dict() if metrics else None

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all servers"""
        return {
            name: metrics.to_dict()
            for name, metrics in self.metrics.items()
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get overall summary across all servers"""
        total_calls = sum(m.total_calls for m in self.metrics.values())
        total_success = sum(m.successful_calls for m in self.metrics.values())
        total_failed = sum(m.failed_calls for m in self.metrics.values())

        return {
            "total_calls": total_calls,
            "successful_calls": total_success,
            "failed_calls": total_failed,
            "success_rate": total_success / total_calls if total_calls > 0 else 0.0,
            "active_servers": len(self.metrics),
            "active_calls": len(self.active_calls),
            "servers": list(self.metrics.keys()),
        }

    def get_paradigm_usage(self) -> Dict[str, int]:
        """Get aggregated paradigm usage across all servers"""
        paradigm_counts = defaultdict(int)
        for metrics in self.metrics.values():
            for paradigm, count in metrics.paradigms_used.items():
                paradigm_counts[paradigm] += count
        return dict(paradigm_counts)

    def get_tool_usage(self) -> Dict[str, int]:
        """Get aggregated tool usage across all servers"""
        tool_counts = defaultdict(int)
        for metrics in self.metrics.values():
            for tool, count in metrics.tools_used.items():
                tool_counts[tool] += count
        return dict(tool_counts)

    def reset_metrics(self):
        """Reset all metrics (for testing or periodic reset)"""
        self.metrics.clear()
        self.active_calls.clear()
        logger.info("MCP telemetry metrics reset")

    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics in a format suitable for external systems"""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": self.get_summary(),
            "servers": self.get_all_metrics(),
            "paradigm_usage": self.get_paradigm_usage(),
            "tool_usage": self.get_tool_usage(),
        }


# Global telemetry instance
mcp_telemetry = MCPTelemetry()


# Convenience functions
def track_mcp_call(
    tool_name: str,
    server_name: str,
    paradigm: Optional[str] = None,
    research_id: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> str:
    """Start tracking an MCP tool call"""
    return mcp_telemetry.start_tool_call(
        tool_name=tool_name,
        server_name=server_name,
        paradigm=paradigm,
        research_id=research_id,
        parameters=parameters,
    )


def complete_mcp_call(
    call_id: str,
    success: bool = True,
    error: Optional[str] = None,
    result_size: Optional[int] = None,
):
    """Complete tracking an MCP tool call"""
    mcp_telemetry.end_tool_call(
        call_id=call_id,
        success=success,
        error=error,
        result_size=result_size,
    )