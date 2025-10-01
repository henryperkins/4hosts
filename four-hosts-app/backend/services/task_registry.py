"""
Global Task Registry for Four Hosts Research Application
Manages all background tasks and ensures proper cleanup during shutdown
"""

import asyncio
import structlog
from typing import Dict, Set, Optional, Any, Callable
from datetime import datetime
from enum import Enum
import weakref
from contextlib import asynccontextmanager

from logging_config import configure_logging

configure_logging()
logger = structlog.get_logger(__name__)


class TaskPriority(str, Enum):
    """Task priority levels for shutdown ordering"""
    CRITICAL = "critical"  # Must complete before shutdown
    HIGH = "high"         # Should complete if possible
    NORMAL = "normal"     # Can be cancelled
    LOW = "low"          # Can be cancelled immediately


class TaskInfo:
    """Information about a registered task"""
    def __init__(self, task: asyncio.Task, name: str, priority: TaskPriority = TaskPriority.NORMAL):
        self.task = task
        self.name = name
        self.priority = priority
        self.created_at = datetime.utcnow()
        self.cancelled = False
        
    def __repr__(self):
        return f"TaskInfo(name={self.name}, priority={self.priority}, done={self.task.done()})"


class TaskRegistry:
    """Global registry for all background tasks"""
    
    def __init__(self):
        self._tasks: Dict[str, TaskInfo] = {}
        self._task_counter = 0
        self._shutdown_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._cleanup_tasks: Set[asyncio.Task[Any]] = set()
        
    async def register_task(
        self, 
        task: asyncio.Task, 
        name: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> str:
        """Register a task with the registry"""
        async with self._lock:
            if name is None:
                name = f"task_{self._task_counter}"
                self._task_counter += 1
            
            task_id = f"{name}_{id(task)}"
            self._tasks[task_id] = TaskInfo(task, name, priority)

            def _on_task_done(_: asyncio.Task) -> None:
                self._schedule_cleanup(task_id)

            task.add_done_callback(_on_task_done)
            
            logger.debug(f"Registered task: {task_id} (priority: {priority})")
            return task_id

    def _schedule_cleanup(self, task_id: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("task_registry.cleanup_loop_missing", task_id=task_id)
            return

        async def _cleanup() -> None:
            await self._remove_task(task_id)

        cleanup_task = loop.create_task(
            _cleanup(),
            name=f"task_registry_cleanup:{task_id}",
        )
        self._cleanup_tasks.add(cleanup_task)

        def _cleanup_done(done: asyncio.Task[Any]) -> None:
            self._cleanup_tasks.discard(done)
            try:
                exc = done.exception()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.debug(
                    "task_registry.cleanup_inspection_failed",
                    task_id=task_id,
                    error=str(exc),
                )
                return
            if exc is not None:
                logger.warning(
                    "task_registry.cleanup_failed",
                    task_id=task_id,
                    error=str(exc),
                )

        cleanup_task.add_done_callback(_cleanup_done)

    async def _remove_task(self, task_id: str):
        """Remove a completed task from the registry"""
        async with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                logger.debug(f"Removed completed task: {task_id}")
    
    @asynccontextmanager
    async def create_task(
        self,
        coro,
        name: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL
    ):
        """Context manager for creating and registering a task"""
        task = asyncio.create_task(coro)
        task_id = await self.register_task(task, name, priority)
        try:
            yield task
        finally:
            # Task will be auto-removed when done
            pass
    
    async def cancel_task(self, task_id: str, msg: Optional[str] = None) -> bool:
        """Cancel a specific task"""
        async with self._lock:
            if task_id in self._tasks:
                task_info = self._tasks[task_id]
                if not task_info.task.done():
                    task_info.task.cancel(msg)
                    task_info.cancelled = True
                    logger.info(f"Cancelled task: {task_id}")
                    return True
        return False
    
    async def get_active_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all active tasks"""
        async with self._lock:
            return {
                task_id: {
                    "name": info.name,
                    "priority": info.priority.value,
                    "created_at": info.created_at.isoformat(),
                    "done": info.task.done(),
                    "cancelled": info.cancelled,
                }
                for task_id, info in self._tasks.items()
            }
    
    async def graceful_shutdown(self, timeout: float = 30.0) -> Dict[str, str]:
        """Perform graceful shutdown of all tasks"""
        logger.info("Starting graceful task shutdown...")
        self._shutdown_event.set()
        
        shutdown_results = {}
        
        async with self._lock:
            # Group tasks by priority
            tasks_by_priority = {
                TaskPriority.CRITICAL: [],
                TaskPriority.HIGH: [],
                TaskPriority.NORMAL: [],
                TaskPriority.LOW: [],
            }
            
            for task_id, info in self._tasks.items():
                if not info.task.done():
                    tasks_by_priority[info.priority].append((task_id, info))
        
        # Process tasks by priority
        remaining_timeout = timeout
        
        # Critical tasks - wait for completion
        for task_id, info in tasks_by_priority[TaskPriority.CRITICAL]:
            if remaining_timeout > 0:
                start_time = asyncio.get_event_loop().time()
                try:
                    logger.info(f"Waiting for critical task: {info.name}")
                    await asyncio.wait_for(info.task, timeout=min(remaining_timeout, 10.0))
                    shutdown_results[task_id] = "completed"
                except asyncio.TimeoutError:
                    logger.warning(f"Critical task {info.name} timed out, cancelling...")
                    info.task.cancel()
                    shutdown_results[task_id] = "timeout_cancelled"
                except Exception as e:
                    logger.error(f"Critical task {info.name} failed: {e}")
                    shutdown_results[task_id] = f"failed: {str(e)}"
                
                elapsed = asyncio.get_event_loop().time() - start_time
                remaining_timeout -= elapsed
        
        # High priority tasks - give them some time
        high_priority_tasks = [info.task for _, info in tasks_by_priority[TaskPriority.HIGH]]
        if high_priority_tasks and remaining_timeout > 0:
            logger.info(f"Waiting up to {min(remaining_timeout, 5.0)}s for {len(high_priority_tasks)} high priority tasks...")
            done, pending = await asyncio.wait(
                high_priority_tasks,
                timeout=min(remaining_timeout, 5.0),
                return_when=asyncio.ALL_COMPLETED
            )
            
            for task_id, info in tasks_by_priority[TaskPriority.HIGH]:
                if info.task in done:
                    shutdown_results[task_id] = "completed"
                else:
                    info.task.cancel()
                    shutdown_results[task_id] = "cancelled"
        
        # Normal and Low priority - cancel immediately
        for priority in [TaskPriority.NORMAL, TaskPriority.LOW]:
            for task_id, info in tasks_by_priority[priority]:
                if not info.task.done():
                    info.task.cancel()
                    shutdown_results[task_id] = "cancelled"
        
        # Wait for all cancellations to complete
        all_tasks = [info.task for info in self._tasks.values()]
        if all_tasks:
            try:
                await asyncio.gather(*all_tasks, return_exceptions=True)
            except Exception:
                pass  # Ignore errors during final gathering

        if self._cleanup_tasks:
            await asyncio.gather(*self._cleanup_tasks, return_exceptions=True)
            self._cleanup_tasks.clear()

        logger.info(f"Task shutdown complete. Results: {shutdown_results}")
        return shutdown_results
    
    def is_shutting_down(self) -> bool:
        """Check if the system is shutting down"""
        return self._shutdown_event.is_set()


# Global task registry instance
task_registry = TaskRegistry()


# Convenience decorators
def background_task(name: Optional[str] = None, priority: TaskPriority = TaskPriority.NORMAL):
    """Decorator to automatically register async functions as background tasks"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            async with task_registry.create_task(
                func(*args, **kwargs),
                name=name or func.__name__,
                priority=priority
            ) as task:
                return await task
        return wrapper
    return decorator


# Example usage for critical research tasks
@background_task(name="research_synthesis", priority=TaskPriority.HIGH)
async def long_running_research_task(query: str):
    """Example of a high-priority research task"""
    # Task implementation
    pass
