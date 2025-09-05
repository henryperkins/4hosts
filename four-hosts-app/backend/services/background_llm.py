"""
Background LLM Task Management for Azure OpenAI Responses API
Handles long-running LLM tasks using the background mode
"""

import asyncio
import logging
from typing import Dict, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import os

from services.cache import cache_manager

logger = logging.getLogger(__name__)


class BackgroundTaskStatus(str, Enum):
    """Status of background LLM tasks"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BackgroundLLMManager:
    """Manages background LLM tasks for long-running operations"""
    
    def __init__(self, azure_client):
        self.azure_client = azure_client
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self._polling_tasks: Dict[str, asyncio.Task] = {}
        try:
            self.poll_interval = int(os.getenv("LLM_BG_POLL_INTERVAL", "2") or 2)
        except Exception:
            self.poll_interval = 2
        try:
            self.max_poll_duration = int(os.getenv("LLM_BG_MAX_SECS", "300") or 300)
        except Exception:
            self.max_poll_duration = 300
    
    async def submit_background_task(
        self,
        messages: list,
        tools: Optional[list] = None,
        callback: Optional[Callable] = None,
        **kwargs
    ) -> str:
        """Submit a task to run in background mode"""
        try:
            # Prepare request with background mode
            response_kwargs = {
                "model": os.getenv("AZURE_OPENAI_DEPLOYMENT", "o3"),
                "input": messages,
                "mode": "background",
                "max_output_tokens": kwargs.get("max_output_tokens", 4000),
            }
            
            if tools:
                response_kwargs["tools"] = tools
            
            # Submit to Azure OpenAI
            response = await self.azure_client.responses.create(**response_kwargs)
            
            # Extract task ID from response
            task_id = response.id if hasattr(response, 'id') else str(response)
            
            # Store task info
            self.active_tasks[task_id] = {
                "status": BackgroundTaskStatus.PENDING,
                "created_at": datetime.utcnow(),
                "callback": callback,
                "metadata": kwargs.get("metadata", {}),
            }
            
            # Start polling task with task registry
            from services.task_registry import task_registry, TaskPriority
            
            polling_task = asyncio.create_task(
                self._poll_task_status(task_id)
            )
            self._polling_tasks[task_id] = polling_task
            
            # Register with high priority since LLM tasks are expensive
            await task_registry.register_task(
                polling_task,
                name=f"llm_poll_{task_id[:8]}",
                priority=TaskPriority.HIGH
            )
            
            logger.info(f"Submitted background LLM task: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to submit background task: {e}")
            raise
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the current status of a background task"""
        if task_id not in self.active_tasks:
            # Check cache for completed tasks
            cached = await cache_manager.get_llm_response(task_id)
            if cached:
                return {
                    "id": task_id,
                    "status": BackgroundTaskStatus.COMPLETED,
                    "result": cached,
                }
            return {"id": task_id, "status": "unknown", "error": "Task not found"}
        
        return {
            "id": task_id,
            "status": self.active_tasks[task_id]["status"],
            "metadata": self.active_tasks[task_id].get("metadata", {}),
            "created_at": self.active_tasks[task_id]["created_at"].isoformat(),
        }
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a background task"""
        if task_id not in self.active_tasks:
            return False
        
        # Update status
        self.active_tasks[task_id]["status"] = BackgroundTaskStatus.CANCELLED
        
        # Cancel polling task
        if task_id in self._polling_tasks:
            self._polling_tasks[task_id].cancel()
            del self._polling_tasks[task_id]
        
        logger.info(f"Cancelled background task: {task_id}")
        return True
    
    async def _poll_task_status(self, task_id: str):
        """Poll for task completion"""
        start_time = datetime.utcnow()
        
        try:
            while True:
                # Check if cancelled
                if self.active_tasks[task_id]["status"] == BackgroundTaskStatus.CANCELLED:
                    break
                
                # Check timeout
                if (datetime.utcnow() - start_time).seconds > self.max_poll_duration:
                    logger.warning(f"Background task {task_id} timed out")
                    self.active_tasks[task_id]["status"] = BackgroundTaskStatus.FAILED
                    self.active_tasks[task_id]["error"] = "Task timed out"
                    break
                
                # Poll Azure API for status
                try:
                    # Get task status from Azure
                    response = await self.azure_client.responses.retrieve(task_id)
                    
                    if hasattr(response, 'status'):
                        if response.status == "completed":
                            # Task completed
                            self.active_tasks[task_id]["status"] = BackgroundTaskStatus.COMPLETED
                            
                            # Extract result
                            result = self._extract_result(response)
                            
                            # Cache result
                            await cache_manager.set_llm_response(task_id, result)
                            
                            # Execute callback if provided
                            callback = self.active_tasks[task_id].get("callback")
                            if callback:
                                await callback(task_id, result)
                            
                            logger.info(f"Background task {task_id} completed")
                            break
                            
                        elif response.status == "failed":
                            self.active_tasks[task_id]["status"] = BackgroundTaskStatus.FAILED
                            self.active_tasks[task_id]["error"] = getattr(response, 'error', 'Unknown error')
                            logger.error(f"Background task {task_id} failed")
                            break
                        
                        else:
                            # Still processing
                            self.active_tasks[task_id]["status"] = BackgroundTaskStatus.RUNNING
                    
                except Exception as e:
                    logger.error(f"Error polling task {task_id}: {e}")
                
                # Wait before next poll
                await asyncio.sleep(self.poll_interval)
                
        except asyncio.CancelledError:
            logger.info(f"Polling cancelled for task {task_id}")
        finally:
            # Cleanup
            if task_id in self._polling_tasks:
                del self._polling_tasks[task_id]
    
    def _extract_result(self, response) -> Any:
        """Extract result from completed response"""
        # Handle different response formats
        if hasattr(response, 'output_text') and response.output_text:
            return response.output_text
        
        if hasattr(response, 'output') and response.output:
            # Extract from output array
            text_content = ""
            tool_calls = []
            
            for output in response.output:
                if hasattr(output, 'content'):
                    for item in output.content:
                        if hasattr(item, 'text'):
                            text_content += item.text
                
                if hasattr(output, 'tool_calls') and output.tool_calls:
                    tool_calls.extend(output.tool_calls)
            
            if tool_calls:
                return {"text": text_content, "tool_calls": tool_calls}
            return text_content
        
        # Return raw response if format unknown
        return response
    
    async def wait_for_task(self, task_id: str, timeout: int = 60) -> Any:
        """Wait for a task to complete and return result"""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).seconds < timeout:
            status = await self.get_task_status(task_id)
            
            if status["status"] == BackgroundTaskStatus.COMPLETED:
                # Get from cache
                result = await cache_manager.get_llm_response(task_id)
                return result
            
            elif status["status"] in [BackgroundTaskStatus.FAILED, BackgroundTaskStatus.CANCELLED]:
                raise Exception(f"Task {task_id} failed: {status.get('error', 'Unknown error')}")
            
            await asyncio.sleep(1)
        
        raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
    
    async def cleanup(self):
        """Cleanup all active polling tasks"""
        logger.info("Cleaning up background LLM manager...")
        
        # Cancel all polling tasks
        for task_id, polling_task in list(self._polling_tasks.items()):
            if not polling_task.done():
                polling_task.cancel()
                logger.debug(f"Cancelled polling task for {task_id}")
        
        # Wait for all polling tasks to complete
        if self._polling_tasks:
            await asyncio.gather(*self._polling_tasks.values(), return_exceptions=True)
        
        # Mark all active tasks as cancelled
        for task_id in self.active_tasks:
            if self.active_tasks[task_id]["status"] in [BackgroundTaskStatus.PENDING, BackgroundTaskStatus.RUNNING]:
                self.active_tasks[task_id]["status"] = BackgroundTaskStatus.CANCELLED
        
        logger.info("Background LLM manager cleanup complete")


# Global instance (initialized when LLM client is initialized)
background_llm_manager: Optional[BackgroundLLMManager] = None


def initialize_background_manager(azure_client):
    """Initialize the background LLM manager"""
    global background_llm_manager
    background_llm_manager = BackgroundLLMManager(azure_client)
    logger.info("Background LLM manager initialized")
