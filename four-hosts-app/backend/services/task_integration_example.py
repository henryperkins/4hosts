"""
Example integration of Task Registry with Four Hosts services
Shows how to properly register and manage background tasks
"""

import asyncio
from services.task_registry import task_registry, TaskPriority, background_task
from services.search_apis import SearchAPIManager, SearchConfig
from typing import List, Dict, Any


class TaskAwareSearchManager(SearchAPIManager):
    """Extended SearchAPIManager that uses the task registry"""
    
    async def search_all_registered(
        self, query: str, config: SearchConfig
    ) -> Dict[str, List[Any]]:
        """Search using all APIs with proper task registration"""
        # Ensure APIs are initialized
        if not self._initialized:
            await self.initialize()

        results = {}
        tasks = []

        # Create search tasks with registration
        for name, api in self.apis.items():
            async with task_registry.create_task(
                api.search(query, config),
                name=f"search_{name}_{query[:20]}",
                priority=TaskPriority.HIGH  # Search tasks are important
            ) as task:
                tasks.append((name, task))

        # Wait for all tasks
        for name, task in tasks:
            try:
                api_results = await task
                results[name] = api_results
            except asyncio.CancelledError:
                logger.warning(f"{name} search was cancelled during shutdown")
                results[name] = []
            except Exception as e:
                logger.error(f"{name} search failed: {str(e)}")
                results[name] = []

        return results


# Example usage in research orchestrator
@background_task(name="deep_research", priority=TaskPriority.CRITICAL)
async def perform_deep_research(query: str, user_id: str):
    """
    Example of a critical research task that should complete before shutdown
    """
    # Check if system is shutting down
    if task_registry.is_shutting_down():
        logger.warning("System is shutting down, skipping new research task")
        return None
    
    try:
        # Perform research steps
        logger.info(f"Starting deep research for query: {query}")
        
        # Step 1: Classification (HIGH priority subtask)
        async with task_registry.create_task(
            classify_query(query),
            name="classify_query",
            priority=TaskPriority.HIGH
        ) as classification_task:
            paradigm = await classification_task
        
        # Step 2: Search (HIGH priority subtask)
        async with task_registry.create_task(
            search_sources(query, paradigm),
            name="search_sources",
            priority=TaskPriority.HIGH
        ) as search_task:
            sources = await search_task
        
        # Step 3: Generate answer (CRITICAL - must complete)
        async with task_registry.create_task(
            generate_answer(query, sources, paradigm),
            name="generate_answer",
            priority=TaskPriority.CRITICAL
        ) as answer_task:
            answer = await answer_task
        
        return {
            "query": query,
            "paradigm": paradigm,
            "sources": sources,
            "answer": answer
        }
        
    except asyncio.CancelledError:
        logger.warning(f"Deep research task cancelled for query: {query}")
        # Save partial results to cache for recovery
        await save_partial_results(query, user_id)
        raise
    except Exception as e:
        logger.error(f"Deep research failed: {e}")
        raise


# WebSocket handler with task registration
async def handle_websocket_with_tasks(websocket, user_id: str):
    """Example WebSocket handler that registers its task"""
    task_id = await task_registry.register_task(
        asyncio.current_task(),
        name=f"websocket_{user_id}",
        priority=TaskPriority.NORMAL  # WebSockets can be closed during shutdown
    )
    
    try:
        while True:
            message = await websocket.receive_text()
            
            # Process message with subtask
            async with task_registry.create_task(
                process_websocket_message(message, user_id),
                name=f"ws_msg_{user_id}",
                priority=TaskPriority.LOW
            ) as process_task:
                response = await process_task
                await websocket.send_text(response)
                
    except asyncio.CancelledError:
        # Graceful WebSocket closure
        await websocket.close(code=1001, reason="Server shutting down")
        raise


# Monitoring endpoint that shows active tasks
async def get_system_status():
    """Get current system status including active tasks"""
    active_tasks = await task_registry.get_active_tasks()
    
    # Group by priority
    tasks_by_priority = {
        "critical": [],
        "high": [],
        "normal": [],
        "low": []
    }
    
    for task_id, info in active_tasks.items():
        tasks_by_priority[info["priority"]].append({
            "id": task_id,
            "name": info["name"],
            "created_at": info["created_at"],
            "done": info["done"]
        })
    
    return {
        "total_active_tasks": len(active_tasks),
        "tasks_by_priority": tasks_by_priority,
        "is_shutting_down": task_registry.is_shutting_down()
    }


# Helper functions (stubs)
async def classify_query(query: str) -> str:
    """Stub for query classification"""
    await asyncio.sleep(0.1)  # Simulate work
    return "dolores"


async def search_sources(query: str, paradigm: str) -> List[Dict]:
    """Stub for source search"""
    await asyncio.sleep(0.5)  # Simulate work
    return [{"url": "example.com", "content": "..."}]


async def generate_answer(query: str, sources: List[Dict], paradigm: str) -> str:
    """Stub for answer generation"""
    await asyncio.sleep(1.0)  # Simulate work
    return "Generated answer..."


async def save_partial_results(query: str, user_id: str):
    """Stub for saving partial results"""
    pass


async def process_websocket_message(message: str, user_id: str) -> str:
    """Stub for processing WebSocket messages"""
    await asyncio.sleep(0.1)
    return f"Processed: {message}"