"""
Enhanced Main Integration Module
Shows how to integrate all the V2 services into main.py
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio

from fastapi import BackgroundTasks
from models.context_models import (
    ClassificationResultSchema, ResearchRequestSchema,
    UserContextSchema, ResearchStatus, ResearchResultSchema,
    HostParadigm
)
from services.research_store_v2 import research_store_v2
from services.websocket_service_v2 import connection_manager_v2, WSEventType
from services.context_engineering_v2 import context_pipeline_v2
from services.research_orchestrator_v2 import research_orchestrator_v2
from services.classification_engine import classification_engine

logger = logging.getLogger(__name__)


class EnhancedResearchExecutor:
    """Orchestrates research execution with all V2 enhancements"""
    
    def __init__(self):
        self.research_store = research_store_v2
        self.ws_manager = connection_manager_v2
        self.context_pipeline = context_pipeline_v2
        self.orchestrator = research_orchestrator_v2
    
    async def execute_research_with_context(
        self,
        research_id: str,
        query: str,
        user_context: UserContextSchema,
        classification: Optional[ClassificationResultSchema] = None,
        options: Optional[Dict[str, Any]] = None
    ):
        """Execute research with full context preservation"""
        
        try:
            # Create research request
            research_request = ResearchRequestSchema(
                id=research_id,
                query=query,
                user_context=user_context,
                options=options or {},
                classification=classification,
                status=ResearchStatus.PROCESSING
            )
            
            # Store initial request
            await self.research_store.store_research_request(research_request)
            
            # Notify via WebSocket
            await self.ws_manager.broadcast_to_research(
                research_id,
                WSEventType.RESEARCH_STARTED,
                {
                    "query": query,
                    "paradigm": classification.primary_paradigm.value if classification else "unknown",
                    "user_role": user_context.role,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            # If no classification provided, generate it
            if not classification:
                await self._update_progress(research_id, "Classifying query", 10)
                
                classification_result = await classification_engine.classify_query(query)
                classification = self._convert_classification(classification_result)
                
                # Store classification
                await self.research_store.store_classification(research_id, classification)
            
            # Process through context engineering
            await self._update_progress(research_id, "Engineering context", 20)
            
            context_engineered = await self.context_pipeline.process_query(
                classification,
                include_debug=user_context.is_pro_user  # Include debug for pro users
            )
            
            # Execute research
            await self._update_progress(research_id, "Executing searches", 40)
            
            research_results = await self.orchestrator.execute_research(
                classification,
                context_engineered,
                user_context,
                progress_callback=lambda msg: self._update_progress(research_id, msg, 50)
            )
            
            # Generate answer (placeholder - would call answer generator)
            await self._update_progress(research_id, "Generating answer", 80)
            
            answer = await self._generate_answer(
                classification,
                context_engineered,
                research_results,
                user_context
            )
            
            # Create final result
            result = ResearchResultSchema(
                research_id=research_id,
                query=query,
                classification=classification,
                context_engineering=context_engineered,
                search_results=research_results['results'],
                answer=answer['content'],
                paradigm_tone=classification.primary_paradigm,
                sources_used=answer['sources'],
                credibility_summary=research_results['metadata']['credibility_summary'],
                user_context=user_context,
                processing_time_ms=(datetime.utcnow() - research_request.created_at).total_seconds() * 1000
            )
            
            # Store result
            await self.research_store.store_result(result)
            
            # Final notification
            await self.ws_manager.broadcast_to_research(
                research_id,
                WSEventType.RESEARCH_COMPLETED,
                {
                    "query": query,
                    "answer_preview": answer['content'][:200] + "...",
                    "sources_count": len(answer['sources']),
                    "paradigm": classification.primary_paradigm.value,
                    "processing_time_ms": result.processing_time_ms
                }
            )
            
            await self._update_progress(research_id, "Research completed", 100)
            
            return result
            
        except Exception as e:
            logger.error(f"Research execution failed: {e}")
            
            # Update status
            await self.research_store.update_status(
                research_id,
                ResearchStatus.FAILED,
                error_message=str(e)
            )
            
            # Notify failure
            await self.ws_manager.broadcast_to_research(
                research_id,
                WSEventType.RESEARCH_FAILED,
                {
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            raise
    
    async def _update_progress(self, research_id: str, message: str, progress: int):
        """Update research progress via WebSocket"""
        await self.ws_manager.broadcast_to_research(
            research_id,
            WSEventType.RESEARCH_PROGRESS,
            {
                "message": message,
                "progress": progress,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    def _convert_classification(self, classification_result) -> ClassificationResultSchema:
        """Convert between classification formats"""
        # This would handle the conversion from the engine's format
        # to our schema format
        pass
    
    async def _generate_answer(
        self,
        classification: ClassificationResultSchema,
        context_engineered: ContextEngineeredQuerySchema,
        research_results: Dict[str, Any],
        user_context: UserContextSchema
    ) -> Dict[str, Any]:
        """Generate answer with full context"""
        # This would call the answer generator with all context
        # For now, return a placeholder
        return {
            "content": f"Answer for {classification.query} from {classification.primary_paradigm.value} perspective",
            "sources": research_results['results'][:5] if research_results['results'] else []
        }


# Example of how to modify main.py endpoints

async def research_endpoint_v2(
    research_request: Dict[str, Any],
    current_user: Any,
    background_tasks: BackgroundTasks
):
    """Enhanced research endpoint with V2 services"""
    
    # Create user context
    user_context = UserContextSchema(
        user_id=str(current_user.id),
        role=current_user.role.value,
        preferences=current_user.preferences or {},
        location=current_user.location,
        language=current_user.language or "en",
        default_paradigm=current_user.default_paradigm
    )
    
    # Generate research ID
    research_id = f"research_{current_user.id}_{datetime.utcnow().timestamp()}"
    
    # Execute in background
    executor = EnhancedResearchExecutor()
    background_tasks.add_task(
        executor.execute_research_with_context,
        research_id,
        research_request['query'],
        user_context,
        options=research_request.get('options', {})
    )
    
    return {
        "research_id": research_id,
        "status": "processing",
        "message": "Research started successfully"
    }


# WebSocket endpoint enhancement
async def websocket_endpoint_v2(websocket, token: str):
    """Enhanced WebSocket endpoint with V2 service"""
    
    # Verify token and get user
    user_data = decode_token(token)
    if not user_data:
        await websocket.close(code=1008, reason="Invalid token")
        return
    
    # Connect with metadata
    await connection_manager_v2.connect(
        websocket,
        user_data['user_id'],
        metadata={
            "connected_at": datetime.utcnow().isoformat(),
            "user_agent": websocket.headers.get("User-Agent", "Unknown"),
            "protocol_version": "2.0"
        }
    )
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            # Handle different message types
            if data.get("type") == "subscribe":
                research_id = data.get("research_id")
                if research_id:
                    await connection_manager_v2.subscribe_to_research(
                        websocket, research_id
                    )
            
            elif data.get("type") == "ping":
                await connection_manager_v2.handle_ping(websocket)
            
            elif data.get("type") == "get_status":
                research_id = data.get("research_id")
                if research_id:
                    research_data = await research_store_v2.get_research(research_id)
                    if research_data:
                        await websocket.send_json({
                            "type": "status_update",
                            "data": research_data
                        })
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await connection_manager_v2.disconnect(websocket)


# Startup and shutdown hooks
async def startup_v2():
    """Initialize V2 services on startup"""
    # Initialize stores
    await research_store_v2.initialize()
    
    # Initialize other services
    logger.info("V2 services initialized")


async def shutdown_v2():
    """Cleanup V2 services on shutdown"""
    # Close connections
    await research_store_v2.close()
    
    logger.info("V2 services shut down")


# Memory management endpoint
async def get_system_metrics():
    """Get system metrics from V2 services"""
    return {
        "research_store": research_store_v2.get_metrics(),
        "websocket": connection_manager_v2.get_metrics(),
        "context_pipeline": context_pipeline_v2.get_processing_metrics(),
        "timestamp": datetime.utcnow().isoformat()
    }