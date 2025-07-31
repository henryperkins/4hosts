"""
Research Integration with MCP Tools and Background Processing
Extends the research orchestrator to use MCP tools and background LLM tasks
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from services.research_orchestrator import research_orchestrator
from services.llm_client import llm_client
from services.mcp_integration import mcp_integration, MCPToolDefinition
from services.background_llm import background_llm_manager

logger = logging.getLogger(__name__)


class ToolEnhancedResearchOrchestrator:
    """Research orchestrator enhanced with MCP tools and background processing"""
    
    def __init__(self):
        self.base_orchestrator = research_orchestrator
        self.active_tool_tasks: Dict[str, Any] = {}
    
    async def execute_research_with_tools(
        self,
        query: str,
        paradigm: str,
        options: Dict[str, Any],
        available_mcp_servers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Execute research with MCP tools support"""
        
        # Discover available tools from MCP servers
        all_tools = []
        if available_mcp_servers:
            for server_name in available_mcp_servers:
                try:
                    tools = await mcp_integration.discover_tools(server_name)
                    all_tools.extend(tools)
                    logger.info(f"Discovered {len(tools)} tools from {server_name}")
                except Exception as e:
                    logger.error(f"Failed to discover tools from {server_name}: {e}")
        
        # Prepare the research prompt with tool awareness
        research_prompt = self._build_tool_aware_prompt(query, paradigm, all_tools)
        
        # Convert tools to Azure OpenAI format
        azure_tools = [tool.dict() for tool in all_tools]
        
        # Prepare messages
        messages = [
            {"role": "system", "content": research_prompt["system"]},
            {"role": "user", "content": research_prompt["user"]}
        ]
        
        # Decide whether to use background mode based on research depth
        use_background = options.get("depth") in ["deep", "deep_research"]
        
        if use_background and background_llm_manager:
            # Submit to background processing
            task_id = await llm_client.generate_background(
                messages=messages,
                tools=azure_tools if azure_tools else None,
                callback=self._handle_research_completion,
                metadata={
                    "query": query,
                    "paradigm": paradigm,
                    "research_id": options.get("research_id")
                }
            )
            
            logger.info(f"Research submitted to background processing: {task_id}")
            
            # Return task info for tracking
            return {
                "status": "background_processing",
                "task_id": task_id,
                "message": "Research is being processed in the background"
            }
        
        else:
            # Execute synchronously with tool support
            response = await llm_client.chat(
                messages=messages,
                tools=azure_tools if azure_tools else None,
                model="o3"  # Use o3 for tool calling
            )
            
            # Handle tool calls if present
            if hasattr(response, 'output') and response.output:
                for output in response.output:
                    if hasattr(output, 'tool_calls') and output.tool_calls:
                        # Execute tool calls
                        tool_results = await self._execute_tool_calls(output.tool_calls)
                        
                        # Add tool results to conversation
                        for tool_call, result in zip(output.tool_calls, tool_results):
                            messages.append({
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [tool_call]
                            })
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "content": str(result)
                            })
                        
                        # Get final response with tool results
                        final_response = await llm_client.chat(
                            messages=messages,
                            model="o3"
                        )
                        
                        return self._format_research_result(final_response, paradigm)
            
            # No tool calls, return direct response
            return self._format_research_result(response, paradigm)
    
    def _build_tool_aware_prompt(
        self, 
        query: str, 
        paradigm: str, 
        tools: List[MCPToolDefinition]
    ) -> Dict[str, str]:
        """Build prompts that inform the model about available tools"""
        
        tool_descriptions = "\n".join([
            f"- {tool.function['name']}: {tool.function['description']}"
            for tool in tools
        ]) if tools else "No external tools available."
        
        system_prompt = f"""You are a {paradigm}-aligned research assistant with access to external tools.

Available tools:
{tool_descriptions}

Use these tools when they would enhance your research quality. Always cite sources from tool results.

Research paradigm: {paradigm.upper()}
- Focus on {self._get_paradigm_focus(paradigm)}
- Maintain {paradigm} perspective throughout
"""
        
        user_prompt = f"""Research Query: {query}

Please conduct comprehensive research on this topic. Use available tools to gather additional information when relevant."""
        
        return {"system": system_prompt, "user": user_prompt}
    
    def _get_paradigm_focus(self, paradigm: str) -> str:
        """Get the focus description for each paradigm"""
        focus_map = {
            "dolores": "exposing injustices and challenging systems",
            "teddy": "helping and protecting vulnerable populations",
            "bernard": "data-driven analysis and empirical evidence",
            "maeve": "strategic optimization and competitive advantage"
        }
        return focus_map.get(paradigm.lower(), "comprehensive analysis")
    
    async def _execute_tool_calls(self, tool_calls: List[Any]) -> List[Any]:
        """Execute a list of tool calls"""
        results = []
        
        for tool_call in tool_calls:
            try:
                # Parse function name and arguments
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                # Execute through MCP integration
                result = await mcp_integration.execute_tool_call(
                    function_name, 
                    arguments
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Tool execution error for {function_name}: {e}")
                results.append({"error": str(e)})
        
        return results
    
    async def _handle_research_completion(self, task_id: str, result: Any):
        """Callback for background research completion"""
        metadata = self.active_tool_tasks.get(task_id, {})
        research_id = metadata.get("research_id")
        
        if research_id:
            # Update research store with results
            from services.research_store import research_store
            await research_store.update_field(
                research_id,
                "results",
                self._format_research_result(result, metadata.get("paradigm"))
            )
            await research_store.update_field(
                research_id,
                "status",
                "completed"
            )
            
            logger.info(f"Background research {research_id} completed")
    
    def _format_research_result(self, response: Any, paradigm: str) -> Dict[str, Any]:
        """Format the research result"""
        # Extract text content
        if isinstance(response, str):
            content = response
        elif hasattr(response, 'output_text'):
            content = response.output_text
        else:
            content = str(response)
        
        return {
            "content": content,
            "paradigm": paradigm,
            "timestamp": datetime.utcnow().isoformat(),
            "tools_used": True if "tool" in content.lower() else False
        }


# Global instance
tool_enhanced_orchestrator = ToolEnhancedResearchOrchestrator()


# Example usage
async def example_research_with_tools():
    """Example of using tool-enhanced research"""
    
    # Register an MCP server (example)
    from services.mcp_integration import MCPServer, MCPCapability
    
    mcp_integration.register_server(MCPServer(
        name="research_db",
        url="http://localhost:8080/mcp",
        capabilities=[MCPCapability.DATABASE, MCPCapability.SEARCH],
        auth_token="example_token"
    ))
    
    # Execute research with tools
    result = await tool_enhanced_orchestrator.execute_research_with_tools(
        query="What are the latest breakthroughs in quantum computing?",
        paradigm="bernard",  # Analytical paradigm
        options={
            "depth": "deep",
            "research_id": "res_example_123"
        },
        available_mcp_servers=["research_db"]
    )
    
    return result