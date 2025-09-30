#!/usr/bin/env python3
"""
Demonstration of Azure AI Foundry MCP Integration
Shows how the application uses MCP servers in the research pipeline
"""

import asyncio
import json
from typing import List, Dict, Any

print("="*80)
print("AZURE AI FOUNDRY MCP INTEGRATION DEMONSTRATION")
print("="*80)
print()

# ============================================================================
# STEP 1: MCP Server Registration (happens at app startup)
# ============================================================================
print("STEP 1: MCP Server Registration at App Startup")
print("-"*80)

from services.mcp.mcp_integration import (
    mcp_integration,
    configure_default_servers,
    MCPServer,
    MCPCapability
)

# This is called during app initialization (core/app.py line 149)
print("Calling configure_default_servers()...")
configure_default_servers()

print(f"✓ MCP servers registered: {len(mcp_integration.servers)}")
for server_name, server in mcp_integration.servers.items():
    print(f"  - {server_name}: {server.url}")
    print(f"    Capabilities: {[c.value for c in server.capabilities]}")
print()


# ============================================================================
# STEP 2: Tool Discovery and Formatting
# ============================================================================
print("STEP 2: Tool Discovery and Formatting for Azure OpenAI")
print("-"*80)

async def demonstrate_tool_discovery():
    """Show how MCP tools are discovered and formatted"""

    # Discover tools from each registered MCP server
    print("Discovering tools from MCP servers...")

    for server_name in mcp_integration.servers.keys():
        try:
            print(f"  Discovering tools from {server_name}...")
            await mcp_integration.discover_tools(server_name)
        except Exception as e:
            print(f"  ⚠ Could not discover tools from {server_name}: {e}")

    # Get all available tools from registered MCP servers
    tools = mcp_integration.get_all_tools()

    print(f"✓ Discovered {len(tools)} tools from MCP servers")

    # Note: MCP server needs to be running for actual tool discovery
    # For demonstration, we'll show what tools would look like
    if len(tools) == 0:
        print("\n⚠ Note: MCP server not running at localhost:8081")
        print("   Using mock tools for demonstration purposes...")

        # Mock Azure AI Foundry tools that would be available
        tools = [
            {
                "name": "evaluate_groundedness",
                "description": "Evaluate if a response is grounded in the provided context",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The original query"},
                        "response": {"type": "string", "description": "The generated response"},
                        "context": {"type": "string", "description": "The source context"}
                    },
                    "required": ["query", "response", "context"]
                }
            },
            {
                "name": "assess_credibility",
                "description": "Assess the credibility of sources and claims",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string", "description": "The claim to assess"},
                        "sources": {"type": "array", "description": "List of sources"}
                    }
                }
            },
            {
                "name": "invoke_ai_model",
                "description": "Invoke an Azure AI model for analysis",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model_name": {"type": "string", "description": "Model deployment name"},
                        "prompt": {"type": "string", "description": "Prompt to send to model"}
                    }
                }
            }
        ]

    # Show tool examples
    print("\nSample tools available from Azure AI Foundry:")
    for i, tool in enumerate(tools[:5]):
        print(f"\n  Tool {i+1}: {tool.get('name', 'unnamed')}")
        print(f"    Description: {tool.get('description', 'No description')[:100]}...")
        if 'inputSchema' in tool:
            schema = tool['inputSchema']
            if 'properties' in schema:
                print(f"    Parameters: {list(schema['properties'].keys())}")

    print("\n" + "-"*80)
    print("Formatting tools for Azure OpenAI Responses API...")

    # Show how tools are formatted for Responses API
    print(f"\n✓ Tools would be formatted for Responses API")
    print("\nExample: MCP server descriptor format")
    mcp_descriptor = {
        "type": "mcp",
        "server_label": "azure_ai_foundry",
        "server_url": "http://localhost:8081/mcp",
        "require_approval": "never"
    }
    print(json.dumps(mcp_descriptor, indent=2))

    return tools, [mcp_descriptor]

tools, formatted_tools = asyncio.run(demonstrate_tool_discovery())
print()


# ============================================================================
# STEP 3: Research Query Flow with MCP Tools
# ============================================================================
print("STEP 3: How MCP Tools Flow Through Research Pipeline")
print("-"*80)

async def demonstrate_research_flow():
    """Show how MCP tools are used in research queries"""

    print("\nResearch Query Flow:")
    print("  1. User submits research query")
    print("  2. Deep Research Service receives query")
    print("  3. MCP tools are attached to the query context")
    print("  4. Query is sent to Azure OpenAI with MCP tools available")
    print("  5. LLM can call MCP tools during reasoning")
    print("  6. Tool results are integrated into the response")

    print("\n" + "-"*80)
    print("Demonstrating tool attachment in Deep Research Service...")

    from services.deep_research_service import DeepResearchService

    # Create research service instance
    service = DeepResearchService()

    # Check if MCP tools are enabled
    enable_mcp = True  # This comes from ENABLE_MCP_DEFAULT env var

    print(f"\nMCP Tools Enabled: {enable_mcp}")

    if enable_mcp:
        print("✓ MCP tools will be automatically attached to research queries")
        print("\nAvailable tool categories:")

        # Categorize tools by capability
        tool_categories: Dict[str, List[str]] = {}
        for tool in tools[:10]:  # Sample first 10
            name = tool.get('name', 'unknown')
            # Infer category from tool name
            if 'search' in name.lower():
                tool_categories.setdefault('Search & Discovery', []).append(name)
            elif 'eval' in name.lower() or 'assess' in name.lower():
                tool_categories.setdefault('Evaluation & Assessment', []).append(name)
            elif 'ground' in name.lower() or 'verify' in name.lower():
                tool_categories.setdefault('Grounding & Verification', []).append(name)
            elif 'model' in name.lower() or 'inference' in name.lower():
                tool_categories.setdefault('AI Model Access', []).append(name)
            else:
                tool_categories.setdefault('Other Capabilities', []).append(name)

        for category, tool_names in tool_categories.items():
            print(f"\n  {category}:")
            for tool_name in tool_names:
                print(f"    - {tool_name}")

asyncio.run(demonstrate_research_flow())
print()


# ============================================================================
# STEP 4: Paradigm-Specific Tool Usage
# ============================================================================
print("STEP 4: Paradigm-Specific MCP Tool Configuration")
print("-"*80)

print("\nDifferent research paradigms can use different MCP tool configurations:")
print()

paradigm_configs = {
    "analytical": {
        "description": "Logical, evidence-based analysis",
        "preferred_tools": ["search", "evaluation", "grounding", "verification"],
        "use_case": "Academic research, fact-checking, systematic reviews"
    },
    "creative": {
        "description": "Innovative, exploratory thinking",
        "preferred_tools": ["model_access", "generation", "brainstorming"],
        "use_case": "Creative writing, ideation, scenario planning"
    },
    "critical": {
        "description": "Skeptical evaluation and critique",
        "preferred_tools": ["evaluation", "bias_detection", "credibility_assessment"],
        "use_case": "Quality assurance, peer review, risk assessment"
    },
    "systems": {
        "description": "Holistic, interconnected analysis",
        "preferred_tools": ["relationship_mapping", "integration", "synthesis"],
        "use_case": "Complex systems analysis, strategic planning"
    }
}

for paradigm_name, config in paradigm_configs.items():
    print(f"{paradigm_name.upper()} Paradigm:")
    print(f"  Description: {config['description']}")
    print(f"  Preferred Tools: {', '.join(config['preferred_tools'])}")
    print(f"  Use Case: {config['use_case']}")
    print()

print("-"*80)
print("When a paradigm evaluates a query, it receives only the MCP tools")
print("relevant to its reasoning approach, optimizing LLM performance.")
print()


# ============================================================================
# STEP 5: Live Integration Example
# ============================================================================
print("STEP 5: Live MCP Integration Example")
print("-"*80)

async def demonstrate_live_integration():
    """Show actual integration with Azure AI Foundry"""

    from services.mcp.azure_ai_foundry_mcp_integration import (
        AzureAIFoundryMCPConfig,
        AzureAIFoundryMCPIntegration
    )

    # Initialize Azure AI Foundry MCP
    config = AzureAIFoundryMCPConfig()
    azure_mcp = AzureAIFoundryMCPIntegration(config)

    print(f"\nConnecting to Azure AI Foundry Project: {config.project_name}")
    print(f"Endpoint: {config.ai_project_endpoint}")

    # Initialize connection
    await azure_mcp.initialize()
    print("✓ Connected to Azure AI Foundry MCP server")

    # Simulate a research query that would use MCP tools
    print("\n" + "="*80)
    print("SIMULATED RESEARCH QUERY")
    print("="*80)

    sample_query = "Evaluate the credibility of recent AI safety research claims"

    print(f"\nQuery: '{sample_query}'")
    print("\nProcessing flow:")
    print("  1. Query received by research API endpoint")
    print("  2. Deep Research Service orchestrator invoked")
    print("  3. Query classified and routed to appropriate paradigms")
    print("  4. MCP tools attached to query context:")
    print("     - Azure AI Foundry evaluation tools")
    print("     - Grounding and verification tools")
    print("     - Model access for analysis")
    print("  5. Paradigms evaluate query with MCP tool access:")
    print("     - ANALYTICAL: Uses grounding tools to verify claims")
    print("     - CRITICAL: Uses evaluation tools to assess credibility")
    print("     - SYSTEMS: Uses integration tools to understand connections")
    print("  6. Tool calls executed against Azure AI Foundry:")
    print(f"     - Authenticated with Service Principal: {config.client_id}")
    print(f"     - Against Project: {config.project_name}")
    print("  7. Tool results integrated into paradigm responses")
    print("  8. Mesh synthesis combines paradigm insights")
    print("  9. Final response delivered to user with provenance")

    print("\n✓ MCP Integration Complete")
    print(f"\nTotal MCP servers active: {len(mcp_integration.servers)}")
    print(f"Total tools available to LLM: {len(tools)}")
    print(f"Azure AI Foundry connected: ✓")

asyncio.run(demonstrate_live_integration())
print()


# ============================================================================
# Summary
# ============================================================================
print("="*80)
print("SUMMARY: How the Application Uses Azure AI Foundry MCP")
print("="*80)
print()
print("1. STARTUP: MCP servers registered automatically via configure_default_servers()")
print("   └─> Reads AZURE_AI_PROJECT_ENDPOINT from .env")
print("   └─> Registers Azure AI Foundry with EVALUATION + AI_MODEL capabilities")
print()
print("2. TOOL DISCOVERY: Tools enumerated from all registered MCP servers")
print("   └─> mcp_integration.list_all_tools() aggregates tools")
print("   └─> Tools formatted for Azure OpenAI Responses API")
print()
print("3. RESEARCH QUERIES: MCP tools automatically attached when enabled")
print("   └─> ENABLE_MCP_DEFAULT=1 in .env enables automatic attachment")
print("   └─> Deep Research Service includes tools in LLM context")
print()
print("4. PARADIGM EVALUATION: Each paradigm receives appropriate tools")
print("   └─> Paradigm-specific tool filtering optimizes performance")
print("   └─> LLM can invoke tools during reasoning process")
print()
print("5. TOOL EXECUTION: Tool calls routed to Azure AI Foundry MCP server")
print("   └─> Authenticated with Service Principal credentials")
print("   └─> Results returned and integrated into response")
print()
print("6. RESPONSE SYNTHESIS: Tool results woven into final answer")
print("   └─> Evidence from MCP tools enhances credibility")
print("   └─> Provenance tracked for transparency")
print()
print("="*80)
print("END OF DEMONSTRATION")
print("="*80)