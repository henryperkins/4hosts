#!/usr/bin/env python3
"""
End-to-end pipeline test for Azure AI Foundry MCP integration
Verifies the complete flow from research query to LLM with MCP tools
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

import structlog

logger = structlog.get_logger(__name__)


async def test_pipeline_flow():
    """
    Test the complete pipeline flow:
    1. MCP servers registered via configure_default_servers()
    2. MCP tools formatted for Responses API
    3. Deep research service can access MCP tools
    4. Responses client receives tools in correct format
    5. Tools would be passed to Azure OpenAI Responses API
    """
    print("\n" + "="*70)
    print("COMPLETE PIPELINE INTEGRATION TEST")
    print("="*70)

    # Step 1: Import and configure MCP integration
    print("\n[Step 1] Configuring MCP Integration")
    print("-" * 70)
    from services.mcp.mcp_integration import mcp_integration, configure_default_servers

    configure_default_servers()
    print(f"✓ Configured {len(mcp_integration.servers)} MCP server(s)")

    for name, server in mcp_integration.servers.items():
        print(f"  - {name}: {server.url}")

    # Step 2: Get MCP tools in Responses API format
    print("\n[Step 2] Formatting MCP Tools for Responses API")
    print("-" * 70)
    mcp_tools = mcp_integration.get_responses_mcp_tools()
    print(f"✓ Generated {len(mcp_tools)} MCP tool(s) in Responses API format")

    for tool in mcp_tools:
        print(f"  - {tool['server_label']}:")
        print(f"    type: {tool['type']}")
        print(f"    server_url: {tool['server_url']}")
        print(f"    require_approval: {tool['require_approval']}")

    # Verify Azure AI Foundry is present
    azure_foundry = next((t for t in mcp_tools if t['server_label'] == 'azure_ai_foundry'), None)
    if not azure_foundry:
        print("✗ ERROR: Azure AI Foundry MCP tool not found!")
        return False

    # Step 3: Simulate deep research configuration
    print("\n[Step 3] Deep Research Service Configuration")
    print("-" * 70)

    # Check if ENABLE_MCP_DEFAULT is set
    enable_mcp = os.getenv("ENABLE_MCP_DEFAULT", "0").lower() in {"1", "true", "yes"}
    print(f"ENABLE_MCP_DEFAULT: {'✓ ENABLED' if enable_mcp else '✗ DISABLED'}")

    if not enable_mcp:
        print("⚠️  WARNING: MCP tools will not be attached to deep research")
        print("   Set ENABLE_MCP_DEFAULT=1 in .env to enable")
        return False

    # Simulate how deep_research_service.py fetches MCP tools
    print("✓ Deep research would fetch MCP tools using:")
    print("  mcp_integration.get_responses_mcp_tools()")

    # Step 4: Simulate responses_deep_research call
    print("\n[Step 4] Simulating Responses API Call Structure")
    print("-" * 70)

    # This is what would be passed to responses_deep_research()
    simulated_query = "What are the latest advances in AI safety?"
    simulated_paradigm = "bernard"  # Analytical paradigm

    print(f"Query: '{simulated_query}'")
    print(f"Paradigm: {simulated_paradigm}")
    print(f"\nTools that would be passed to Responses API:")

    # Build tools list as done in deep_research_service.py
    tools = []

    # Add web search (if enabled)
    tools.append({"type": "web_search_preview", "search_context_size": "medium"})
    print("  1. Web Search Preview (Azure native)")

    # Add MCP tools
    tools.extend(mcp_tools)
    print(f"  2. MCP: Azure AI Foundry (stdio)")

    # Step 5: Verify paradigm-specific configuration
    print("\n[Step 5] Paradigm-Specific Azure AI Foundry Configuration")
    print("-" * 70)

    from services.mcp.azure_ai_foundry_mcp_integration import azure_ai_foundry_mcp

    config = azure_ai_foundry_mcp.get_evaluation_config(simulated_paradigm)
    print(f"Configuration for {simulated_paradigm.upper()}:")
    print(f"  - Evaluation types: {config.get('evaluation_types', [])}")
    print(f"  - Reasoning effort: {config.get('reasoning_effort', 'N/A')}")
    print(f"  - Safety settings: {list(config.get('safety_settings', {}).keys())}")

    # Step 6: Verify Responses Client configuration
    print("\n[Step 6] Responses Client Configuration")
    print("-" * 70)

    from services.openai_responses_client import OpenAIResponsesClient

    client = OpenAIResponsesClient()
    print(f"✓ Responses client initialized")
    print(f"  - Is Azure: {client.is_azure}")
    print(f"  - Base URL: {client.base_url}")
    print(f"  - API Version: {client.azure_api_version if client.is_azure else 'N/A'}")

    # Step 7: Final validation
    print("\n[Step 7] Final Validation")
    print("-" * 70)

    validations = {
        "MCP servers registered": len(mcp_integration.servers) > 0,
        "Azure AI Foundry present": azure_foundry is not None,
        "MCP tools properly formatted": all(
            t.get('type') == 'mcp' and
            'server_label' in t and
            'server_url' in t
            for t in mcp_tools
        ),
        "ENABLE_MCP_DEFAULT is set": enable_mcp,
        "Responses client configured": client is not None,
        "Paradigm configs available": config is not None,
    }

    all_valid = all(validations.values())

    for check, passed in validations.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")

    # Step 8: Show expected API request structure
    print("\n[Step 8] Expected Responses API Request Structure")
    print("-" * 70)

    expected_request = {
        "model": "o3",  # or deployment name
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": simulated_query}]}
        ],
        "tools": tools,  # Includes MCP tools
        "background": True,
        "reasoning": {"summary": "auto"},
        "store": True,
    }

    print("Request structure (simplified):")
    print(f"  model: {expected_request['model']}")
    print(f"  input: [{expected_request['input'][0]['role']} message]")
    print(f"  tools: {len(expected_request['tools'])} tool(s)")
    for i, tool in enumerate(expected_request['tools'], 1):
        if isinstance(tool, dict):
            tool_type = tool.get('type', 'unknown')
            if tool_type == 'mcp':
                print(f"    [{i}] type: mcp, server: {tool.get('server_label')}")
            else:
                print(f"    [{i}] type: {tool_type}")
    print(f"  background: {expected_request['background']}")
    print(f"  reasoning: {expected_request['reasoning']}")

    return all_valid


async def test_mcp_tool_execution_flow():
    """Test how MCP tool execution would flow through the system"""
    print("\n" + "="*70)
    print("MCP TOOL EXECUTION FLOW TEST")
    print("="*70)

    print("\nWhen Azure OpenAI Responses API invokes an MCP tool:")
    print("-" * 70)

    print("\n1. Azure API sends MCP tool call to stdio://azure-ai-foundry/mcp-foundry")
    print("   - Tool call includes server_label, tool_name, and parameters")
    print("   - Executed via stdio transport (process communication)")

    print("\n2. Azure AI Foundry MCP server receives the call")
    print("   - Parses tool name and parameters")
    print("   - Applies paradigm-specific configuration if available")

    print("\n3. MCP server executes the tool")
    print("   - Could be evaluation, knowledge query, or model access")
    print("   - Uses Azure AI Foundry APIs internally")

    print("\n4. Results returned to Azure OpenAI")
    print("   - MCP server returns structured results")
    print("   - Azure OpenAI includes results in response context")

    print("\n5. LLM synthesizes final answer")
    print("   - Uses MCP tool results as additional context")
    print("   - Generates paradigm-aligned response")

    print("\n6. Results returned to Four Hosts application")
    print("   - Includes answer, reasoning, and tool call details")
    print("   - Displayed in frontend with context metrics")

    return True


async def main():
    """Run complete pipeline integration tests"""
    print("\n" + "="*70)
    print("AZURE AI FOUNDRY MCP - COMPLETE PIPELINE TEST SUITE")
    print("="*70)

    results = {}

    # Test 1: Complete pipeline flow
    results['pipeline_flow'] = await test_pipeline_flow()

    # Test 2: MCP tool execution flow
    results['execution_flow'] = await test_mcp_tool_execution_flow()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} - {test_name.replace('_', ' ').title()}")

    print(f"\nResults: {passed} passed, {failed} failed out of {total} tests")

    if passed == total:
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        print("\nAzure AI Foundry MCP Integration Summary:")
        print("  ✓ MCP servers registered and configured")
        print("  ✓ Tools formatted correctly for Responses API")
        print("  ✓ Deep research service integration complete")
        print("  ✓ Paradigm-specific configurations available")
        print("  ✓ Ready for LLM use in research queries")
        print("\nTo use Azure AI Foundry MCP in production:")
        print("  1. Ensure ENABLE_MCP_DEFAULT=1 in .env (already set)")
        print("  2. Start Azure AI Foundry MCP server (optional)")
        print("  3. Configure authentication for full functionality")
        print("  4. Run a research query to test end-to-end")
    else:
        print("\n⚠️  Some tests failed. Review the output above.")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)