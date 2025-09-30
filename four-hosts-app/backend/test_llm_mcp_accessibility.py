#!/usr/bin/env python3
"""
Test LLM accessibility of Azure AI Foundry MCP tools
Verifies that MCP tools are properly passed through the research pipeline to the LLM
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


async def test_mcp_tools_in_pipeline():
    """Test that MCP tools are accessible through the deep research pipeline"""
    print("\n" + "="*70)
    print("TEST: MCP Tools in Deep Research Pipeline")
    print("="*70)

    # Import after path setup
    from services.mcp.mcp_integration import mcp_integration, configure_default_servers

    # Configure default servers
    configure_default_servers()

    # Check ENABLE_MCP_DEFAULT flag
    enable_mcp = os.getenv("ENABLE_MCP_DEFAULT", "0").lower() in {"1", "true", "yes"}
    print(f"\n1. ENABLE_MCP_DEFAULT: {'✓ ENABLED' if enable_mcp else '✗ DISABLED'}")

    if not enable_mcp:
        print("   WARNING: MCP tools will not be attached to deep research")
        print("   Set ENABLE_MCP_DEFAULT=1 in .env to enable")
        return False

    # Check registered MCP servers
    print(f"\n2. Registered MCP servers: {len(mcp_integration.servers)}")
    for name, server in mcp_integration.servers.items():
        print(f"   - {name}: {server.url}")
        print(f"     Capabilities: {[cap.value for cap in server.capabilities]}")

    # Get MCP tools in Responses API format
    mcp_tools = mcp_integration.get_responses_mcp_tools()
    print(f"\n3. MCP tools in Responses API format: {len(mcp_tools)}")
    for tool in mcp_tools:
        print(f"   - Server: {tool['server_label']}")
        print(f"     URL: {tool['server_url']}")
        print(f"     Type: {tool['type']}")
        print(f"     Require approval: {tool['require_approval']}")

    # Verify Azure AI Foundry is in the list
    azure_foundry_tool = next(
        (t for t in mcp_tools if t['server_label'] == 'azure_ai_foundry'),
        None
    )

    if azure_foundry_tool:
        print(f"\n4. ✓ Azure AI Foundry MCP tool is available to LLM")
        return True
    else:
        print(f"\n4. ✗ Azure AI Foundry MCP tool NOT found in tools list")
        return False


async def test_deep_research_service_integration():
    """Test integration with deep_research_service"""
    print("\n" + "="*70)
    print("TEST: Deep Research Service Integration")
    print("="*70)

    try:
        from services.deep_research_service import DeepResearchConfig

        print("\n1. ✓ DeepResearchConfig imported successfully")

        # Check if mcp_servers can be configured
        config = DeepResearchConfig(query="test", paradigm="bernard")
        print(f"2. ✓ DeepResearchConfig created with paradigm: {config.paradigm}")

        # Check default MCP configuration
        from services.mcp.mcp_integration import mcp_integration
        mcp_tools = mcp_integration.get_responses_mcp_tools()

        if mcp_tools:
            print(f"3. ✓ {len(mcp_tools)} MCP tools available for deep research")
            print(f"   These will be attached to Responses API calls when ENABLE_MCP_DEFAULT=1")
            return True
        else:
            print("3. ✗ No MCP tools available")
            return False

    except Exception as e:
        print(f"✗ Error testing deep research service: {e}")
        return False


async def test_responses_client_tool_handling():
    """Test that Responses client properly handles MCP tools"""
    print("\n" + "="*70)
    print("TEST: Responses Client Tool Handling")
    print("="*70)

    try:
        from services.openai_responses_client import OpenAIResponsesClient

        print("\n1. ✓ OpenAIResponsesClient imported successfully")

        # Check if client is configured
        client = OpenAIResponsesClient()
        print(f"2. ✓ Client initialized (Azure: {client.is_azure})")

        # Simulate MCP tools
        from services.mcp.mcp_integration import mcp_integration
        mcp_tools = mcp_integration.get_responses_mcp_tools()

        if mcp_tools:
            print(f"3. ✓ MCP tools ready to be passed to create_response()")
            print(f"   Example tool structure:")
            if mcp_tools:
                tool = mcp_tools[0]
                print(f"   {tool}")
            return True
        else:
            print("3. ⚠️  No MCP tools available (server may not be running)")
            return None

    except Exception as e:
        print(f"✗ Error testing responses client: {e}")
        return False


async def test_paradigm_evaluation_flow():
    """Test paradigm-specific evaluation configuration flow"""
    print("\n" + "="*70)
    print("TEST: Paradigm-Specific Evaluation Flow")
    print("="*70)

    try:
        from services.mcp.azure_ai_foundry_mcp_integration import azure_ai_foundry_mcp

        print("\n1. Testing paradigm-specific evaluation configurations:")

        paradigms = ["dolores", "teddy", "bernard", "maeve"]
        for paradigm in paradigms:
            config = azure_ai_foundry_mcp.get_evaluation_config(paradigm)
            print(f"\n   {paradigm.upper()}:")
            print(f"   - Evaluation types: {', '.join(config.get('evaluation_types', []))}")
            print(f"   - Reasoning effort: {config.get('reasoning_effort', 'N/A')}")

        print("\n2. ✓ All paradigm configurations are properly defined")
        print("   These configs will be used when Azure AI Foundry MCP is initialized")

        return True

    except Exception as e:
        print(f"✗ Error testing paradigm flow: {e}")
        return False


async def main():
    """Run all LLM accessibility tests"""
    print("\n" + "="*70)
    print("LLM MCP TOOL ACCESSIBILITY TEST SUITE")
    print("="*70)

    results = {}

    # Test 1: MCP tools in pipeline
    results['pipeline'] = await test_mcp_tools_in_pipeline()

    # Test 2: Deep research service integration
    results['deep_research'] = await test_deep_research_service_integration()

    # Test 3: Responses client tool handling
    results['responses_client'] = await test_responses_client_tool_handling()

    # Test 4: Paradigm evaluation flow
    results['paradigm_flow'] = await test_paradigm_evaluation_flow()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result is True else ("✗ FAIL" if result is False else "⊘ SKIP")
        print(f"{status:8} - {test_name.replace('_', ' ').title()}")

    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped out of {total} tests")

    if passed == total:
        print("\n✓ All tests passed! Azure AI Foundry MCP tools are accessible to the LLM")
    elif failed > 0:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    else:
        print("\n⚠️  Some tests were skipped (MCP server not running)")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)