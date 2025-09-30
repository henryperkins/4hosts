#!/usr/bin/env python3
"""
Test script to verify Azure AI Foundry MCP integration
Tests configuration, registration, tool discovery, and LLM accessibility
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

import structlog
from services.mcp.azure_ai_foundry_mcp_integration import (
    AzureAIFoundryMCPConfig,
    AzureAIFoundryMCPIntegration,
    initialize_azure_ai_foundry_mcp,
    azure_ai_foundry_mcp,
)
from services.mcp.mcp_integration import mcp_integration, configure_default_servers

logger = structlog.get_logger(__name__)


async def test_configuration():
    """Test 1: Configuration validation"""
    print("\n" + "="*60)
    print("TEST 1: Configuration Validation")
    print("="*60)

    config = AzureAIFoundryMCPConfig()

    print(f"✓ AI Project Endpoint: {config.ai_project_endpoint or 'NOT SET'}")
    print(f"✓ Project Name: {config.project_name or 'NOT SET'}")
    print(f"✓ MCP URL: {config.mcp_url}")
    print(f"✓ MCP Transport: {config.mcp_transport}")
    print(f"✓ MCP Host: {config.mcp_host}")
    print(f"✓ MCP Port: {config.mcp_port}")

    is_configured = config.is_configured()
    has_auth = config.has_authentication()

    print(f"\nConfiguration Status:")
    print(f"  - Is Configured: {'✓ YES' if is_configured else '✗ NO'}")
    print(f"  - Has Authentication: {'✓ YES' if has_auth else '✗ NO'}")

    if not is_configured:
        missing_required, missing_optional = config.get_missing_config()
        print(f"\n  Missing Required: {missing_required}")
        print(f"  Missing Optional: {missing_optional}")
        return False

    return True


async def test_mcp_registration():
    """Test 2: MCP server registration"""
    print("\n" + "="*60)
    print("TEST 2: MCP Server Registration")
    print("="*60)

    # Configure default servers (includes Azure AI Foundry if configured)
    configure_default_servers()

    # Check if Azure AI Foundry is registered
    if "azure_ai_foundry" in mcp_integration.servers:
        server = mcp_integration.servers["azure_ai_foundry"]
        print(f"✓ Azure AI Foundry MCP server registered")
        print(f"  - Name: {server.name}")
        print(f"  - URL: {server.url}")
        print(f"  - Capabilities: {[cap.value for cap in server.capabilities]}")
        print(f"  - Timeout: {server.timeout}s")
        return True
    else:
        print("✗ Azure AI Foundry MCP server NOT registered")
        print(f"  Registered servers: {list(mcp_integration.servers.keys())}")
        return False


async def test_initialization():
    """Test 3: Azure AI Foundry MCP initialization"""
    print("\n" + "="*60)
    print("TEST 3: Azure AI Foundry MCP Initialization")
    print("="*60)

    try:
        result = await initialize_azure_ai_foundry_mcp()

        if result:
            print("✓ Azure AI Foundry MCP initialized successfully")
            print(f"  - Server registered: {azure_ai_foundry_mcp.server_registered}")
            print(f"  - Available capabilities: {azure_ai_foundry_mcp.get_available_capabilities()}")
            return True
        else:
            print("✗ Azure AI Foundry MCP initialization failed")
            print("  This may be expected if the MCP server is not running")
            return False
    except Exception as e:
        print(f"✗ Azure AI Foundry MCP initialization error: {e}")
        return False


async def test_tool_discovery():
    """Test 4: Tool discovery"""
    print("\n" + "="*60)
    print("TEST 4: Tool Discovery")
    print("="*60)

    if "azure_ai_foundry" not in mcp_integration.servers:
        print("✗ Cannot test tool discovery - server not registered")
        return False

    try:
        tools = await mcp_integration.discover_tools("azure_ai_foundry")

        if tools:
            print(f"✓ Discovered {len(tools)} tools from Azure AI Foundry")
            for tool in tools[:5]:  # Show first 5 tools
                print(f"  - {tool.function['name']}: {tool.function.get('description', 'No description')[:60]}...")
            if len(tools) > 5:
                print(f"  ... and {len(tools) - 5} more tools")
            return True
        else:
            print("✗ No tools discovered from Azure AI Foundry")
            print("  This may mean the MCP server is not running or not responding")
            return False
    except Exception as e:
        print(f"✗ Tool discovery error: {e}")
        return False


async def test_paradigm_configs():
    """Test 5: Paradigm-specific configurations"""
    print("\n" + "="*60)
    print("TEST 5: Paradigm-Specific Configurations")
    print("="*60)

    paradigms = ["dolores", "teddy", "bernard", "maeve"]

    for paradigm in paradigms:
        config = azure_ai_foundry_mcp.get_evaluation_config(paradigm)
        print(f"\n{paradigm.upper()}:")
        print(f"  - Evaluation types: {config.get('evaluation_types', [])}")
        print(f"  - Reasoning effort: {config.get('reasoning_effort', 'N/A')}")
        if 'safety_settings' in config:
            safety = config['safety_settings']
            print(f"  - Safety settings: {list(safety.keys())}")

    return True


async def test_llm_tool_accessibility():
    """Test 6: LLM tool accessibility via get_all_tools"""
    print("\n" + "="*60)
    print("TEST 6: LLM Tool Accessibility")
    print("="*60)

    # Get all tools that would be available to LLM
    all_tools = mcp_integration.get_all_tools()

    print(f"Total tools available to LLM: {len(all_tools)}")

    # Filter Azure AI Foundry tools
    azure_tools = [t for t in all_tools if 'azure_ai_foundry' in t.function.get('name', '')]

    if azure_tools:
        print(f"✓ Azure AI Foundry tools accessible to LLM: {len(azure_tools)}")
        for tool in azure_tools[:3]:
            print(f"  - {tool.function['name']}")
        if len(azure_tools) > 3:
            print(f"  ... and {len(azure_tools) - 3} more")
        return True
    else:
        print("✗ No Azure AI Foundry tools accessible to LLM")
        print("  This may mean tool discovery hasn't been run yet")
        return False


async def test_responses_api_format():
    """Test 7: Responses API MCP tool format"""
    print("\n" + "="*60)
    print("TEST 7: Responses API MCP Tool Format")
    print("="*60)

    # Get tools in Responses API format
    responses_tools = mcp_integration.get_responses_mcp_tools()

    if responses_tools:
        print(f"✓ MCP tools in Responses API format: {len(responses_tools)}")
        for tool in responses_tools:
            print(f"  - {tool['server_label']}: {tool['server_url']}")
            print(f"    Type: {tool['type']}, Approval: {tool['require_approval']}")
        return True
    else:
        print("✗ No MCP tools in Responses API format")
        return False


async def test_research_orchestrator_integration():
    """Test 8: Research orchestrator integration"""
    print("\n" + "="*60)
    print("TEST 8: Research Orchestrator Integration")
    print("="*60)

    try:
        from services.research_orchestrator import UnifiedResearchOrchestrator

        print("✓ Research orchestrator can be imported")
        print("  Note: Full initialization requires database and other services")
        print("  Azure AI Foundry MCP initialization happens in orchestrator.__init__")
        return True
    except Exception as e:
        print(f"✗ Research orchestrator import error: {e}")
        return False


async def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("AZURE AI FOUNDRY MCP INTEGRATION TEST SUITE")
    print("="*70)

    results = {}

    # Test 1: Configuration
    results['configuration'] = await test_configuration()

    # Test 2: MCP Registration
    results['registration'] = await test_mcp_registration()

    # Test 3: Initialization
    results['initialization'] = await test_initialization()

    # Test 4: Tool Discovery (only if initialized)
    if results['initialization']:
        results['tool_discovery'] = await test_tool_discovery()
    else:
        print("\n" + "="*60)
        print("TEST 4: Tool Discovery - SKIPPED (initialization failed)")
        print("="*60)
        results['tool_discovery'] = None

    # Test 5: Paradigm Configs
    results['paradigm_configs'] = await test_paradigm_configs()

    # Test 6: LLM Tool Accessibility
    results['llm_tools'] = await test_llm_tool_accessibility()

    # Test 7: Responses API Format
    results['responses_api'] = await test_responses_api_format()

    # Test 8: Research Orchestrator Integration
    results['orchestrator'] = await test_research_orchestrator_integration()

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

    if failed > 0:
        print("\n⚠️  Some tests failed. This may be expected if:")
        print("   - Azure AI Foundry MCP server is not running")
        print("   - Authentication is not configured")
        print("   - Network connectivity issues")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)