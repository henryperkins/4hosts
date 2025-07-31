#!/usr/bin/env python3
"""
Test script to verify orchestrator initialization
"""

import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

async def test_initialization():
    """Test the unified orchestrator initialization"""
    
    print("Testing Unified Research Orchestrator initialization...")
    
    try:
        # Test import
        from services.unified_research_orchestrator import initialize_unified_orchestrator, unified_orchestrator
        print("✓ Import successful")
        
        # Test initialization
        result = await initialize_unified_orchestrator()
        print(f"✓ Initialization result: {result}")
        
        # Check capabilities
        capabilities = unified_orchestrator.get_capabilities()
        print("\nOrchestrator Capabilities:")
        print(f"- Mode: {capabilities.get('mode')}")
        print(f"- Brave MCP Enabled: {capabilities.get('brave_mcp')}")
        print(f"- Search APIs: {[api for api in capabilities.get('search_apis', []) if api]}")
        print(f"- Features: {capabilities.get('features')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_initialization())
    print(f"\nTest {'PASSED' if success else 'FAILED'}")