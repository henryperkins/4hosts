#!/usr/bin/env python3
"""
Test the Brave MCP integration with fallback functionality
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.brave_mcp_integration import brave_mcp, initialize_brave_mcp, cleanup_brave_mcp
from services.mcp_integration import mcp_integration

# Load environment variables
load_dotenv()


async def test_mcp_integration():
    """Test the Brave MCP integration"""
    print("\n=== Testing Brave MCP Integration ===\n")
    
    # Initialize the integration
    print("1. Initializing Brave MCP...")
    success = await initialize_brave_mcp()
    print(f"   Initialization result: {success}")
    print(f"   Server registered: {brave_mcp.server_registered}")
    print(f"   API key configured: {brave_mcp.config.is_configured()}")
    
    if not brave_mcp.config.is_configured():
        print("\n❌ Brave API key not configured!")
        print("   Set BRAVE_API_KEY or BRAVE_SEARCH_API_KEY in your .env file")
        return
    
    # Check server health
    print("\n2. Checking MCP server health...")
    if "brave_search" in mcp_integration.servers:
        is_healthy = await mcp_integration.check_server_health("brave_search")
        print(f"   Server health: {is_healthy}")
    else:
        print("   Server not registered")
    
    # Test paradigm-aware search
    print("\n3. Testing paradigm-aware search...")
    test_queries = [
        ("Dolores", "corporate corruption whistleblowers 2024"),
        ("Teddy", "community support mental health resources"),
        ("Bernard", "machine learning research papers"),
        ("Maeve", "AI startup funding strategies 2024")
    ]
    
    for paradigm, query in test_queries:
        print(f"\n   Testing {paradigm} paradigm:")
        print(f"   Query: {query}")
        
        try:
            result = await brave_mcp.search_with_paradigm(
                query=query,
                paradigm=paradigm.lower()
            )
            
            print(f"   Results: {result.get('result_count', 0)} items")
            print(f"   Search method: {'MCP' if brave_mcp.server_registered and mcp_integration.is_server_healthy('brave_search') else 'Direct API'}")
            
            # Show first result
            if result.get("results"):
                first = result["results"][0]
                print(f"   First result: {first.get('title', 'No title')}")
                print(f"   URL: {first.get('url', 'No URL')}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    # Test direct API fallback
    print("\n\n4. Testing direct API fallback (simulating MCP failure)...")
    
    # Temporarily mark server as unhealthy
    if "brave_search" in mcp_integration._server_health:
        original_health = mcp_integration._server_health["brave_search"]
        mcp_integration._server_health["brave_search"] = False
        
        try:
            result = await brave_mcp.search_with_paradigm(
                query="test fallback mechanism",
                paradigm="bernard"
            )
            print(f"   Fallback successful! Got {result.get('result_count', 0)} results")
        except Exception as e:
            print(f"   ❌ Fallback failed: {str(e)}")
        finally:
            # Restore original health status
            mcp_integration._server_health["brave_search"] = original_health
    
    # Cleanup
    print("\n5. Cleaning up...")
    await cleanup_brave_mcp()
    print("   Cleanup complete")


async def test_mcp_server_connection():
    """Test direct connection to MCP server"""
    print("\n=== Testing Direct MCP Server Connection ===\n")
    
    import aiohttp
    
    mcp_url = os.getenv("BRAVE_MCP_URL", "http://localhost:8080")
    
    # Test health endpoint
    print(f"Testing health endpoint at {mcp_url}/health...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{mcp_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                print(f"Health check response: {response.status}")
                if response.status == 200:
                    content = await response.text()
                    print(f"Response: {content}")
                else:
                    print(f"Unexpected status: {response.status}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # Test MCP endpoint
    print(f"\nTesting MCP endpoint at {mcp_url}...")
    try:
        async with aiohttp.ClientSession() as session:
            # Send a basic MCP request
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            mcp_request = {
                "method": "tools.list",
                "params": {},
                "id": "test_connection"
            }
            
            async with session.post(
                mcp_url,
                json=mcp_request,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                print(f"MCP response status: {response.status}")
                print(f"Content-Type: {response.headers.get('content-type', 'Not specified')}")
                
                if response.status == 200:
                    try:
                        data = await response.json()
                        print(f"Response: {data}")
                    except:
                        text = await response.text()
                        print(f"Non-JSON response: {text[:200]}")
                else:
                    text = await response.text()
                    print(f"Error response: {text[:200]}")
                    
    except Exception as e:
        print(f"❌ MCP request failed: {e}")


if __name__ == "__main__":
    print("Starting Brave MCP Integration Tests")
    print("====================================")
    
    # First test direct server connection
    asyncio.run(test_mcp_server_connection())
    
    # Then test the integration
    asyncio.run(test_mcp_integration())
    
    print("\n✅ Tests completed!")