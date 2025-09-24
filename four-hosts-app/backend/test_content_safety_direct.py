#!/usr/bin/env python3
"""
Direct test of Azure Content Safety API using aiohttp
"""

import asyncio
import aiohttp
import json
import os


async def test_direct_api():
    """Test the Content Safety API directly"""

    endpoint = "https://4hosts.cognitiveservices.azure.com/"
    api_key = "b5b138156d6a469086e9eb5fedd11413"

    url = f"{endpoint.rstrip('/')}/contentsafety/text:detectGroundedness?api-version=2024-09-15-preview"

    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Content-Type": "application/json"
    }

    request_body = {
        "domain": "Generic",
        "task": "Summarization",
        "text": "The patient's name is John.",
        "groundingSources": ["Medical record: Patient Jane, age 45"],
        "reasoning": False
    }

    print(f"URL: {url}")
    print(f"Headers: {headers}")
    print(f"Body: {json.dumps(request_body, indent=2)}")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, headers=headers, json=request_body, timeout=30) as response:
                print(f"\nResponse Status: {response.status}")
                response_text = await response.text()
                print(f"Response Text: {response_text}")

                if response.status == 200:
                    result = await response.json()
                    print("\n✓ SUCCESS!")
                    print(json.dumps(result, indent=2))
                else:
                    print("\n✗ Failed with status:", response.status)
                    print("Response:", response_text)

        except Exception as e:
            print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_direct_api())