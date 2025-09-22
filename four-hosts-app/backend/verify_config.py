#!/usr/bin/env python3
"""
Verify that the updated configuration is loaded in the backend.
"""
import os
import sys
from dotenv import load_dotenv

# Reload environment variables
load_dotenv(override=True)

print("=" * 60)
print("CONFIGURATION VERIFICATION")
print("=" * 60)

# Check Google CSE
print("\n1. Google Custom Search Engine:")
google_key = os.getenv("GOOGLE_CSE_API_KEY")
google_cx = os.getenv("GOOGLE_CSE_CX")
print(f"   API Key: {'✅ Set' if google_key else '❌ Not set'} ({google_key[:10]}...)" if google_key else "   API Key: ❌ Not set")
print(f"   CX (Search Engine ID): {google_cx if google_cx else '❌ Not set'}")
print(f"   Expected CX: 356dad27485a547ad")
print(f"   Match: {'✅ Yes' if google_cx == '356dad27485a547ad' else '❌ No'}")

# Check timeouts
print("\n2. Timeout Configuration:")
timeouts = {
    "SEARCH_TASK_TIMEOUT_SEC": "60",
    "SEARCH_PROVIDER_TIMEOUT_SEC": "30",
    "SEARCH_PER_PROVIDER_TIMEOUT_SEC": "20"
}

for key, expected in timeouts.items():
    actual = os.getenv(key)
    match = "✅" if actual == expected else "⚠️"
    print(f"   {key}: {actual} (expected: {expected}) {match}")

# Check Brave
print("\n3. Brave Search:")
brave_key = os.getenv("BRAVE_SEARCH_API_KEY")
print(f"   API Key: {'✅ Set' if brave_key else '❌ Not set'}")
print(f"   Disabled: {os.getenv('SEARCH_DISABLE_BRAVE', '0')}")

print("\n" + "=" * 60)
print("RESTART REQUIRED?")
print("=" * 60)

# Check if all critical settings are correct
all_good = True

if google_cx != "356dad27485a547ad":
    print("❌ Google CSE CX doesn't match - restart needed!")
    all_good = False

if os.getenv("SEARCH_TASK_TIMEOUT_SEC") != "60":
    print("❌ Search timeout not updated - restart needed!")
    all_good = False

if all_good:
    print("✅ Configuration looks good!")
    print("\nIf the backend is still having issues, restart it with:")
else:
    print("\n⚠️ Backend restart needed to apply new configuration!")
    print("\nTo restart the backend:")

print("   1. Find the process: ps aux | grep uvicorn")
print("   2. Kill it: kill -9 <PID>")
print("   3. Start it again: cd /home/azureuser/4hosts/four-hosts-app/backend")
print("                      uvicorn main_new:app --reload --port 8001")