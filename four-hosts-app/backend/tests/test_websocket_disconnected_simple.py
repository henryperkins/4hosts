"""
Simple test to verify the disconnected variable initialization fix
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_disconnected_variable_initialization():
    """Test that verifies the fix for disconnected variable initialization

    This test simulates the scenario where the disconnected variable
    was not initialized before use, which caused the error:
    "cannot access local variable 'disconnected' where it is not associated with a value"
    """

    # This is a simple check to verify the code structure is correct
    # Read the websocket_service.py file to check the fix is in place
    websocket_service_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'services',
        'websocket_service.py'
    )

    with open(websocket_service_path, 'r') as f:
        content = f.read()

    # Check that send_to_user method initializes disconnected before the if statement
    send_to_user_section = content[content.find("async def send_to_user"):content.find("async def broadcast_to_research")]

    # Look for the pattern: disconnected = [] before the if statement
    lines = send_to_user_section.split('\n')
    found_disconnected_init = False
    found_if_statement = False

    for line in lines:
        if "disconnected = []" in line and not line.strip().startswith("#"):
            found_disconnected_init = True
        if "if user_id in self.active_connections:" in line:
            found_if_statement = True
            # disconnected should be initialized before this point
            assert found_disconnected_init, "disconnected variable must be initialized before the if statement"
            break

    assert found_if_statement, "Could not find the if statement in send_to_user method"
    assert found_disconnected_init, "Could not find disconnected = [] initialization"

    print("✓ send_to_user method properly initializes disconnected variable")

    # Check broadcast_to_research method similarly
    broadcast_section = content[content.find("async def broadcast_to_research"):]
    broadcast_section = broadcast_section[:broadcast_section.find("async def", 10) if broadcast_section.find("async def", 10) != -1 else len(broadcast_section)]

    lines = broadcast_section.split('\n')
    found_disconnected_init = False
    found_if_statement = False

    for line in lines:
        if "disconnected = []" in line and not line.strip().startswith("#"):
            found_disconnected_init = True
        if "if research_id in self.research_subscriptions:" in line:
            found_if_statement = True
            # disconnected should be initialized before this point
            assert found_disconnected_init, "disconnected variable must be initialized before the if statement in broadcast_to_research"
            break

    assert found_if_statement, "Could not find the if statement in broadcast_to_research method"
    assert found_disconnected_init, "Could not find disconnected = [] initialization in broadcast_to_research"

    print("✓ broadcast_to_research method properly initializes disconnected variable")
    print("\n✅ All checks passed - the disconnected variable initialization issue is fixed!")


if __name__ == "__main__":
    test_disconnected_variable_initialization()