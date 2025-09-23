"""
Test for WebSocket disconnected variable initialization regression
Ensures that the disconnected variable is always initialized before use
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from services.websocket_service import ConnectionManager, WSMessage, WSEventType


@pytest.mark.asyncio
async def test_send_to_user_with_no_active_connections():
    """Test send_to_user when user has no active connections

    This test ensures that the disconnected variable is properly initialized
    even when the user_id is not in active_connections, preventing the
    "cannot access local variable 'disconnected'" error.
    """
    # Create a connection manager
    conn_manager = ConnectionManager()

    # Create a test message
    test_message = WSMessage(
        type=WSEventType.SYSTEM_NOTIFICATION,
        data={"message": "Test message"}
    )

    # Call send_to_user with a user_id that has no active connections
    # This should not raise an error
    await conn_manager.send_to_user("nonexistent_user", test_message)

    # If we get here without an exception, the test passes
    assert True


@pytest.mark.asyncio
async def test_send_to_user_with_disconnected_websockets():
    """Test send_to_user properly cleans up disconnected websockets"""
    # Create a connection manager
    conn_manager = ConnectionManager()

    # Create mock websockets
    ws1 = MagicMock()
    ws2 = MagicMock()

    # Set up active connections
    user_id = "test_user"
    conn_manager.active_connections[user_id] = {ws1, ws2}

    # Create a test message
    test_message = WSMessage(
        type=WSEventType.SYSTEM_NOTIFICATION,
        data={"message": "Test message"}
    )

    # Mock _send_json to simulate one websocket failing
    with patch.object(conn_manager, '_send_json') as mock_send:
        # First call succeeds, second call raises exception
        mock_send.side_effect = [None, Exception("Connection lost")]

        # Mock disconnect method
        with patch.object(conn_manager, 'disconnect') as mock_disconnect:
            mock_disconnect.return_value = AsyncMock()

            # Call send_to_user
            await conn_manager.send_to_user(user_id, test_message)

            # Verify that disconnect was called for the failed websocket
            mock_disconnect.assert_called_once()


@pytest.mark.asyncio
async def test_broadcast_to_research_with_no_subscribers():
    """Test broadcast_to_research when research has no subscribers

    This test ensures that the disconnected variable is properly initialized
    even when the research_id has no subscribers.
    """
    # Create a connection manager
    conn_manager = ConnectionManager()

    # Create a test message
    test_message = WSMessage(
        type=WSEventType.RESEARCH_PROGRESS,
        data={"research_id": "test_research", "progress": 50}
    )

    # Call broadcast_to_research with a research_id that has no subscribers
    # This should not raise an error
    await conn_manager.broadcast_to_research("test_research", test_message)

    # Verify that message was added to history
    assert "test_research" in conn_manager.message_history
    assert len(conn_manager.message_history["test_research"]) == 1


@pytest.mark.asyncio
async def test_broadcast_to_research_with_disconnected_websockets():
    """Test broadcast_to_research properly cleans up disconnected websockets"""
    # Create a connection manager
    conn_manager = ConnectionManager()

    # Create mock websockets
    ws1 = MagicMock()
    ws2 = MagicMock()
    ws3 = MagicMock()

    # Set up research subscriptions
    research_id = "test_research"
    conn_manager.research_subscriptions[research_id] = {ws1, ws2, ws3}

    # Create a test message
    test_message = WSMessage(
        type=WSEventType.RESEARCH_PROGRESS,
        data={"research_id": research_id, "progress": 75}
    )

    # Mock _send_json to simulate some websockets failing
    with patch.object(conn_manager, '_send_json') as mock_send:
        # First call succeeds, second fails, third succeeds
        mock_send.side_effect = [None, Exception("Connection lost"), None]

        # Mock disconnect method
        with patch.object(conn_manager, 'disconnect') as mock_disconnect:
            mock_disconnect.return_value = AsyncMock()

            # Call broadcast_to_research
            await conn_manager.broadcast_to_research(research_id, test_message)

            # Verify that disconnect was called for the failed websocket
            mock_disconnect.assert_called_once()

            # Verify that message was added to history
            assert research_id in conn_manager.message_history
            assert len(conn_manager.message_history[research_id]) == 1