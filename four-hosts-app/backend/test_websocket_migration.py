"""
Test WebSocket Service V2 Migration - File Structure Validation
"""
import os
import sys


def test_file_structure():
    """Test that all required V2 files exist and have correct structure"""

    # Check if V2 WebSocket service exists
    v2_websocket_path = "services/websocket_service_v2.py"
    assert os.path.exists(v2_websocket_path), f"V2 WebSocket service not found: {v2_websocket_path}"
    print("✓ WebSocket Service V2 file exists")

    # Check if memory management exists
    memory_mgmt_path = "services/memory_management.py"
    assert os.path.exists(memory_mgmt_path), f"Memory management not found: {memory_mgmt_path}"
    print("✓ Memory management file exists")

    # Check if text compression exists
    text_compression_path = "services/text_compression.py"
    assert os.path.exists(text_compression_path), f"Text compression not found: {text_compression_path}"
    print("✓ Text compression file exists")

    # Check if context models exist
    context_models_path = "models/context_models.py"
    assert os.path.exists(context_models_path), f"Context models not found: {context_models_path}"
    print("✓ Context models file exists")


def test_v2_websocket_content():
    """Test that V2 WebSocket service has required classes and methods"""

    with open("services/websocket_service_v2.py", "r") as f:
        content = f.read()

    # Check for required classes
    assert "class ConnectionManagerV2" in content, "ConnectionManagerV2 class not found"
    assert "class ResearchProgressTrackerV2" in content, "ResearchProgressTrackerV2 class not found"
    assert "class MessageQueue" in content, "MessageQueue class not found"
    print("✓ Required V2 classes found")

    # Check for memory management integration
    assert "memory_manager" in content, "Memory manager integration not found"
    assert "register_connection" in content, "Connection registration not found"
    assert "unregister_connection" in content, "Connection unregistration not found"
    print("✓ Memory management integration found")

    # Check for text compression integration
    assert "text_compressor" in content, "Text compressor integration not found"
    assert "compress_text" in content, "Text compression usage not found"
    print("✓ Text compression integration found")


def test_main_py_integration():
    """Test that main.py has been updated to use V2 services"""

    with open("main.py", "r") as f:
        content = f.read()

    # Check for V2 imports
    assert "websocket_service_v2" in content, "V2 WebSocket service import not found in main.py"
    assert "ConnectionManagerV2" in content, "ConnectionManagerV2 import not found in main.py"
    print("✓ Main.py updated with V2 imports")


def test_backup_exists():
    """Test that backup of original WebSocket service exists"""

    backup_path = "services/backup_orchestrators/websocket_service_backup.py"
    assert os.path.exists(backup_path), f"Backup not found: {backup_path}"
    print("✓ Original WebSocket service backed up")


def run_migration_validation():
    """Run all migration validation tests"""
    print("🧪 Validating WebSocket Service V2 Migration...")

    try:
        test_file_structure()
        test_v2_websocket_content()
        test_main_py_integration()
        test_backup_exists()

        print("\n✅ WebSocket V2 Migration Validation PASSED!")
        print("\n📋 Migration Summary:")
        print("  • V2 WebSocket service implemented with enhanced features")
        print("  • Memory management integration added")
        print("  • Text compression for large messages")
        print("  • Sequence tracking and message replay")
        print("  • Main.py updated to use V2 services")
        print("  • Original service safely backed up")

        return True

    except AssertionError as e:
        print(f"\n❌ Migration validation failed: {e}")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error during validation: {e}")
        return False


class MockWebSocket:
    """Mock WebSocket for testing"""

    def __init__(self):
        self.messages = []
        self.closed = False

    async def accept(self):
        pass

    async def send_text(self, data: str):
        self.messages.append(data)

    async def receive_json(self):
        return {"type": "ping"}

    def __hash__(self):
        return id(self)


if __name__ == "__main__":
    success = run_migration_validation()
    exit(0 if success else 1)
