"""
Unified OpenAI Client Manager
-----------------------------
Centralized management of OpenAI/Azure OpenAI clients to eliminate duplication
and ensure consistent configuration across the application.
"""

import os
from typing import Optional, TYPE_CHECKING
from functools import lru_cache

import structlog
from openai import AsyncOpenAI

if TYPE_CHECKING:
    from services.clients.openai_responses_client import OpenAIResponsesClient

logger = structlog.get_logger(__name__)


class OpenAIClientManager:
    """Centralized OpenAI/Azure client management"""

    def __init__(self):
        self._chat_client: Optional[AsyncOpenAI] = None
        self._responses_client: Optional['OpenAIResponsesClient'] = None
        self._initialized = False
        self._is_azure = False
        self._azure_endpoint: Optional[str] = None
        self._azure_api_version: str = "preview"
        self._azure_deployment: Optional[str] = None

    def _detect_configuration(self) -> None:
        """Detect and validate OpenAI/Azure configuration"""
        # Check for Azure configuration first (preferred)
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self._azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "preview")

        if azure_key and azure_endpoint and azure_deployment:
            self._is_azure = True
            self._azure_endpoint = azure_endpoint.rstrip("/")
            self._azure_deployment = azure_deployment
            logger.info(
                "Azure OpenAI configuration detected",
                endpoint=self._azure_endpoint,
                deployment=azure_deployment,
                api_version=self._azure_api_version
            )
        else:
            # Fall back to OpenAI Cloud
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key:
                raise RuntimeError(
                    "No OpenAI configuration found. Set either "
                    "AZURE_OPENAI_* or OPENAI_API_KEY environment variables"
                )

            self._is_azure = False
            logger.info("OpenAI Cloud configuration detected")

    def _create_chat_client(self) -> AsyncOpenAI:
        """Create AsyncOpenAI client for chat completions"""
        if self._is_azure:
            azure_key = os.getenv("AZURE_OPENAI_API_KEY")
            # Use OpenAI v1-compatible path with explicit api-version
            azure_base = f"{self._azure_endpoint}/openai/v1"
            if self._azure_api_version:
                azure_base = f"{azure_base}?api-version={self._azure_api_version}"

            return AsyncOpenAI(
                api_key=azure_key,
                base_url=azure_base,
                timeout=120  # Standardized chat timeout
            )
        else:
            openai_key = os.getenv("OPENAI_API_KEY")
            return AsyncOpenAI(
                api_key=openai_key,
                timeout=120  # Standardized chat timeout
            )

    def _create_responses_client(self) -> 'OpenAIResponsesClient':
        """Create OpenAIResponsesClient for responses API"""
        # Import here to avoid circular imports
        from services.clients.openai_responses_client import OpenAIResponsesClient
        return OpenAIResponsesClient(client_manager=self)

    def initialize(self) -> None:
        """Initialize all clients"""
        if self._initialized:
            return

        try:
            self._detect_configuration()

            # Create chat client
            self._chat_client = self._create_chat_client()

            # Create responses client
            self._responses_client = self._create_responses_client()

            self._initialized = True
            logger.info(
                "✓ OpenAI Client Manager initialized successfully",
                client_type="azure" if self._is_azure else "openai"
            )

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI Client Manager: {e}")
            raise

    def get_chat_client(self) -> AsyncOpenAI:
        """Get the chat completions client"""
        if not self._initialized:
            self.initialize()
        if self._chat_client is None:
            raise RuntimeError("Chat client not initialized")
        return self._chat_client

    def get_responses_client(self) -> 'OpenAIResponsesClient':
        """Get the responses API client"""
        if not self._initialized:
            self.initialize()
        if self._responses_client is None:
            raise RuntimeError("Responses client not initialized")
        return self._responses_client

    def is_azure_configured(self) -> bool:
        """Check if Azure OpenAI is configured"""
        if not self._initialized:
            self._detect_configuration()
        return self._is_azure

    def get_deployment_name(self, model: str) -> str:
        """Get deployment name for Azure or model name for OpenAI"""
        if self.is_azure_configured():
            return self._azure_deployment or model
        return model

    def get_base_url(self) -> str:
        """Get the base URL for API calls"""
        if self.is_azure_configured():
            return f"{self._azure_endpoint}/openai/v1"
        return "https://api.openai.com/v1"

    def get_api_key(self) -> str:
        """Get the appropriate API key"""
        if self.is_azure_configured():
            key = os.getenv("AZURE_OPENAI_API_KEY")
            if not key:
                raise RuntimeError("AZURE_OPENAI_API_KEY not set")
            return key
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        return key

    def get_azure_api_version(self) -> str:
        """Get Azure API version"""
        return self._azure_api_version

    def is_initialized(self) -> bool:
        """Check if the manager is initialized"""
        return self._initialized

    def get_health_status(self) -> dict:
        """Get health status of all clients"""
        return {
            "initialized": self._initialized,
            "is_azure": self._is_azure,
            "azure_endpoint": self._azure_endpoint,
            "azure_deployment": self._azure_deployment,
            "chat_client_available": self._chat_client is not None,
            "responses_client_available": self._responses_client is not None,
        }


# Global singleton instance
@lru_cache(maxsize=1)
def get_client_manager() -> OpenAIClientManager:
    """Get the singleton OpenAI Client Manager instance"""
    return OpenAIClientManager()
