"""
Azure AI Foundry Authentication Module
Provides DefaultAzureCredential support for Azure AI Foundry MCP integration
"""

import os
import structlog
from typing import Optional, Dict, Any

logger = structlog.get_logger(__name__)


class AzureAIFoundryAuth:
    """
    Manages authentication for Azure AI Foundry services.
    Supports multiple authentication methods in priority order:
    1. Service Principal (CLIENT_ID + CLIENT_SECRET)
    2. Azure CLI (already logged in)
    3. Managed Identity (for production deployments)
    """

    def __init__(self):
        self._credential = None
        self._auth_method = None

    def get_credential(self):
        """
        Get Azure credential using DefaultAzureCredential.
        This supports multiple authentication methods automatically.
        """
        if self._credential is not None:
            return self._credential

        try:
            from azure.identity import DefaultAzureCredential

            # Create credential with all available auth methods
            self._credential = DefaultAzureCredential(
                exclude_visual_studio_code_credential=True,  # Not needed in server
                exclude_shared_token_cache_credential=True,  # Not needed
            )

            # Try to get a token to verify authentication works
            try:
                # Use ARM scope to test
                token = self._credential.get_token("https://management.azure.com/.default")
                if token:
                    self._auth_method = self._detect_auth_method()
                    logger.info(
                        "Azure authentication successful",
                        method=self._auth_method,
                        tenant_id=os.getenv("AZURE_TENANT_ID", "detected")
                    )
                    return self._credential
            except Exception as token_error:
                logger.warning(
                    "Could not verify Azure authentication",
                    error=str(token_error),
                    hint="Run 'az login' or set service principal credentials"
                )
                return None

        except ImportError:
            logger.warning(
                "azure-identity package not installed",
                hint="Install with: pip install azure-identity"
            )
            return None
        except Exception as e:
            logger.error(f"Failed to create Azure credential: {e}")
            return None

    def _detect_auth_method(self) -> str:
        """Detect which authentication method is being used"""
        # Check for service principal
        if os.getenv("AZURE_CLIENT_ID") and os.getenv("AZURE_CLIENT_SECRET"):
            return "service_principal"

        # Check for Azure CLI
        try:
            import subprocess
            result = subprocess.run(
                ["az", "account", "show"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return "azure_cli"
        except:
            pass

        # Check for managed identity (in Azure environment)
        if os.getenv("MSI_ENDPOINT"):
            return "managed_identity"

        return "unknown"

    def get_auth_headers(self, scope: str = "https://cognitiveservices.azure.com/.default") -> Dict[str, str]:
        """
        Get authentication headers for Azure AI Foundry API calls.

        Args:
            scope: OAuth scope for the token (default is Cognitive Services)

        Returns:
            Dictionary with Authorization header
        """
        credential = self.get_credential()
        if not credential:
            return {}

        try:
            token = credential.get_token(scope)
            return {
                "Authorization": f"Bearer {token.token}"
            }
        except Exception as e:
            logger.error(f"Failed to get authentication token: {e}")
            return {}

    def get_auth_info(self) -> Dict[str, Any]:
        """Get information about current authentication status"""
        return {
            "configured": self._credential is not None,
            "method": self._auth_method or "none",
            "tenant_id": os.getenv("AZURE_TENANT_ID"),
            "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID"),
            "has_service_principal": bool(
                os.getenv("AZURE_CLIENT_ID") and os.getenv("AZURE_CLIENT_SECRET")
            ),
        }


# Global auth instance
azure_foundry_auth = AzureAIFoundryAuth()