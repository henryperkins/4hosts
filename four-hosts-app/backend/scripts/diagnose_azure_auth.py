#!/usr/bin/env python3
"""
Azure OpenAI and Content Safety Authorization Diagnostic Script

This script helps diagnose and fix authorization issues with Azure OpenAI
and Content Safety integration.
"""

import os
import sys
import json
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, Any, Optional, Tuple


class AzureAuthDiagnostic:
    """Diagnose Azure OpenAI and Content Safety authentication issues"""

    def __init__(self):
        # Load environment variables (support both old and new naming)
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        self.openai_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        self.openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

        # Support both naming conventions
        self.cs_endpoint = (os.getenv("AZURE_CS_ENDPOINT") or
                           os.getenv("CONTENT_SAFETY_ENDPOINT") or "").strip()
        self.cs_key = (os.getenv("AZURE_CS_KEY") or
                      os.getenv("CONTENT_SAFETY_API_KEY") or "")

        self.results = []

    def print_header(self, title: str):
        """Print a formatted header"""
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

    def check_result(self, test_name: str, success: bool, details: str = ""):
        """Record and print test result"""
        status = "✓ PASS" if success else "✗ FAIL"
        self.results.append((test_name, success, details))
        print(f"{status}: {test_name}")
        if details:
            print(f"   Details: {details}")

    def validate_endpoints(self):
        """Validate endpoint formats"""
        self.print_header("Endpoint Validation")

        # Check OpenAI endpoint
        if not self.openai_endpoint:
            self.check_result("Azure OpenAI endpoint configured", False, "AZURE_OPENAI_ENDPOINT not set")
        elif not self.openai_endpoint.startswith("https://"):
            self.check_result("Azure OpenAI endpoint format", False, f"Must start with https:// (got: {self.openai_endpoint})")
        elif ".openai.azure.com" not in self.openai_endpoint:
            self.check_result("Azure OpenAI endpoint format", False, f"Should contain .openai.azure.com (got: {self.openai_endpoint})")
        else:
            self.check_result("Azure OpenAI endpoint format", True, self.openai_endpoint)

        # Check Content Safety endpoint
        if not self.cs_endpoint:
            self.check_result("Content Safety endpoint configured", False, "CONTENT_SAFETY_ENDPOINT not set")
        elif not self.cs_endpoint.startswith("https://"):
            self.check_result("Content Safety endpoint format", False, f"Must start with https:// (got: {self.cs_endpoint})")
        elif ".cognitiveservices.azure.com" not in self.cs_endpoint and ".api.cognitive.microsoft.com" not in self.cs_endpoint:
            self.check_result("Content Safety endpoint format", False, f"Unexpected format: {self.cs_endpoint}")
        else:
            self.check_result("Content Safety endpoint format", True, self.cs_endpoint)

    def validate_credentials(self):
        """Validate API keys are present"""
        self.print_header("Credentials Check")

        if self.openai_key:
            self.check_result("Azure OpenAI API key present", True, f"Key length: {len(self.openai_key)}")
        else:
            self.check_result("Azure OpenAI API key present", False, "AZURE_OPENAI_API_KEY not set")

        if self.cs_key:
            self.check_result("Content Safety API key present", True, f"Key length: {len(self.cs_key)}")
        else:
            self.check_result("Content Safety API key present", False, "CONTENT_SAFETY_API_KEY not set")

    async def test_openai_connection(self) -> Tuple[bool, str]:
        """Test Azure OpenAI API connection"""
        if not self.openai_endpoint or not self.openai_key:
            return False, "Missing endpoint or key"

        url = f"{self.openai_endpoint.rstrip('/')}/openai/deployments/{self.openai_deployment}/chat/completions?api-version={self.openai_api_version}"

        headers = {
            "api-key": self.openai_key,
            "Content-Type": "application/json"
        }

        payload = {
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 5,
            "temperature": 0
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=10) as response:
                    response_text = await response.text()

                    if response.status == 200:
                        return True, "Successfully connected"
                    elif response.status == 401:
                        return False, f"Authentication failed (401). Check API key. Response: {response_text[:200]}"
                    elif response.status == 403:
                        return False, f"Authorization failed (403). Check RBAC roles. Response: {response_text[:200]}"
                    elif response.status == 404:
                        return False, f"Deployment '{self.openai_deployment}' not found (404). Check deployment name."
                    else:
                        return False, f"HTTP {response.status}: {response_text[:200]}"

        except aiohttp.ClientError as e:
            return False, f"Connection error: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

    async def test_content_safety_connection(self) -> Tuple[bool, str]:
        """Test Content Safety API connection"""
        if not self.cs_endpoint or not self.cs_key:
            return False, "Missing endpoint or key"

        url = f"{self.cs_endpoint.rstrip('/')}/contentsafety/text:detectGroundedness?api-version=2024-09-15-preview"

        headers = {
            "Ocp-Apim-Subscription-Key": self.cs_key,
            "Content-Type": "application/json"
        }

        payload = {
            "domain": "Generic",
            "task": "Summarization",
            "text": "Test text",
            "groundingSources": ["Test source"],
            "reasoning": False
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=10) as response:
                    response_text = await response.text()

                    if response.status == 200:
                        return True, "Successfully connected"
                    elif response.status == 401:
                        return False, f"Authentication failed (401). Check API key. Response: {response_text[:200]}"
                    elif response.status == 403:
                        return False, f"Authorization failed (403). Response: {response_text[:200]}"
                    elif response.status == 404:
                        return False, f"Endpoint not found (404). Check URL format."
                    else:
                        return False, f"HTTP {response.status}: {response_text[:200]}"

        except aiohttp.ClientError as e:
            return False, f"Connection error: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

    async def test_content_safety_with_reasoning(self) -> Tuple[bool, str]:
        """Test Content Safety with reasoning (requires Azure OpenAI)"""
        if not self.cs_endpoint or not self.cs_key:
            return False, "Missing Content Safety credentials"

        if not self.openai_endpoint or not self.openai_deployment:
            return False, "Missing Azure OpenAI configuration for reasoning"

        url = f"{self.cs_endpoint.rstrip('/')}/contentsafety/text:detectGroundedness?api-version=2024-09-15-preview"

        headers = {
            "Ocp-Apim-Subscription-Key": self.cs_key,
            "Content-Type": "application/json"
        }

        payload = {
            "domain": "Generic",
            "task": "Summarization",
            "text": "The patient is John",
            "groundingSources": ["The patient is Jane"],
            "reasoning": True,
            "llmResource": {
                "resourceType": "AzureOpenAI",
                "azureOpenAIEndpoint": self.openai_endpoint.rstrip('/'),
                "azureOpenAIDeploymentName": self.openai_deployment
            }
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=30) as response:
                    response_text = await response.text()

                    if response.status == 200:
                        result = json.loads(response_text)
                        if "reason" in str(result):
                            return True, "Reasoning feature working"
                        else:
                            return True, "Connected but no reasoning in response"
                    elif response.status == 401 or response.status == 403:
                        return False, f"Authorization failed ({response.status}). Content Safety needs access to OpenAI. Response: {response_text[:200]}"
                    else:
                        return False, f"HTTP {response.status}: {response_text[:200]}"

        except Exception as e:
            return False, f"Error: {str(e)}"

    async def run_diagnostics(self):
        """Run all diagnostic tests"""
        print("\n" + "=" * 60)
        print("  Azure Authentication Diagnostic Report")
        print(f"  Generated: {datetime.now().isoformat()}")
        print("=" * 60)

        # 1. Validate configuration
        self.validate_endpoints()
        self.validate_credentials()

        # 2. Test connections
        self.print_header("Connection Tests")

        # Test OpenAI
        success, details = await self.test_openai_connection()
        self.check_result("Azure OpenAI API connection", success, details)

        # Test Content Safety
        success, details = await self.test_content_safety_connection()
        self.check_result("Content Safety API connection", success, details)

        # Test Content Safety with reasoning (if both services work)
        if any(r[1] for r in self.results if "Azure OpenAI API connection" in r[0]) and \
           any(r[1] for r in self.results if "Content Safety API connection" in r[0]):
            self.print_header("Advanced Features")
            success, details = await self.test_content_safety_with_reasoning()
            self.check_result("Content Safety with reasoning", success, details)

        # 3. Print summary
        self.print_summary()

    def print_summary(self):
        """Print diagnostic summary and recommendations"""
        self.print_header("Summary")

        passed = sum(1 for _, success, _ in self.results if success)
        failed = len(self.results) - passed

        print(f"\nTests passed: {passed}/{len(self.results)}")
        print(f"Tests failed: {failed}/{len(self.results)}")

        if failed > 0:
            self.print_header("Recommendations")

            # Check for common issues
            for test_name, success, details in self.results:
                if not success:
                    if "401" in details:
                        print("\n⚠️  Authentication Issue Detected:")
                        print("   1. Verify API keys are correct")
                        print("   2. Check keys haven't been regenerated")
                        print("   3. Ensure keys are from the correct resource")
                        break
                    elif "403" in details:
                        print("\n⚠️  Authorization Issue Detected:")
                        print("   1. Check RBAC roles are assigned")
                        print("   2. For Managed Identity, ensure Content Safety has 'Cognitive Services OpenAI User' role")
                        print("   3. Wait 5-15 minutes for role propagation if recently assigned")
                        print("\n   Run this command to check roles:")
                        print("   az role assignment list --scope <resource-id> --query \"[].{role:roleDefinitionName,principalId:principalId}\" -o table")
                        break
                    elif "404" in details:
                        print("\n⚠️  Resource Not Found:")
                        print("   1. Verify deployment name matches exactly")
                        print("   2. Check endpoint URL format")
                        print("   3. Ensure resource exists in the correct region")
                        break
                    elif "Connection error" in details:
                        print("\n⚠️  Network Issue:")
                        print("   1. Check network connectivity")
                        print("   2. Verify firewall rules")
                        print("   3. If using private endpoints, ensure DNS resolution")
                        break

            # Specific recommendations for reasoning feature
            if any("reasoning" in r[0].lower() and not r[1] for r in self.results):
                print("\n📋 To enable Content Safety reasoning with Azure OpenAI:")
                print("   1. Enable Managed Identity on Content Safety resource")
                print("   2. Grant Content Safety MI the 'Cognitive Services OpenAI User' role:")
                print("      az role assignment create \\")
                print("        --assignee <content-safety-mi-principal-id> \\")
                print("        --role \"Cognitive Services OpenAI User\" \\")
                print("        --scope <openai-resource-id>")
                print("   3. Wait 5-15 minutes for propagation")
        else:
            print("\n✅ All tests passed! Your Azure services are properly configured.")


async def main():
    """Main entry point"""
    diagnostic = AzureAuthDiagnostic()
    await diagnostic.run_diagnostics()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDiagnostic interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)