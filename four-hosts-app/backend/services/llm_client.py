"""
LLM Client for Four Hosts Research Application
Supports Azure OpenAI and other LLM providers
"""
import os
import logging
from typing import Dict, List, Any, Optional
from openai import AsyncAzureOpenAI, AsyncOpenAI
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClient:
    """Client for interacting with various LLM providers"""

    def __init__(self):
        self.azure_client = None
        self.openai_client = None
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize LLM clients based on environment variables"""
        # Azure OpenAI configuration
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-07-01-preview")

        if azure_endpoint and azure_api_key:
            self.azure_client = AsyncAzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=azure_api_version
            )
            logger.info("✓ Azure OpenAI client initialized")

        # OpenAI configuration
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=openai_api_key)
            logger.info("✓ OpenAI client initialized")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_completion(self,
                                prompt: str,
                                model: str = "gpt-4o-mini",
                                max_tokens: int = 2000,
                                temperature: float = 0.7,
                                top_p: float = 0.9,
                                frequency_penalty: float = 0.0,
                                presence_penalty: float = 0.0,
                                paradigm: str = "bernard") -> str:
        """
        Generate completion using available LLM provider

        Args:
            prompt: The prompt to send to the LLM
            model: The model to use for generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty (-2.0 to 2.0)
            presence_penalty: Presence penalty (-2.0 to 2.0)
            paradigm: Paradigm for model selection
        """
        # Select appropriate model based on paradigm
        model_mapping = {
            "dolores": "gpt-4o",  # More creative for revolutionary paradigm
            "teddy": "gpt-4o-mini",  # Balanced for supportive paradigm
            "bernard": "gpt-4o",  # More capable for analytical paradigm
            "maeve": "gpt-4o-mini"  # Efficient for strategic paradigm
        }

        selected_model = model_mapping.get(paradigm, model)

        # Try Azure OpenAI first
        if self.azure_client:
            try:
                # Map model names to Azure deployment names
                azure_model_mapping = {
                    "gpt-4o": os.getenv("AZURE_GPT4_DEPLOYMENT_NAME", "gpt-4o"),
                    "gpt-4o-mini": os.getenv("AZURE_GPT4_MINI_DEPLOYMENT_NAME", "gpt-4o-mini")
                }

                azure_deployment = azure_model_mapping.get(selected_model, selected_model)

                response = await self.azure_client.chat.completions.create(
                    model=azure_deployment,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt(paradigm)},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty
                )

                logger.info(f"✓ Generated completion using Azure OpenAI ({selected_model})")
                return response.choices[0].message.content.strip()

            except Exception as e:
                logger.warning(f"Azure OpenAI request failed: {str(e)}")

        # Fallback to OpenAI
        if self.openai_client:
            try:
                response = await self.openai_client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt(paradigm)},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty
                )

                logger.info(f"✓ Generated completion using OpenAI ({selected_model})")
                return response.choices[0].message.content.strip()

            except Exception as e:
                logger.error(f"OpenAI request failed: {str(e)}")
                raise

        # If no LLM clients are available, raise an error
        raise Exception("No LLM clients available. Please configure API keys.")

    def _get_system_prompt(self, paradigm: str) -> str:
        """Get paradigm-specific system prompt"""
        system_prompts = {
            "dolores": "You are a revolutionary truth-seeker exposing systemic injustices. Focus on revealing hidden power structures and systemic failures. Use emotional, impactful language that moves people to action. Cite specific examples and evidence of wrongdoing. Do not pull punches - name names and expose the guilty.",
            "teddy": "You are a compassionate caregiver focused on helping and protecting others. Show deep understanding and empathy for those affected. Provide comprehensive resources and support options. Share uplifting stories of help and recovery. Offer practical, actionable ways to provide care. Use warm, supportive language that builds hope and connection.",
            "bernard": "You are an analytical researcher focused on empirical evidence. Present statistical findings and quantitative data. Identify patterns, correlations, and causal relationships. Maintain scientific objectivity and acknowledge limitations. Provide evidence-based conclusions and recommendations. Use precise, academic language with proper citations.",
            "maeve": "You are a strategic advisor focused on competitive advantage. Identify opportunities for competitive differentiation. Provide specific tactical recommendations. Optimize resource allocation for maximum impact. Define clear success metrics and milestones. Use crisp, action-oriented language focused on results. Emphasize practical implementation over theory."
        }

        return system_prompts.get(paradigm, "You are a helpful AI assistant.")

    async def generate_paradigm_content(self,
                                      prompt: str,
                                      paradigm: str,
                                      max_tokens: int = 2000) -> str:
        """
        Generate paradigm-specific content

        Args:
            prompt: The prompt for content generation
            paradigm: The paradigm to use (dolores, teddy, bernard, maeve)
            max_tokens: Maximum tokens to generate
        """
        # Paradigm-specific parameters
        paradigm_params = {
            "dolores": {"temperature": 0.8, "top_p": 0.9},
            "teddy": {"temperature": 0.6, "top_p": 0.8},
            "bernard": {"temperature": 0.4, "top_p": 0.7},
            "maeve": {"temperature": 0.6, "top_p": 0.8}
        }

        params = paradigm_params.get(paradigm, {"temperature": 0.7, "top_p": 0.9})

        return await self.generate_completion(
            prompt=prompt,
            paradigm=paradigm,
            max_tokens=max_tokens,
            temperature=params["temperature"],
            top_p=params["top_p"]
        )

# Global LLM client instance
llm_client = LLMClient()

async def initialize_llm_client():
    """Initialize the LLM client"""
    logger.info("Initializing LLM client...")
    # The client is already initialized in the constructor
    logger.info("✓ LLM client initialized")
    return True
