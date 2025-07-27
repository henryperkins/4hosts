"""
LLM Client for Four Hosts Research Application
Supports Azure OpenAI and other LLM providers
"""
import os
import logging
from typing import Dict, List, Any, Optional, Union, AsyncIterator
from openai import AsyncAzureOpenAI, AsyncOpenAI
import asyncio
import json
from enum import Enum
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResponseFormat(Enum):
    """Response format types"""
    TEXT = "text"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


class TruncationStrategy(Enum):
    """Truncation strategies for model responses"""
    AUTO = "auto"
    DISABLED = "disabled"


class LLMClient:
    """Client for interacting with various LLM providers with advanced Azure OpenAI support"""

    def __init__(self):
        self.azure_client = None
        self.openai_client = None
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize LLM clients based on environment variables"""
        # Azure OpenAI configuration
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError))
    )
    async def generate_completion(self,
                                prompt: str,
                                model: str = "gpt-4o-mini",
                                max_tokens: int = 2000,
                                temperature: float = 0.7,
                                top_p: float = 0.9,
                                frequency_penalty: float = 0.0,
                                presence_penalty: float = 0.0,
                                paradigm: str = "bernard",
                                response_format: Optional[Dict[str, Any]] = None,
                                tools: Optional[List[Dict[str, Any]]] = None,
                                tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
                                stream: bool = False,
                                json_schema: Optional[Dict[str, Any]] = None) -> Union[str, AsyncIterator[str]]:
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
            response_format: Response format configuration
            tools: List of tools the model may call
            tool_choice: How the model should select tools
            stream: Whether to stream the response
            json_schema: JSON schema for structured outputs
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

                # Prepare request parameters
                request_params = {
                    "model": azure_deployment,
                    "messages": [
                        {"role": "system", "content": self._get_system_prompt(paradigm)},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                    "stream": stream
                }

                # Add response format if specified
                if response_format:
                    request_params["response_format"] = response_format
                elif json_schema:
                    request_params["response_format"] = {
                        "type": "json_schema",
                        "json_schema": json_schema
                    }

                # Add tools if specified
                if tools:
                    request_params["tools"] = tools
                    if tool_choice:
                        request_params["tool_choice"] = tool_choice

                response = await self.azure_client.chat.completions.create(
                    **request_params
                )

                if stream:
                    return self._handle_stream_response(response)
                else:
                    logger.info(f"✓ Generated completion using Azure OpenAI ({selected_model})")
                    content = response.choices[0].message.content
                    return content.strip() if content else ""

            except Exception as e:
                logger.warning(f"Azure OpenAI request failed: {str(e)}")

        # Fallback to OpenAI
        if self.openai_client:
            try:
                # Prepare request parameters
                request_params = {
                    "model": selected_model,
                    "messages": [
                        {"role": "system", "content": self._get_system_prompt(paradigm)},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                    "stream": stream
                }

                # Add response format if specified
                if response_format:
                    request_params["response_format"] = response_format
                elif json_schema:
                    request_params["response_format"] = {
                        "type": "json_schema",
                        "json_schema": json_schema
                    }

                # Add tools if specified
                if tools:
                    request_params["tools"] = tools
                    if tool_choice:
                        request_params["tool_choice"] = tool_choice

                response = await self.openai_client.chat.completions.create(**request_params)

                if stream:
                    return self._handle_stream_response(response)
                else:
                    logger.info(f"✓ Generated completion using OpenAI ({selected_model})")
                    content = response.choices[0].message.content
                    return content.strip() if content else ""

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

    async def _handle_stream_response(self, response) -> AsyncIterator[str]:
        """Handle streaming response from LLM"""
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def generate_structured_output(self,
                                       prompt: str,
                                       schema: dict,
                                       model: str = "gpt-4o-mini",
                                       paradigm: str = "bernard") -> dict:
        """
        Generate structured output using JSON schema
        
        Args:
            prompt: The prompt for generation
            schema: JSON schema defining the expected output structure
            model: Model to use
            paradigm: Paradigm for system prompt
            
        Returns:
            Parsed JSON response matching the schema
        """
        response = await self.generate_completion(
            prompt=prompt,
            model=model,
            paradigm=paradigm,
            json_schema=schema,
            temperature=0.3,  # Lower temperature for structured outputs
            stream=False  # Ensure we don't stream for structured outputs
        )
        
        try:
            if isinstance(response, str):
                return json.loads(response)
            else:
                raise ValueError("Expected string response for structured output")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse structured output: {e}")
            raise

    async def generate_with_tools(self,
                                prompt: str,
                                tools: List[Dict[str, Any]],
                                tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto",
                                model: str = "gpt-4o-mini",
                                paradigm: str = "bernard") -> Dict[str, Any]:
        """
        Generate completion with tool/function calling
        
        Args:
            prompt: The prompt for generation
            tools: List of available tools/functions
            tool_choice: How to choose tools ("auto", "none", or specific tool)
            model: Model to use
            paradigm: Paradigm for system prompt
            
        Returns:
            Response including any tool calls
        """
        if self.azure_client:
            try:
                azure_model_mapping = {
                    "gpt-4o": os.getenv("AZURE_GPT4_DEPLOYMENT_NAME", "gpt-4o"),
                    "gpt-4o-mini": os.getenv("AZURE_GPT4_MINI_DEPLOYMENT_NAME", "gpt-4o-mini")
                }
                
                azure_deployment = azure_model_mapping.get(model, model)
                
                response = await self.azure_client.chat.completions.create(
                    model=azure_deployment,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt(paradigm)},
                        {"role": "user", "content": prompt}
                    ],
                    tools=tools,
                    tool_choice=tool_choice,
                    max_tokens=2000
                )
                
                return {
                    "message": response.choices[0].message,
                    "tool_calls": response.choices[0].message.tool_calls if response.choices[0].message.tool_calls else []
                }
                
            except Exception as e:
                logger.warning(f"Azure OpenAI tool calling failed: {str(e)}")
        
        # Fallback to OpenAI
        if self.openai_client:
            try:
                response = await self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt(paradigm)},
                        {"role": "user", "content": prompt}
                    ],
                    tools=tools,
                    tool_choice=tool_choice,
                    max_tokens=2000
                )
                
                return {
                    "message": response.choices[0].message,
                    "tool_calls": response.choices[0].message.tool_calls if response.choices[0].message.tool_calls else []
                }
                
            except Exception as e:
                logger.error(f"OpenAI tool calling failed: {str(e)}")
                raise
        
        raise Exception("No LLM clients available for tool calling.")

    async def create_conversation(self,
                                messages: List[Dict[str, str]],
                                model: str = "gpt-4o-mini",
                                paradigm: str = "bernard",
                                max_tokens: int = 2000,
                                temperature: float = 0.7) -> str:
        """
        Create a conversation with multiple turns
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use
            paradigm: Paradigm for system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            The model's response
        """
        # Insert system message at the beginning
        full_messages = [{"role": "system", "content": self._get_system_prompt(paradigm)}]
        full_messages.extend(messages)
        
        # Select appropriate model
        model_mapping = {
            "dolores": "gpt-4o",
            "teddy": "gpt-4o-mini",
            "bernard": "gpt-4o",
            "maeve": "gpt-4o-mini"
        }
        
        selected_model = model_mapping.get(paradigm, model)
        
        # Try Azure OpenAI first
        if self.azure_client:
            try:
                azure_model_mapping = {
                    "gpt-4o": os.getenv("AZURE_GPT4_DEPLOYMENT_NAME", "gpt-4o"),
                    "gpt-4o-mini": os.getenv("AZURE_GPT4_MINI_DEPLOYMENT_NAME", "gpt-4o-mini")
                }
                
                azure_deployment = azure_model_mapping.get(selected_model, selected_model)
                
                response = await self.azure_client.chat.completions.create(
                    model=azure_deployment,
                    messages=full_messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                logger.info(f"✓ Created conversation using Azure OpenAI ({selected_model})")
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                logger.warning(f"Azure OpenAI conversation failed: {str(e)}")
        
        # Fallback to OpenAI
        if self.openai_client:
            try:
                response = await self.openai_client.chat.completions.create(
                    model=selected_model,
                    messages=full_messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                logger.info(f"✓ Created conversation using OpenAI ({selected_model})")
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                logger.error(f"OpenAI conversation failed: {str(e)}")
                raise
        
        raise Exception("No LLM clients available for conversation.")

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
