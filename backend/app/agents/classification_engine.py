import logging
from fastapi import HTTPException, status
from openai import RateLimitError

from app.agents.llm_client import LLMClient
from app.schemas.classification import ClassificationResult

logger = logging.getLogger(__name__)


class ClassificationEngine:
    def __init__(self, llm_client: LLMClient, model: str):
        self.llm_client = llm_client
        self.model = model

    async def classify(self, query: str) -> ClassificationResult:
        messages = [{"role": "user", "content": query}]
        try:
            llm_response = await self.llm_client.chat(messages, model=self.model)
        except RateLimitError as exc:
            logger.error(f"OpenAI quota exceeded: {exc}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Upstream LLM quota exceeded. Please try again later.",
            )

        # Process llm_response and return ClassificationResult
        ...
