"""Light-weight evaluation helpers for context packaging.

These routines provide deterministic heuristics that mirror the metrics used
in the context-engineering docs (context precision, context utilisation and
groundedness proxies).  They do not call external LLMs which keeps unit tests
fast and makes the evaluators suitable for smoke tests and CI gating.

Optional: Can use Azure Content Safety API for advanced groundedness detection
when configured via CONTENT_SAFETY_ENDPOINT and CONTENT_SAFETY_API_KEY.
"""

from __future__ import annotations

import os
import re
import json
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Optional, Any

from services.context_packager import ContextPackage


WORD_RE = re.compile(r"[a-z0-9]{3,}")


def _tokenize(text: str) -> List[str]:
    return WORD_RE.findall((text or "").lower())


def _unique_tokens(text: str) -> List[str]:
    seen = set()
    tokens: List[str] = []
    for token in _tokenize(text):
        if token not in seen:
            tokens.append(token)
            seen.add(token)
    return tokens


@dataclass
class ContextEvaluationReport:
    precision: float
    utilization: float
    groundedness: float
    notes: List[str]
    content_safety_groundedness: Optional[float] = None
    content_safety_details: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        result = {
            "precision": round(self.precision, 3),
            "utilization": round(self.utilization, 3),
            "groundedness": round(self.groundedness, 3),
        }
        if self.content_safety_groundedness is not None:
            result["content_safety_groundedness"] = round(self.content_safety_groundedness, 3)
        if self.content_safety_details:
            result["content_safety_details"] = self.content_safety_details
        return result


def evaluate_context_package(
    package: ContextPackage,
    *,
    answer: str,
    retrieved_documents: Sequence[str],
) -> ContextEvaluationReport:
    """Compute heuristic metrics for a packaged context."""

    knowledge_segment = package.segment("knowledge")
    instructions_segment = package.segment("instructions")

    knowledge_tokens = []
    if knowledge_segment:
        for item in knowledge_segment.content:
            knowledge_tokens.extend(_unique_tokens(item.get("content", "")))

    answer_tokens = _unique_tokens(answer)

    if knowledge_tokens:
        precision = _token_overlap(answer_tokens, knowledge_tokens)
    else:
        precision = 0.0

    utilization = _utilization_score(knowledge_tokens, answer_tokens)
    groundedness = _groundedness_score(answer_tokens, retrieved_documents)

    notes: List[str] = []
    if instructions_segment and not instructions_segment.content:
        notes.append("Instruction segment empty")
    if not knowledge_tokens:
        notes.append("Knowledge segment empty")

    return ContextEvaluationReport(precision=precision, utilization=utilization, groundedness=groundedness, notes=notes)


def _token_overlap(a: Iterable[str], b: Iterable[str]) -> float:
    set_a, set_b = set(a), set(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / float(len(set_b))


def _utilization_score(knowledge_tokens: List[str], answer_tokens: List[str]) -> float:
    if not knowledge_tokens or not answer_tokens:
        return 0.0
    hits = sum(1 for token in knowledge_tokens if token in answer_tokens)
    return hits / float(len(knowledge_tokens))


def _groundedness_score(answer_tokens: List[str], retrieved_documents: Sequence[str]) -> float:
    if not answer_tokens or not retrieved_documents:
        return 0.0

    doc_tokens: List[str] = []
    for doc in retrieved_documents:
        doc_tokens.extend(_unique_tokens(doc))
    doc_tokens_set = set(doc_tokens)
    if not doc_tokens_set:
        return 0.0

    supported = sum(1 for token in answer_tokens if token in doc_tokens_set)
    return supported / float(len(answer_tokens))


def _get_groundedness_mode() -> str:
    """Determine the groundedness feature mode from environment"""
    if os.getenv("AZURE_CS_GROUNDEDNESS_CORRECTION") in ("1", "true", "True"):
        return "correction"
    if os.getenv("AZURE_CS_GROUNDEDNESS_REASONING") in ("1", "true", "True"):
        return "reasoning"
    return "basic"


async def _retry_with_backoff(
    func,
    max_retries: int = 2,
    initial_delay: float = 1.0
) -> Any:
    """Execute async function with exponential backoff retry"""
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = initial_delay * (2 ** attempt)
            await asyncio.sleep(delay)
    return None


async def check_content_safety_groundedness(
    text: str,
    grounding_sources: List[str],
    task_type: str = "Summarization",
    domain: str = "Generic"
) -> Optional[Dict[str, Any]]:
    """
    Check groundedness using Azure Content Safety API with optional reasoning/correction.

    Returns None if not configured or on error, never blocks answer pipeline.
    """
    # Check if groundedness is enabled
    if not os.getenv("AZURE_CS_ENABLE_GROUNDEDNESS"):
        return None

    endpoint = os.getenv("AZURE_CS_ENDPOINT")
    api_key = os.getenv("AZURE_CS_KEY")

    if not endpoint:
        return None

    import aiohttp
    import structlog
    logger = structlog.get_logger(__name__)

    mode = _get_groundedness_mode()

    # Build payload
    payload = {
        "domain": domain,
        "task": task_type,
        "text": text[:7500],  # API limit
        "groundingSources": [source[:10000] for source in grounding_sources[:20]]  # Cap at 20 sources
    }

    # Add mode-specific configuration
    if mode == "reasoning":
        payload["reasoning"] = True
    elif mode == "correction":
        payload["correction"] = True

    # Add LLM resource for advanced modes
    if mode in ("reasoning", "correction"):
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        aoai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        if deployment and aoai_endpoint:
            payload["llmResource"] = {
                "resourceType": "AzureOpenAI",
                "azureOpenAIEndpoint": aoai_endpoint.rstrip('/'),
                "azureOpenAIDeploymentName": deployment
            }
        else:
            # Downgrade to basic mode silently
            mode = "basic"
            payload.pop("reasoning", None)
            payload.pop("correction", None)
            logger.info(
                "Downgrading to basic groundedness mode",
                reason="Missing Azure OpenAI configuration"
            )

    # Prepare request
    url = f"{endpoint.rstrip('/')}/contentsafety/text:detectGroundedness?api-version=2024-09-15-preview"
    timeout_seconds = float(os.getenv("AZURE_HTTP_TIMEOUT_SECONDS", "12"))

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Ocp-Apim-Subscription-Key"] = api_key

    async def make_request():
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout_seconds)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()

                    # Normalize output structure
                    result = {
                        "ungrounded_detected": data.get("ungroundedDetected", False),
                        "ungrounded_percentage": float(data.get("ungroundedPercentage", 0.0)),
                        "ungrounded_details": data.get("ungroundedDetails", []),
                        "mode": mode
                    }

                    # Add reasoning if present
                    if mode == "reasoning" and result["ungrounded_details"]:
                        for detail in result["ungrounded_details"]:
                            if "reason" in detail:
                                result.setdefault("reasoning", []).append(detail["reason"])

                    # Add correction text if present
                    if mode == "correction":
                        result["correction_text"] = (
                            data.get("correctionText") or
                            data.get("correction Text")  # Handle both formats
                        )

                    logger.info(
                        "content_safety_groundedness_result",
                        mode=mode,
                        ungrounded_percentage=result["ungrounded_percentage"],
                        details_count=len(result["ungrounded_details"])
                    )

                    return result
                else:
                    body = await resp.text()
                    logger.warning(
                        "content_safety_groundedness_failed",
                        status=resp.status,
                        body=body[:500],
                        mode=mode
                    )

                    # Handle specific error codes
                    if resp.status == 401:
                        logger.error("Invalid Content Safety API key")
                    elif resp.status == 403:
                        logger.error("RBAC missing for Content Safety MI to access Azure OpenAI")
                    elif resp.status == 429:
                        logger.warning("Rate limit exceeded for Content Safety")

                    return None

    try:
        # Use retry logic for transient failures
        if mode in ("reasoning", "correction"):
            # Advanced modes may take longer, use retry
            return await _retry_with_backoff(make_request, max_retries=2)
        else:
            # Basic mode is fast, single attempt
            return await make_request()

    except asyncio.TimeoutError:
        logger.warning(
            "content_safety_groundedness_timeout",
            mode=mode,
            timeout_seconds=timeout_seconds
        )
        return None
    except Exception as e:
        logger.warning(
            "content_safety_groundedness_exception",
            error=str(e),
            mode=mode
        )
        return None


async def evaluate_context_package_async(
    package: ContextPackage,
    *,
    answer: str,
    retrieved_documents: Sequence[str],
    check_content_safety: bool = True
) -> ContextEvaluationReport:
    """
    Async version of evaluate_context_package with optional Content Safety check.
    """
    # Run the basic evaluation first (it's synchronous)
    report = evaluate_context_package(package, answer=answer, retrieved_documents=retrieved_documents)

    # Optionally check Content Safety groundedness
    if check_content_safety and answer and retrieved_documents:
        cs_result = await check_content_safety_groundedness(
            text=answer,
            grounding_sources=list(retrieved_documents)
        )

        if cs_result:
            # Add Content Safety results to the report
            report.content_safety_groundedness = 1.0 - cs_result["ungrounded_percentage"]
            report.content_safety_details = cs_result

            # Add note if significant discrepancy
            heuristic_grounded = report.groundedness
            cs_grounded = report.content_safety_groundedness
            if abs(heuristic_grounded - cs_grounded) > 0.3:
                report.notes.append(
                    f"Groundedness discrepancy: heuristic={heuristic_grounded:.2f}, "
                    f"content_safety={cs_grounded:.2f}"
                )

    return report
