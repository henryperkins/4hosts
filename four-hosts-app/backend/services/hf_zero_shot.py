from functools import lru_cache
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

# Paradigm labels matching the Four Hosts system
LABELS = ["revolutionary", "devotion", "analytical", "strategic"]

@lru_cache(maxsize=1)
def get_classifier(device: int | str = -1):
    """
    Lazily load the Hugging Face zero-shot classifier.
    Set device=-1 for CPU, 0 for the first CUDA GPU.
    The LRU cache ensures this happens only once.
    """
    logger.info(f"Loading DeBERTa zero-shot classifier on device: {device}")
    return pipeline(
        task="zero-shot-classification",
        model="microsoft/deberta-large-mnli",
        device=device
    )

def predict_paradigm(text: str) -> tuple[str, float]:
    """
    Predict the paradigm for a given text using zero-shot classification.
    
    Args:
        text: The query text to classify
        
    Returns:
        Tuple of (paradigm_label, confidence_score)
    """
    clf = get_classifier()
    result = clf(text, candidate_labels=LABELS, multi_label=False)
    
    # Return the top prediction and its score
    paradigm = result["labels"][0]
    score = float(result["scores"][0])
    
    logger.debug(f"Zero-shot prediction: {paradigm} (confidence: {score:.3f})")
    return paradigm, score

async def async_predict_paradigm(text: str) -> tuple[str, float]:
    """
    Async wrapper for predict_paradigm to avoid blocking the event loop.
    Uses asyncio's run_in_executor for CPU-bound operations.
    """
    import asyncio
    from functools import partial
    
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(predict_paradigm, text))