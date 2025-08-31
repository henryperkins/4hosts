"""
Legacy compatibility layer for answer generation.

Provides the symbols expected by older tests/modules:
- BernardAnswerGenerator, MaeveAnswerGenerator
- AnswerGenerationOrchestrator, answer_orchestrator
- initialize_answer_generation

Internally delegates to the enhanced V2 implementations to keep behavior modern
while preserving the import surface.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Re-export context/dataclasses for callers that import them from here
from .answer_generator import (
    BaseAnswerGenerator,
    SynthesisContext,
    GeneratedAnswer,
)

# Use the enhanced generators
from .answer_generator_enhanced import (
    EnhancedBernardAnswerGenerator as BernardAnswerGenerator,
    EnhancedMaeveAnswerGenerator as MaeveAnswerGenerator,
)

# Orchestrator – reuse the enhanced orchestrator implementation
from .enhanced_integration import (
    EnhancedAnswerGenerationOrchestrator as AnswerGenerationOrchestrator,
)


# Singleton orchestrator instance (legacy name)
answer_orchestrator = AnswerGenerationOrchestrator()


async def initialize_answer_generation() -> bool:
    """Optionally perform any warmups for generators/orchestrator.

    Currently a no-op that returns True for compatibility.
    """
    return True

