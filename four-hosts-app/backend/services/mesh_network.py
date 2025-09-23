"""
Mesh Network Negotiation

Implements multi-paradigm negotiation for complex queries where multiple
paradigms have significant probability. This module is feature-gated
via ENABLE_MESH_NETWORK in core.config and is designed to be safe to
import without initializing heavy dependencies.
"""

from __future__ import annotations

import structlog
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    # Prefer canonical enum to keep naming consistent throughout the app
    from models.base import HostParadigm
except Exception:
    # Fallback typing shim if import graph changes
    from enum import Enum

    class HostParadigm(str, Enum):  # type: ignore[no-redef]
        DOLORES = "dolores"
        TEDDY = "teddy"
        BERNARD = "bernard"
        MAEVE = "maeve"


from logging_config import configure_logging

configure_logging()
logger = structlog.get_logger(__name__)


@dataclass
class ParadigmStance:
    """A paradigm's stance on the evidence"""
    paradigm: HostParadigm
    perspective: str
    key_points: List[str]
    evidence_refs: List[str]
    confidence: float


@dataclass
class MeshSynthesis:
    """Result of multi-paradigm negotiation"""
    primary_synthesis: str
    paradigm_stances: List[ParadigmStance]
    synergies: List[str]
    tensions: List[str]
    integrated_recommendation: str
    evidence_refs: List[str]
    negotiation_metadata: Dict[str, Any]


class MeshNetworkNegotiator:
    """Orchestrates multi-paradigm negotiation"""
    def __init__(
        self,
        *,
        min_probability: float = 0.25,
        max_paradigms: int = 3,
    ) -> None:
        from core.config import MESH_MIN_PROBABILITY, MESH_MAX_PARADIGMS  # lazy import
        self.min_paradigm_probability = float(MESH_MIN_PROBABILITY or min_probability)
        self.max_paradigms_in_negotiation = int(MESH_MAX_PARADIGMS or max_paradigms)

    async def should_negotiate(self, classification: Any) -> bool:
        """
        Determine if mesh negotiation is warranted based on the classification distribution.
        Expects classification.distribution to be a mapping of HostParadigm -> float.
        """
        try:
            dist = getattr(classification, "distribution", {}) or {}
            if not isinstance(dist, dict) or not dist:
                return False
            # Count paradigms above threshold
            significant = sum(1 for v in dist.values() if (isinstance(v, (int, float)) and v >= self.min_paradigm_probability))
            if significant >= 2:
                return True
            # Also check closeness of top two
            try:
                vals = sorted([float(v) for v in dist.values()], reverse=True)
                if len(vals) >= 2 and (vals[0] - vals[1]) < 0.15:
                    return True
            except Exception:
                pass
            return False
        except Exception:
            return False

    async def negotiate(
        self,
        classification: Any,
        evidence_pool: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> MeshSynthesis:
        """Execute multi-paradigm negotiation. Best-effort and resilient to LLM errors."""
        paradigms = self._select_paradigms(classification)
        stances = await self._generate_stances(paradigms, getattr(classification, "query", ""), evidence_pool)
        synergies, tensions = await self._analyze_relationships(stances)
        integrated = await self._integrate_perspectives(stances, synergies, tensions, getattr(classification, "query", ""))

        # Collect evidence refs across stances
        refs: List[str] = []
        for s in stances:
            refs.extend(s.evidence_refs)
        refs = list({r for r in refs if r})

        return MeshSynthesis(
            primary_synthesis=integrated.get("synthesis", ""),
            paradigm_stances=stances,
            synergies=synergies,
            tensions=tensions,
            integrated_recommendation=integrated.get("recommendation", ""),
            evidence_refs=refs,
            negotiation_metadata={
                "paradigms_involved": [getattr(p, "value", str(p)) for p in paradigms],
                "negotiation_time": datetime.utcnow().isoformat(),
                "evidence_count": len(evidence_pool),
            },
        )

    def _select_paradigms(self, classification: Any) -> List[HostParadigm]:
        dist = getattr(classification, "distribution", {}) or {}
        # Items are (paradigm, prob); ensure paradigms are HostParadigm where possible
        items: List[Tuple[Any, float]] = []
        for k, v in dist.items():
            try:
                prob = float(v or 0.0)
            except Exception:
                prob = 0.0
            items.append((k, prob))
        items.sort(key=lambda x: x[1], reverse=True)

        selected: List[HostParadigm] = []
        for k, prob in items:
            if prob < self.min_paradigm_probability:
                continue
            try:
                # If key is HostParadigm, keep; if it's Paradigm or string, coerce to enum by name/value
                if isinstance(k, HostParadigm):
                    hp = k
                else:
                    name = getattr(k, "name", None) or getattr(k, "value", None) or str(k)
                    name_upper = str(name).upper()
                    hp = HostParadigm[name_upper] if name_upper in HostParadigm.__members__ else HostParadigm(str(name).lower())
                selected.append(hp)
                if len(selected) >= self.max_paradigms_in_negotiation:
                    break
            except Exception:
                continue
        # Ensure at least primary exists
        if not selected:
            try:
                primary = getattr(classification, "primary_paradigm", None)
                if primary:
                    selected = [primary]  # type: ignore[assignment]
            except Exception:
                pass
        return selected

    async def _generate_stances(
        self,
        paradigms: List[HostParadigm],
        query: str,
        evidence_pool: List[Dict[str, Any]],
    ) -> List[ParadigmStance]:
        stances: List[ParadigmStance] = []
        # Build evidence text
        evidence_text = "\n".join([
            f"{i}. {e.get('title', 'Evidence')}: {str(e.get('snippet') or e.get('summary') or '')[:200]}"
            for i, e in enumerate(evidence_pool[:20])
        ])

        # LLM client (lazy import to avoid heavy import path on app startup)
        from services.llm_client import llm_client

        for paradigm in paradigms:
            prompt = (
                f"As the {getattr(paradigm, 'value', str(paradigm))} paradigm, analyze this query and evidence.\n\n"
                f"Query: {query}\n\nEvidence:\n{evidence_text}\n\n"
                f"Provide your paradigm's perspective, 3-5 key points, indices of evidence items you rely on, and a confidence (0-1)."
            )
            schema = {
                "name": "paradigm_stance",
                "schema": {
                    "type": "object",
                    "properties": {
                        "perspective": {"type": "string"},
                        "key_points": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 5},
                        "evidence_indices": {"type": "array", "items": {"type": "integer"}, "maxItems": 10},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["perspective", "key_points", "evidence_indices", "confidence"],
                    "additionalProperties": False,
                },
                "strict": True,
            }

            try:
                result = await llm_client.generate_structured_output(
                    prompt=prompt,
                    schema=schema,
                    paradigm=getattr(paradigm, "value", str(paradigm)),
                )
                evidence_refs: List[str] = []
                for idx in result.get("evidence_indices", []) or []:
                    if isinstance(idx, int) and 0 <= idx < len(evidence_pool):
                        evidence_refs.append(evidence_pool[idx].get("id") or evidence_pool[idx].get("url") or f"evidence:{idx}")
                stances.append(ParadigmStance(
                    paradigm=paradigm,
                    perspective=str(result.get("perspective", "")),
                    key_points=[str(x) for x in (result.get("key_points") or [])],
                    evidence_refs=evidence_refs,
                    confidence=float(result.get("confidence") or 0.0),
                ))
            except Exception as e:
                logger.debug("Stance generation failed for %s: %s", paradigm, e)
                stances.append(ParadigmStance(
                    paradigm=paradigm,
                    perspective=f"{getattr(paradigm, 'value', str(paradigm))} perspective unavailable",
                    key_points=[],
                    evidence_refs=[],
                    confidence=0.0,
                ))
        return stances

    async def _analyze_relationships(self, stances: List[ParadigmStance]) -> Tuple[List[str], List[str]]:
        if len(stances) < 2:
            return [], []
        try:
            summaries = "\n".join([f"{getattr(s.paradigm, 'value', str(s.paradigm))}: {s.perspective}" for s in stances])
            prompt = (
                "Analyze these paradigm perspectives and identify up to 5 synergies and 5 tensions.\n\n"
                f"{summaries}\n\nReturn concise bullet items."
            )
            schema = {
                "name": "relationship_analysis",
                "schema": {
                    "type": "object",
                    "properties": {
                        "synergies": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
                        "tensions": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
                    },
                    "required": ["synergies", "tensions"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
            from services.llm_client import llm_client
            result = await llm_client.generate_structured_output(
                prompt=prompt,
                schema=schema,
                paradigm="bernard",
            )
            return [str(x) for x in (result.get("synergies") or [])], [str(x) for x in (result.get("tensions") or [])]
        except Exception as e:
            logger.debug("Relationship analysis failed: %s", e)
            return [], []

    async def _integrate_perspectives(
        self,
        stances: List[ParadigmStance],
        synergies: List[str],
        tensions: List[str],
        query: str,
    ) -> Dict[str, str]:
        try:
            formatted_stances = "\n\n".join([
                f"{getattr(s.paradigm, 'value', str(s.paradigm)).upper()}:\n- " + "\n- ".join(s.key_points) for s in stances
            ])
            prompt = (
                "Create an integrated response to this query by synthesizing multiple paradigm perspectives.\n\n"
                f"Query: {query}\n\nPerspectives:\n{formatted_stances}\n\n"
                f"Synergies:\n- " + "\n- ".join(synergies or ["None"]) + "\n\n"
                f"Tensions:\n- " + "\n- ".join(tensions or ["None"]) + "\n\n"
                "Provide a short synthesis paragraph and a clear recommendation paragraph."
            )
            schema = {
                "name": "integrated_synthesis",
                "schema": {
                    "type": "object",
                    "properties": {
                        "synthesis": {"type": "string"},
                        "recommendation": {"type": "string"},
                    },
                    "required": ["synthesis", "recommendation"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
            from services.llm_client import llm_client
            result = await llm_client.generate_structured_output(
                prompt=prompt,
                schema=schema,
                paradigm="bernard",
            )
            return {k: str(v) for k, v in result.items() if isinstance(v, str)}
        except Exception as e:
            logger.debug("Integration failed: %s", e)
            # Fallback to concatenated textual synthesis
            synth = " ".join([s.perspective for s in stances if s.perspective]) or "Multiple perspectives available."
            rec = "Balance complementary strengths while addressing identified tensions."
            return {"synthesis": synth, "recommendation": rec}


# Global instance
mesh_negotiator = MeshNetworkNegotiator()
