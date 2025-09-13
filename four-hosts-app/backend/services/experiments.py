"""
Lightweight experimentation utilities (A/B/C testing)
----------------------------------------------------
Provides a minimal, deterministic variant assignment mechanism that can be
used to A/B prompts or other behaviors. This intentionally avoids external
dependencies and persists no state; assignment is sticky per `unit_id` by
using a stable hash and variant weight buckets.

Enable globally via environment variable `ENABLE_PROMPT_AB=1`.
Optionally override variant weights with `PROMPT_AB_WEIGHTS` as JSON, e.g.:
  PROMPT_AB_WEIGHTS='{"v1":0.5,"v2":0.5}'

Usage:
  if experiments.enabled():
      variant = experiments.assign("deep_research_paradigm_prompt", unit_id)
      # Use `variant` to select prompt text; also record in metrics as needed.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


def enabled() -> bool:
    """Return True if experimentation is globally enabled via env var."""
    try:
        return os.getenv("ENABLE_PROMPT_AB", "0").strip() in {"1", "true", "TRUE", "yes", "on"}
    except Exception:
        return False


def _weights_for(experiment: str) -> List[Tuple[str, float]]:
    """Return a list of (variant, weight) pairs for the experiment.

    We currently support a single experiment name but keep the function
    extensible. Default to two-way 50/50 split.
    """
    default = [("v1", 0.5), ("v2", 0.5)]
    raw = os.getenv("PROMPT_AB_WEIGHTS")
    if not raw:
        return default
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and parsed:
            items = list(parsed.items())
            # Normalize weights to sum to 1.0
            total = float(sum(float(w) for _, w in items)) or 1.0
            normed = [(k, max(0.0, float(v)) / total) for k, v in items]
            return normed
    except Exception:
        pass
    return default


def _stable_bucket(value: str) -> float:
    """Map an arbitrary string to [0,1) using SHA256."""
    h = hashlib.sha256(value.encode("utf-8")).digest()
    # Use first 8 bytes as big-endian integer for stability
    n = int.from_bytes(h[:8], byteorder="big", signed=False)
    return (n % 10_000_000) / 10_000_000.0


def assign(experiment: str, unit_id: str) -> str:
    """Assign a deterministic variant for `unit_id` given weight buckets."""
    buckets = _weights_for(experiment)
    x = _stable_bucket(f"{experiment}::{unit_id}")
    cumulative = 0.0
    for variant, weight in buckets:
        cumulative += weight
        if x < cumulative:
            return variant
    # Fallback to last variant in case of rounding
    return buckets[-1][0]


def variant_or_default(experiment: str, unit_id: str, default: str = "v1") -> str:
    """Return assigned variant when enabled, otherwise the provided default."""
    if enabled():
        try:
            return assign(experiment, unit_id)
        except Exception:
            return default
    return default

