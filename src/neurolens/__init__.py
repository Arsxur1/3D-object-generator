"""NeuroLens — EEG detection and interpretation system.

Layered architecture (see TZ §3):
    [1] ingestion  -> [2] preprocessing/quality -> [3] quantitative features
    -> [4] pattern/event detection -> [5] causal & physiological reasoning
    -> [6] interpretation & explanation (LLM, RU/UZ protocol)

Each layer is a replaceable module behind an explicit contract. All data
contracts live in ``neurolens.contracts`` as pydantic models (the single
source of truth; JSON Schemas are exported from them).

IMPORTANT (safety): NeuroLens is a decision-support tool. It does not make
clinical decisions. The default operating mode (A) always requires review by
a qualified neurophysiologist / intensivist. See ``neurolens.contracts.report``
for the mandatory disclaimer.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
