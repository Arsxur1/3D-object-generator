"""Layer 6 — interpretation & explanation (TZ §9). Grounded RU/UZ protocol."""

from __future__ import annotations

from .llm_base import (
    LLMProvider,
    LLMInput,
    build_llm_input,
    ModeGateResult,
    evaluate_mode_b_gate,
)
from .deterministic_provider import DeterministicProvider
from .grounding import validate_grounding
from .protocol import render_protocol
from .factory import make_provider

__all__ = [
    "LLMProvider",
    "LLMInput",
    "build_llm_input",
    "ModeGateResult",
    "evaluate_mode_b_gate",
    "DeterministicProvider",
    "validate_grounding",
    "render_protocol",
    "make_provider",
]
