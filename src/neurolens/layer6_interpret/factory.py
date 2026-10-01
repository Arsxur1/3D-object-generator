"""Provider selection (TZ §9, §12).

Default: use the real Anthropic provider when an API key is present and the SDK
imports; otherwise fall back to the deterministic provider. Both are validated
against the same grounding rules, so behaviour is safe either way. A local LLM
(full offline) is a v2 option behind this same factory.
"""

from __future__ import annotations

import os

from .deterministic_provider import DeterministicProvider
from .llm_base import LLMProvider


def make_provider(prefer: str = "auto") -> LLMProvider:
    """Return a Layer-6 provider.

    prefer: 'auto' (Anthropic if key+SDK available, else deterministic),
            'anthropic' (force Anthropic, still self-fallbacks on error),
            'deterministic' (offline, reproducible).
    """
    if prefer == "deterministic":
        return DeterministicProvider()

    key_present = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if prefer == "anthropic" or (prefer == "auto" and key_present):
        try:
            import anthropic  # noqa: F401

            from .anthropic_provider import AnthropicProvider

            return AnthropicProvider()
        except Exception:
            return DeterministicProvider()
    return DeterministicProvider()
