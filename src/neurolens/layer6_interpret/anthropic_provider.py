"""Anthropic LLM provider (TZ §9) — schema-constrained, grounded narrative.

Sends ONLY the structured, grounded :class:`LLMInput` to Claude and forces a
tool call whose schema mirrors the report structure. The model is instructed to
cite only ``allowed_grounding_keys``; whatever it returns is still run through
``validate_grounding`` downstream, so an invented reference is stripped and an
ungrounded factual section is dropped. On any API/network/key error it falls
back to the deterministic provider so the pipeline always yields a report.
"""

from __future__ import annotations

import json
import os

from ..contracts.report import (
    Bilingual,
    Confidence,
    LLMReport,
    OperatingMode,
    ReportSection,
)
from .deterministic_provider import DeterministicProvider, _confidence
from .llm_base import LLMInput, LLMProvider

_DEFAULT_MODEL = os.environ.get("NEUROLENS_LLM_MODEL", "claude-sonnet-4-5")

_SYSTEM = (
    "Ты — ассистент нейрофизиолога, формирующий протокол ЭЭГ на РУССКОМ и УЗБЕКСКОМ. "
    "Категорически запрещено выдумывать события, находки или связи, которых нет во "
    "входных данных. Каждое фактическое утверждение обязано ссылаться на ключ из "
    "allowed_grounding_keys через grounding_refs. Используй только предоставленный "
    "каузальный граф, признаки и события. Это инструмент поддержки решений врача, "
    "не диагностическое заключение."
)

_TOOL = {
    "name": "emit_eeg_protocol",
    "description": "Return the structured, grounded EEG protocol (RU+UZ).",
    "input_schema": {
        "type": "object",
        "properties": {
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "enum": ["background", "epileptiform", "ictal_acns", "causal", "next_step"],
                        },
                        "text_ru": {"type": "string"},
                        "text_uz": {"type": "string"},
                        "grounding_refs": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["key", "text_ru", "text_uz", "grounding_refs"],
                },
            },
            "impression_ru": {"type": "string"},
            "impression_uz": {"type": "string"},
            "overall_confidence": {"type": "number"},
            "reasoning_trace": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["sections", "impression_ru", "impression_uz", "overall_confidence", "reasoning_trace"],
    },
}


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str | None = None, max_tokens: int = 2000):
        self.model = model or _DEFAULT_MODEL
        self.max_tokens = max_tokens
        self._fallback = DeterministicProvider()

    def generate(self, data: LLMInput) -> LLMReport:
        try:
            return self._generate_api(data)
        except Exception as exc:  # never let the LLM step break the pipeline
            report = self._fallback.generate(data)
            report.provider = "anthropic_fallback_deterministic"
            report.reasoning_trace.append(f"Anthropic провайдер недоступен ({exc}); использован детерминированный фолбэк.")
            return report

    def _generate_api(self, data: LLMInput) -> LLMReport:
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        payload = {
            "mode": data.mode.value,
            "patient": data.patient,
            "context": data.context,
            "background": data.background,
            "events": data.events,
            "causal_nodes": data.causal_nodes,
            "causal_edges": data.causal_edges,
            "critical_flags": data.critical_flags,
            "implausible_flags": data.implausible_flags,
            "allowed_grounding_keys": data.allowed_grounding_keys,
        }
        msg = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM,
            tools=[_TOOL],
            tool_choice={"type": "tool", "name": "emit_eeg_protocol"},
            messages=[{
                "role": "user",
                "content": (
                    "Составь протокол ЭЭГ строго по этим данным (JSON ниже). "
                    "Ссылайся только на allowed_grounding_keys.\n\n"
                    + json.dumps(payload, ensure_ascii=False)
                ),
            }],
        )
        tool_input = _extract_tool_input(msg)
        return self._to_report(data, tool_input)

    def _to_report(self, data: LLMInput, out: dict) -> LLMReport:
        sections = [
            ReportSection(
                key=s["key"],
                text=Bilingual(ru=s["text_ru"], uz=s["text_uz"]),
                grounding_refs=list(s.get("grounding_refs", [])),
            )
            for s in out.get("sections", [])
        ]
        conf = float(out.get("overall_confidence", 0.5))

        mode_gate_passed = False
        escalation = None
        if data.mode == OperatingMode.B_AUTONOMOUS:
            if data.gate.passed:
                mode_gate_passed = True
            else:
                escalation = Bilingual(
                    ru="Требуется просмотр врача: " + "; ".join(data.gate.reasons_ru),
                    uz="Shifokor ko‘rigi zarur: " + "; ".join(data.gate.reasons_uz),
                )

        return LLMReport(
            mode=data.mode,
            mode_gate_passed=mode_gate_passed,
            escalation_reason=escalation,
            sections=sections,
            impression=Bilingual(ru=out["impression_ru"], uz=out["impression_uz"]),
            overall_confidence=_confidence(conf),
            reasoning_trace=list(out.get("reasoning_trace", [])),
            provider=self.name,
        )


def _extract_tool_input(msg) -> dict:
    for block in getattr(msg, "content", []):
        if getattr(block, "type", None) == "tool_use":
            return dict(block.input)
    raise ValueError("model did not return a tool_use block")
