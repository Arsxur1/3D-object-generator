"""Report contract — output of Layer 6 (TZ §9, §11).

Standard EEG-protocol structure, bilingual (RU + UZ), schema-constrained.
Every user-facing statement is grounded (``grounding_refs``) against a
Layer-3 feature, a Layer-4 event, or a Layer-5 causal-graph node/edge.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class OperatingMode(str, Enum):
    A_DECISION_SUPPORT = "A"  # default, safe: physician decides
    B_AUTONOMOUS = "B"  # gated autonomous impression


DISCLAIMER_RU = (
    "Система поддержки принятия решений. Не является медицинским заключением. "
    "Итоговая интерпретация и решение — за квалифицированным нейрофизиологом/"
    "реаниматологом."
)
DISCLAIMER_UZ = (
    "Qaror qabul qilishni qo‘llab-quvvatlash tizimi. Bu tibbiy xulosa emas. "
    "Yakuniy talqin va qaror malakali neyrofiziolog/reanimatolog zimmasida."
)


class Bilingual(BaseModel):
    """A piece of text in both report languages (TZ: заключения на RU и UZ)."""

    model_config = ConfigDict(extra="forbid")

    ru: str
    uz: str


class Confidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float = Field(ge=0, le=1)
    label_ru: str = Field(description="Word form, e.g. 'высокая'/'средняя'/'низкая'.")
    label_uz: str


class ReportSection(BaseModel):
    """One grounded section/statement of the protocol."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(description="background | epileptiform | ictal_acns | causal | impression | next_step")
    text: Bilingual
    grounding_refs: list[str] = Field(
        default_factory=list,
        description="Identifiers this statement is grounded on (feature/event/node/edge).",
    )
    confidence: Optional[Confidence] = None


class CriticalFinding(BaseModel):
    """Hard-safety finding surfaced independently of LLM/mode (TZ §10)."""

    model_config = ConfigDict(extra="forbid")

    code: str
    text: Bilingual
    grounding_refs: list[str] = Field(default_factory=list)


class LLMReport(BaseModel):
    """Full Layer-6 output (schema-constrained; TZ §9)."""

    model_config = ConfigDict(extra="forbid")

    mode: OperatingMode
    mode_gate_passed: bool = Field(
        default=False,
        description="For mode B: whether autonomous gates (TZ §2) were satisfied.",
    )
    escalation_reason: Optional[Bilingual] = Field(
        default=None,
        description="Set when review is required (mode B gate failed or low quality).",
    )

    sections: list[ReportSection] = Field(default_factory=list)
    impression: Bilingual
    overall_confidence: Confidence
    critical_findings: list[CriticalFinding] = Field(default_factory=list)
    reasoning_trace: list[str] = Field(
        default_factory=list, description="Visible chain of reasoning (TZ §9)."
    )

    provider: str = Field(default="deterministic", description="Which Layer-6 provider ran.")
    disclaimer_ru: str = DISCLAIMER_RU
    disclaimer_uz: str = DISCLAIMER_UZ

    def section(self, key: str) -> Optional[ReportSection]:
        for s in self.sections:
            if s.key == key:
                return s
        return None
