"""LLM provider interface + shared input assembly and mode-B gating (TZ §9, §2)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..contracts.causal import CausalGraph
from ..contracts.config import Thresholds
from ..contracts.events import DetectionResult
from ..contracts.report import LLMReport, OperatingMode
from ..contracts.signal import ClinicalContext, PatientInfo
from ..layer3_features.feature_set import FeatureSet


@dataclass
class ModeGateResult:
    """Result of the autonomous-mode (B) gate check (TZ §2B)."""

    passed: bool
    reasons_ru: list[str] = field(default_factory=list)
    reasons_uz: list[str] = field(default_factory=list)


def evaluate_mode_b_gate(
    mode: OperatingMode,
    features: FeatureSet,
    detection: DetectionResult,
    graph: CausalGraph,
    signal_quality: float,
    artifact_fraction: float,
    thresholds: Thresholds,
) -> ModeGateResult:
    """Decide whether autonomous mode B may issue an affirmative impression.

    Gates (all must hold): confidence, signal quality, artifact fraction, and
    physiological plausibility. Critical findings are never suppressed — they
    are reported regardless; the gate only governs the *affirmative* impression.
    """
    if mode != OperatingMode.B_AUTONOMOUS:
        return ModeGateResult(passed=False)

    g = thresholds.mode_b
    reasons_ru: list[str] = []
    reasons_uz: list[str] = []

    top_conf = max((e.confidence for e in detection.events), default=1.0)
    if top_conf < g.min_confidence:
        reasons_ru.append(f"уверенность {top_conf:.2f} < порога {g.min_confidence}")
        reasons_uz.append(f"ishonch {top_conf:.2f} < chegara {g.min_confidence}")
    if signal_quality < g.min_signal_quality:
        reasons_ru.append(f"качество сигнала {signal_quality:.2f} < {g.min_signal_quality}")
        reasons_uz.append(f"signal sifati {signal_quality:.2f} < {g.min_signal_quality}")
    if artifact_fraction > g.max_artifact_fraction:
        reasons_ru.append(f"доля артефактов {artifact_fraction:.2f} > {g.max_artifact_fraction}")
        reasons_uz.append(f"artefakt ulushi {artifact_fraction:.2f} > {g.max_artifact_fraction}")
    if g.require_plausible and graph.implausible_flags:
        reasons_ru.append("есть неправдоподобные констелляции — нужен просмотр")
        reasons_uz.append("nomuvofiq konstellyatsiyalar bor — ko‘rib chiqish kerak")

    return ModeGateResult(passed=not reasons_ru, reasons_ru=reasons_ru, reasons_uz=reasons_uz)


@dataclass
class LLMInput:
    """Structured, grounded input handed to a provider (never free-form)."""

    mode: OperatingMode
    patient: dict[str, Any]
    context: dict[str, Any]
    background: dict[str, Any]
    events: list[dict[str, Any]]
    causal_nodes: list[dict[str, Any]]
    causal_edges: list[dict[str, Any]]
    critical_flags: list[str]
    implausible_flags: list[str]
    seizure_burden: dict[str, Any] | None
    allowed_grounding_keys: list[str]
    gate: ModeGateResult


def build_llm_input(
    mode: OperatingMode,
    patient: PatientInfo,
    context: ClinicalContext,
    features: FeatureSet,
    detection: DetectionResult,
    graph: CausalGraph,
    gate: ModeGateResult,
) -> LLMInput:
    """Assemble the structured input; this is the ONLY thing a provider may use."""
    background = {
        "pdr_hz": round(features.pdr_hz, 2),
        "global_rel_delta": round(features.global_relpow["delta"], 3),
        "global_rel_theta": round(features.global_relpow["theta"], 3),
        "global_rel_alpha": round(features.global_relpow["alpha"], 3),
        "global_asymmetry": round(features.global_asymmetry, 3),
        "continuity": round(1.0 - features.burst.suppression_ratio, 3),
        "representative_channel": features.representative_channel,
    }
    events = [
        {
            "code": e.code,
            "label_ru": e.label_ru,
            "label_uz": e.label_uz,
            "group": e.group,
            "localization": e.localization.model_dump(),
            "t_start": e.t_start,
            "t_end": e.t_end,
            "confidence": e.confidence,
            "evidence": [ev.model_dump() for ev in e.evidence],
            "is_artifact_hypothesis": e.is_artifact_hypothesis,
            "acns": e.acns.model_dump() if e.acns else None,
            "metadata": e.metadata,
        }
        for e in detection.events
    ]
    nodes = [n.model_dump() for n in graph.nodes]
    edges = [ed.model_dump() for ed in graph.edges]
    seizure_burden = (
        detection.seizure_burden.model_dump() if detection.seizure_burden else None
    )
    allowed = (
        graph.grounding_keys()
        | {k for e in detection.events for k in ([e.code] + [ev.feature for ev in e.evidence])}
        | set(background.keys())
    )
    if seizure_burden:
        allowed.add("seizure_burden")
    return LLMInput(
        mode=mode,
        patient=patient.model_dump(),
        context=context.model_dump(),
        background=background,
        events=events,
        causal_nodes=nodes,
        causal_edges=edges,
        critical_flags=graph.critical_flags,
        implausible_flags=graph.implausible_flags,
        seizure_burden=seizure_burden,
        allowed_grounding_keys=sorted(allowed),
        gate=gate,
    )


class LLMProvider(ABC):
    """Layer-6 provider: turns the grounded input into an :class:`LLMReport`."""

    name: str = "provider"

    @abstractmethod
    def generate(self, data: LLMInput) -> LLMReport:
        ...
