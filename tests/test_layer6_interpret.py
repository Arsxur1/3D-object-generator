"""Layer 6 interpretation & grounding tests."""

from __future__ import annotations

from neurolens.contracts.causal import (
    CausalEdge,
    CausalGraph,
    CausalNode,
    NodeKind,
    PhysiologyLabel,
)
from neurolens.contracts.events import DetectionResult, Event, EventEvidence, Localization
from neurolens.contracts.report import Bilingual, LLMReport, OperatingMode, ReportSection
from neurolens.contracts.signal import ClinicalContext, PatientInfo
from neurolens.layer3_features.feature_set import compute_features
from neurolens.layer6_interpret.deterministic_provider import DeterministicProvider
from neurolens.layer6_interpret.grounding import validate_grounding
from neurolens.layer6_interpret.llm_base import build_llm_input, evaluate_mode_b_gate
from neurolens.layer6_interpret.protocol import render_protocol


def _minimal_report():
    return LLMReport(
        mode=OperatingMode.A_DECISION_SUPPORT,
        sections=[
            ReportSection(key="background", text=Bilingual(ru="x", uz="x"),
                          grounding_refs=["pdr_hz"]),
            ReportSection(key="causal", text=Bilingual(ru="y", uz="y"),
                          grounding_refs=["not_a_real_key"]),
        ],
        impression=Bilingual(ru="i", uz="i"),
        overall_confidence={"value": 0.6, "label_ru": "средняя", "label_uz": "o‘rtacha"},
    )


def test_grounding_drops_ungrounded_section():
    report = _minimal_report()
    res = validate_grounding(report, allowed_keys={"pdr_hz"})
    keys = {s.key for s in report.sections}
    assert "background" in keys
    assert "causal" not in keys  # dropped: no valid grounding
    assert res.fully_grounded is False
    assert "causal" in res.dropped_sections


def test_grounding_keeps_all_valid():
    report = _minimal_report()
    res = validate_grounding(report, allowed_keys={"pdr_hz", "not_a_real_key"})
    assert res.fully_grounded is True
    assert len(report.sections) == 2


def test_deterministic_provider_is_grounded(demo_edf, configs):
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.layer2_preprocess.filters import apply_filters
    from neurolens.layer2_preprocess.reref import rereference
    from neurolens.layer4_detect.registry import run_detectors
    from neurolens.layer5_reasoning.physiology import PhysiologyEngine
    from neurolens.layer5_reasoning.rules_engine import build_causal_graph
    from neurolens.layer3_features.norms import NormsEngine

    sig = rereference(apply_filters(ingest(demo_edf), configs.filters), "average")
    feats = compute_features(sig, configs.filters)
    det = run_detectors(sig, feats, configs.thresholds)
    graph = build_causal_graph(det, ClinicalContext(), sig.patient, configs.rule_base,
                               PhysiologyEngine(NormsEngine(configs.norms)))
    gate = evaluate_mode_b_gate(OperatingMode.A_DECISION_SUPPORT, feats, det, graph,
                                0.9, 0.1, configs.thresholds)
    data = build_llm_input(OperatingMode.A_DECISION_SUPPORT, sig.patient,
                           ClinicalContext(), feats, det, graph, gate)
    report = DeterministicProvider().generate(data)
    res = validate_grounding(report, set(data.allowed_grounding_keys))
    # nothing should have been dropped: deterministic provider is grounded by construction
    assert res.dropped_sections == []
    assert render_protocol(report).startswith("=")
