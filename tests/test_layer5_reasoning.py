"""Layer 5 causal-reasoning tests."""

from __future__ import annotations

from neurolens.contracts.causal import PhysiologyLabel
from neurolens.contracts.events import DetectionResult, Event, EventEvidence, Localization
from neurolens.contracts.signal import ClinicalContext, PatientInfo
from neurolens.layer2_preprocess.artifacts import ArtifactReport
from neurolens.layer3_features.norms import NormsEngine
from neurolens.layer5_reasoning.physiology import PhysiologyEngine
from neurolens.layer5_reasoning.rules_engine import build_causal_graph


def _engine(configs):
    return PhysiologyEngine(NormsEngine(configs.norms))


def _event(code, group, **kw):
    return Event(code=code, label_ru=code, label_uz=code, group=group,
                 localization=Localization(), t_start=0.0, t_end=10.0,
                 confidence=kw.get("confidence", 0.8),
                 evidence=kw.get("evidence", []),
                 is_artifact_hypothesis=kw.get("artifact", False))


def test_diffuse_slowing_creates_encephalopathy_edge(configs):
    det = DetectionResult(events=[_event("diffuse_slowing", "background",
                                         evidence=[EventEvidence(feature="pdr_hz", value=5.0)])])
    graph = build_causal_graph(det, ClinicalContext(), PatientInfo(age_years=50),
                               configs.rule_base, _engine(configs))
    rule_ids = {e.rule_id for e in graph.edges}
    assert "diffuse_slowing_encephalopathy" in rule_ids


def test_ecg_artifact_resolves_spikes(configs):
    spike = _event("spike", "ied", confidence=0.5)
    spike.localization = Localization(channels=["T3"])
    det = DetectionResult(events=[_event("ecg_artifact", "artifact", artifact=True), spike])
    art = ArtifactReport(ecg_present=True, heart_rate_hz=1.0,
                         ecg_qrs_times_s=[5.0], ecg_contaminated_channels=["T3"])
    graph = build_causal_graph(det, ClinicalContext(), PatientInfo(age_years=50),
                               configs.rule_base, _engine(configs), art)
    # spike node re-labeled as artifact
    spike_nodes = [n for n in graph.nodes if n.event_code == "spike"]
    assert spike_nodes and spike_nodes[0].physiology == PhysiologyLabel.ARTIFACT
    # pseudo-IED mechanism node exists
    assert any(n.id == "mech:pseudo_ied" for n in graph.nodes)


def test_burst_suppression_is_critical(configs):
    det = DetectionResult(events=[_event("burst_suppression", "suppression")])
    graph = build_causal_graph(det, ClinicalContext(sedatives=["propofol"]),
                               PatientInfo(age_years=50), configs.rule_base, _engine(configs))
    assert graph.has_critical()
