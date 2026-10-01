"""End-to-end pipeline tests."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from neurolens.contracts.report import OperatingMode
from neurolens.contracts.signal import ClinicalContext, PatientInfo
from neurolens.pipeline.pipeline import Pipeline

SCHEMAS = Path(__file__).resolve().parents[1] / "schemas"


def _run(demo_edf, configs, mode=OperatingMode.A_DECISION_SUPPORT):
    pipe = Pipeline(config=configs, provider_pref="deterministic")
    return pipe, pipe.analyze_file(
        demo_edf, mode=mode, montage_name="double_banana",
        patient=PatientInfo(age_years=55),
        context=ClinicalContext(sedatives=["propofol"], temperature_c=36.5),
    )


def test_e2e_produces_expected_events(demo_edf, configs):
    _, out = _run(demo_edf, configs)
    codes = out.detection.codes()
    assert "diffuse_slowing" in codes
    assert out.detection.by_group("ictal")
    assert "burst_suppression" in codes
    assert "ecg_artifact" in codes


def test_e2e_causal_edges(demo_edf, configs):
    _, out = _run(demo_edf, configs)
    rule_ids = {e.rule_id for e in out.graph.edges}
    assert "diffuse_slowing_encephalopathy" in rule_ids
    assert "ecg_artifact_vs_ied" in rule_ids


def test_e2e_critical_findings_present(demo_edf, configs):
    _, out = _run(demo_edf, configs)
    codes = {cf.code for cf in out.report.critical_findings}
    # sustained ictal -> status; burst-suppression -> critical
    assert "status_epilepticus_suspected" in codes
    assert "burst_suppression" in codes


def test_e2e_json_validates_against_schemas(demo_edf, configs):
    _, out = _run(demo_edf, configs)
    result = out.result_json

    causal_schema = json.loads((SCHEMAS / "causal_graph.schema.json").read_text())
    jsonschema.validate(result["causal_graph"], causal_schema)

    event_schema = json.loads((SCHEMAS / "event.schema.json").read_text())
    for ev in result["events"]:
        jsonschema.validate(ev, event_schema)

    report_schema = json.loads((SCHEMAS / "llm_report.schema.json").read_text())
    jsonschema.validate(result["report"], report_schema)


def test_e2e_deterministic_provider_when_forced(demo_edf, configs):
    _, out = _run(demo_edf, configs)
    assert out.report.provider == "deterministic"


def test_e2e_save_outputs(demo_edf, configs, tmp_path):
    pipe, out = _run(demo_edf, configs)
    paths = pipe.save_outputs(out, tmp_path / "out")
    assert paths["json"].exists()
    assert paths["protocol"].exists()
    # protocol has both languages and the disclaimer
    txt = paths["protocol"].read_text(encoding="utf-8")
    assert "ПРОТОКОЛ ЭЭГ" in txt and "Xulosa" in txt


def test_mode_b_gate_escalates_on_low_quality(demo_edf):
    # fresh configs (do not mutate the shared session fixture)
    from neurolens.pipeline.config_loader import load_configs

    cfg = load_configs()
    cfg.thresholds.mode_b.min_confidence = 0.999  # unreachable -> must escalate
    _, out = _run(demo_edf, cfg, mode=OperatingMode.B_AUTONOMOUS)
    assert out.report.mode_gate_passed is False
    assert out.report.escalation_reason is not None
