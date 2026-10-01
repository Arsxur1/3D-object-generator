"""Neonatal specialization tests (TZ §1, §6): aEEG classification, PMA norms."""

from __future__ import annotations

import numpy as np
import pytest

from neurolens.contracts.events import DetectionResult, Event, Localization
from neurolens.contracts.signal import ClinicalContext, PatientInfo
from neurolens.layer3_features.aeeg import AEEG
from neurolens.layer3_features.aeeg_classify import AeegCategory, classify_aeeg, detect_swc
from neurolens.layer3_features.norms import NormsEngine
from neurolens.layer5_reasoning.physiology import PhysiologyEngine
from neurolens.layer5_reasoning.rules_engine import build_causal_graph


def _aeeg(upper, lower, lower_series=None):
    n = 16
    up = np.full(n, upper, dtype=float)
    lo = np.full(n, lower, dtype=float) if lower_series is None else np.asarray(lower_series, float)
    return AEEG(times_s=np.arange(n) * 15.0, upper_uv=up, lower_uv=lo,
               channel="C3", upper_raw=up, lower_raw=lo)


# --- aEEG classification --------------------------------------------------
def test_classify_categories():
    assert classify_aeeg(_aeeg(20, 7)).category == AeegCategory.CNV
    assert classify_aeeg(_aeeg(20, 3)).category == AeegCategory.DNV
    assert classify_aeeg(_aeeg(30, 1)).category == AeegCategory.BS
    assert classify_aeeg(_aeeg(3, 1)).category == AeegCategory.FT
    assert classify_aeeg(_aeeg(8, 3)).category == AeegCategory.CLV


def test_bs_via_suppression_ratio():
    # DNV-looking margins but a high suppression ratio -> burst-suppression
    assert classify_aeeg(_aeeg(20, 3), burst_suppression_ratio=0.6).category == AeegCategory.BS


def test_swc_detection():
    t = np.arange(32)
    modulated = 4.0 + 3.0 * np.sin(2 * np.pi * t / 16.0)  # two cycles
    present, cycles = detect_swc(_aeeg(20, 4, lower_series=modulated))
    assert present and cycles >= 1
    flat_present, _ = detect_swc(_aeeg(20, 4, lower_series=np.full(32, 4.0)))
    assert flat_present is False


# --- maturational norms ---------------------------------------------------
def test_pma_norm_bands(configs):
    eng = NormsEngine(configs.norms)
    preterm = eng.neonatal_expectations(PatientInfo(postmenstrual_age_weeks=26))
    term = eng.neonatal_expectations(PatientInfo(postmenstrual_age_weeks=40))
    assert preterm["expected_aeeg"] == "DNV" and preterm["swc_expected"] is False
    assert term["expected_aeeg"] == "CNV" and term["swc_expected"] is True
    # discontinuity (continuity ~0.5) is normal preterm, abnormal at term
    assert eng.is_continuity_normal_for_pma(0.5, PatientInfo(postmenstrual_age_weeks=26)) is True
    assert eng.is_continuity_normal_for_pma(0.5, PatientInfo(postmenstrual_age_weeks=40)) is False


def test_norms_none_for_non_neonate(configs):
    eng = NormsEngine(configs.norms)
    assert eng.neonatal_expectations(PatientInfo(age_years=40)) is None


# --- neonatal causal rule -------------------------------------------------
def test_neonatal_bs_triggers_hie(configs):
    eng = PhysiologyEngine(NormsEngine(configs.norms))
    det = DetectionResult(events=[Event(
        code="neonatal_burst_suppression", label_ru="x", label_uz="x", group="suppression",
        localization=Localization(), t_start=0, t_end=60, confidence=0.75)])
    graph = build_causal_graph(det, ClinicalContext(), PatientInfo(postmenstrual_age_weeks=40),
                               configs.rule_base, eng)
    assert "neonatal_bs_hie" in {e.rule_id for e in graph.edges}
    assert graph.has_critical()


# --- end-to-end via pipeline ---------------------------------------------
@pytest.fixture(scope="module")
def neonatal_edf(tmp_path_factory):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "synthetic"))
    from make_synthetic_edf import make_synthetic_edf

    return make_synthetic_edf(tmp_path_factory.mktemp("neo") / "neo.edf",
                              scenario="neonatal", duration_s=240)


def _analyze(edf, configs, pma):
    from neurolens.pipeline.pipeline import Pipeline

    return Pipeline(config=configs, provider_pref="deterministic").analyze_file(
        edf, patient=PatientInfo(postmenstrual_age_weeks=pma), montage_name="neonatal")


def test_neonatal_detected_and_age_appropriate(neonatal_edf, configs):
    preterm = _analyze(neonatal_edf, configs, 28)
    term = _analyze(neonatal_edf, configs, 40)

    for out in (preterm, term):
        nb = [e for e in out.detection.events if e.code == "neonatal_background"]
        assert nb, "expected a neonatal_background event"
        assert nb[0].metadata["aeeg_category"] in {"DNV", "CNV", "BS", "CLV", "FT"}

    def phys(out):
        return next(n.physiology.value for n in out.graph.nodes if n.event_code == "neonatal_background")

    # same recording: appropriate (physiologic) preterm, abnormal (pathologic) at term
    assert phys(preterm) == "physiologic"
    assert phys(term) == "pathologic"


def test_neonatal_not_run_without_pma(demo_edf, configs):
    from neurolens.pipeline.pipeline import Pipeline

    out = Pipeline(config=configs, provider_pref="deterministic").analyze_file(demo_edf)
    assert "neonatal_background" not in out.detection.codes()
