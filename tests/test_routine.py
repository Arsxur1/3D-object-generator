"""Increment 17: routine-EEG specificity fixes (ACNS suppression, time-domain rhythmic
delta, time-locked periodic discharges, genuine PDR peak, age norms, uncertain tier)."""

import numpy as np
import pytest

from neurolens.contracts.events import Event, EventEvidence, Localization
from neurolens.contracts.signal import PatientInfo
from neurolens.layer3_features.norms import NormsEngine
from neurolens.layer3_features.peak_utils import spectral_peak
from neurolens.layer3_features.rhythmicity import delta_rhythmicity_map
from neurolens.layer3_features.suppression import (
    burst_suppression_acns, generalized_suppressed_mask, ptp_envelope,
)
from neurolens.layer4_detect.periodic import _consensus_discharges
from neurolens.layer5_reasoning.physiology import PhysiologyEngine

FS = 200.0


def _pink(rng, n, scale=10.0):
    """1/f noise with unit SD x scale (spectral shaping of white noise)."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / FS)
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    x = np.fft.irfft(spec, n)
    return scale * x / x.std()


def test_ptp_envelope_is_peak_to_trough():
    t = np.arange(int(4 * FS)) / FS
    env = ptp_envelope(4.0 * np.sin(2 * np.pi * 10 * t), FS)
    assert np.median(env) == pytest.approx(8.0, rel=0.05)


def test_low_voltage_eeg_is_not_suppression():
    """20-30 µV peak-to-trough background was 'suppressed' with the mean-|x| envelope."""
    rng = np.random.default_rng(0)
    n = int(60 * FS)
    x = np.stack([_pink(rng, n, 6.0) for _ in range(8)])  # SD 6 µV: ~ 20-30 µV ptp
    assert np.median(ptp_envelope(x[0], FS)) > 10
    assert generalized_suppressed_mask(x, FS).mean() < 0.05


def test_burst_suppression_quantified_per_channel():
    rng = np.random.default_rng(1)
    n = int(30 * FS)
    t = np.arange(n) / FS
    burst = ((t % 5.0) < 1.0).astype(float)
    x = np.stack([burst * 60 * np.sin(2 * np.pi * 3 * t + i) + rng.normal(0, 0.5, n) for i in range(8)])
    x = x - x.mean(axis=0, keepdims=True)  # average reference: the channel mean is ~0
    bs = burst_suppression_acns(x, FS)
    assert 0.5 <= bs.suppression_ratio <= 0.85 and bs.n_bursts >= 4


def test_rhythmic_delta_vs_aperiodic_background():
    rng = np.random.default_rng(2)
    n = int(30 * FS)
    t = np.arange(n) / FS
    rhythmic = np.stack([40 * np.sin(2 * np.pi * 2.0 * t + 0.3 * i) + rng.normal(0, 5, n) for i in range(6)])
    aperiodic = np.stack([_pink(rng, n, 3.0) for _ in range(6)])
    _, f, rho, _ = delta_rhythmicity_map(rhythmic, FS)
    assert np.median(rho) > 0.8 and np.median(f) == pytest.approx(2.0, abs=0.15)
    _, _, rho2, _ = delta_rhythmicity_map(aperiodic, FS)
    assert np.median(rho2) < 0.5


def test_spectral_peak_requires_a_real_peak():
    rng = np.random.default_rng(3)
    n = int(60 * FS)
    t = np.arange(n) / FS
    bg = _pink(rng, n, 3.0)
    assert spectral_peak(bg, FS) == 0.0  # 1/f only: no PDR (previously 4 Hz)
    assert spectral_peak(bg + 30 * np.sin(2 * np.pi * 9.5 * t), FS) == pytest.approx(9.5, abs=0.5)


def test_consensus_discharges_need_two_channels():
    times = {"C3": np.array([1.00, 2.00, 3.00]), "P3": np.array([1.02, 2.03, 5.0]), "O1": np.array([4.0])}
    d = _consensus_discharges(times)
    assert [x[0] for x in d] == pytest.approx([1.01, 2.015])
    assert d[0][1] == ["C3", "P3"]


@pytest.mark.parametrize("age,band", [(0, "infant"), (0.5, "infant"), (2.5, "toddler"),
                                      (5.5, "child_early"), (17.5, "child_school"), (40, "adult")])
def test_norm_bands_are_contiguous(configs, age, band):
    assert NormsEngine(configs.norms).band_for_patient(PatientInfo(age_years=age))[0] == band


def _ev(code, conf=0.5, meta=None, evidence=None, group="x"):
    return Event(code=code, label_ru=code, label_uz=code, group=group,
                 localization=Localization(channels=["C3"]), t_start=0.0, t_end=1.0,
                 confidence=conf, metadata=meta or {}, evidence=evidence or [])


def test_unvalidated_evidence_is_uncertain(configs):
    eng = PhysiologyEngine(NormsEngine(configs.norms))
    adult = PatientInfo(age_years=40)
    assert eng.label_event(_ev("spike", group="ied"), adult).value == "uncertain"
    assert eng.label_event(_ev("spike", group="ied", meta={"ied_probability_learned": 0.97}), adult).value == "pathologic"
    assert eng.label_event(_ev("spike", group="ied", meta={"detector": "ied_gbm_v1"}), adult).value == "pathologic"
    assert eng.label_event(_ev("ictal_ml", conf=0.2), adult).value == "uncertain"
    assert eng.label_event(_ev("ictal_ml", conf=0.8), adult).value == "pathologic"


def test_slowing_in_young_child_without_pdr_is_uncertain(configs):
    eng = PhysiologyEngine(NormsEngine(configs.norms))
    ds = _ev("diffuse_slowing", evidence=[EventEvidence(feature="pdr_hz", value=0.0)])
    assert eng.label_event(ds, PatientInfo(age_years=2)).value == "uncertain"
    assert eng.label_event(ds, PatientInfo(age_years=40)).value == "pathologic"
    infant_ok = _ev("diffuse_slowing", evidence=[EventEvidence(feature="pdr_hz", value=6.5)])
    assert eng.label_event(infant_ok, PatientInfo(age_years=0)).value == "uncertain"  # normal for age


def test_triphasic_dropped_without_gpds(configs, monkeypatch):
    from neurolens.layer4_detect import registry

    class Fixed:
        code = "fixed"

        def __init__(self, evs):
            self.evs = evs

        def detect(self, *a, **k):
            return self.evs

    class Sig:
        duration_s = 60.0
        patient = PatientInfo(age_years=40)

    tri = _ev("triphasic_waves", conf=0.8, group="special")
    res = registry.run_detectors(Sig(), None, configs.thresholds, None, detectors=[Fixed([tri])])
    assert not [e for e in res.events if e.code == "triphasic_waves"]
    gp = _ev("gpds", conf=0.8, group="periodic")
    res = registry.run_detectors(Sig(), None, configs.thresholds, None, detectors=[Fixed([tri, gp])])
    assert [e for e in res.events if e.code == "triphasic_waves"]


def test_mechanism_never_more_certain_than_its_trigger(configs):
    """'diffuse slowing -> encephalopathy' must not stay pathologic when the slowing
    itself is judged normal for age."""
    from neurolens.contracts.signal import ClinicalContext
    from neurolens.layer4_detect.registry import DetectionResult
    from neurolens.layer5_reasoning.rules_engine import RulesEngine

    eng = RulesEngine(configs.rule_base, PhysiologyEngine(NormsEngine(configs.norms)))
    ds = _ev("diffuse_slowing", conf=0.8, group="background",
             evidence=[EventEvidence(feature="pdr_hz", value=6.5)])
    g = eng.build(DetectionResult(events=[ds]), ClinicalContext(), PatientInfo(age_years=0.5))
    mech = [n for n in g.nodes if n.id.startswith("mech:")]
    assert mech and all(n.physiology.value != "pathologic" for n in mech)
    g2 = eng.build(DetectionResult(events=[ds]), ClinicalContext(), PatientInfo(age_years=40))
    assert any(n.physiology.value == "pathologic" for n in g2.nodes if n.id.startswith("mech:"))


def _rows(tn, n_n, tp, n_a, extra=None):
    rows = [{"abnormal": False, "impression_abnormal": i >= tn} for i in range(n_n)]
    rows += [{"abnormal": True, "impression_abnormal": i < tp} for i in range(n_a)]
    return {"records": rows, "summary": {"detector_rate": extra or {}}}


def test_prereg17_rule_and_apply(tmp_path):
    import yaml

    from neurolens.contracts.config import ImpressionConfig
    from neurolens.evaluation import prereg17

    ok = prereg17.assess(_rows(55, 95, 70, 90), _rows(2, 95, 90, 90))
    assert ok["decision"]["routine_impression_validated"]
    assert "подтверждена" in prereg17.markdown_report(ok)
    out = tmp_path / "impression.yaml"
    prereg17.apply_decision(ok, str(out))
    cfg = ImpressionConfig(**yaml.safe_load(out.read_text()))
    assert cfg.routine_validated and cfg.specificity == pytest.approx(55 / 95, abs=1e-3)
    low_sens = prereg17.assess(_rows(55, 95, 50, 90), _rows(2, 95, 90, 90))
    assert not low_sens["decision"]["routine_impression_validated"]
    prereg17.apply_decision(low_sens, str(out))
    assert ImpressionConfig(**yaml.safe_load(out.read_text())).routine_validated is False
    no_gain = prereg17.assess(_rows(55, 95, 70, 90), _rows(40, 95, 90, 90))
    assert not no_gain["hypotheses"]["H2_specificity_gain_at_least_0.30"]


def test_impression_caveat(configs):
    from neurolens.contracts.config import ImpressionConfig
    from neurolens.contracts.report import Bilingual
    from neurolens.pipeline.pipeline import _impression_with_validation

    imp = Bilingual(ru="Заключение: x.", uz="Xulosa: x.")
    assert "не прошло проверку" in _impression_with_validation(imp, ImpressionConfig()).ru
    v = _impression_with_validation(imp, ImpressionConfig(routine_validated=True, specificity=0.6, sensitivity=0.75))
    assert "специфичность 0.60" in v.ru and "sezgirlik 0.75" in v.uz
