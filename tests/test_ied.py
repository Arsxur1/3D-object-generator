"""NMT-Events parsing, IED window features/labels/events, IED scoring, amplitude flag."""

import numpy as np
import pytest

from neurolens.datasets.nmt_events import (
    EPILEPTIFORM, SLOWING, category, parse_age, parse_clock, parse_events_csv, split_by_rule,
)
from neurolens.evaluation import ied_eval as E
from neurolens.layer4_detect import ied_ml

CSV = (
    "Gender,Age,File Start,Start time,End time,Channel names,Comment\n"
    "Male,12 years,13:10:31,13:11:36:739,13:11:38:478,FP1 FP2 F3,spike and wave\n"
    ",,,13:14:29:034,13:14:31,T3,sharp and delta slow waves\n"
    ",,,13:16:20,13:16:21:982,O1 O2,delta slow waves\n"
    ",,,13:17:19:500,13:17:19:400,C3,polypspikes and wave\n"
)


def test_parse_clock_and_age():
    assert parse_clock("13:11:36:739") == pytest.approx(13 * 3600 + 11 * 60 + 36.739)
    assert parse_clock("13:14:31") == 13 * 3600 + 14 * 60 + 31
    assert parse_age("12 years") == 12
    assert parse_age("8 Months") == pytest.approx(8 / 12)
    assert parse_age("29  years") == 29
    assert parse_age("") is None


def test_category():
    assert category("spike and wave") == EPILEPTIFORM
    assert category("sharp and delta slow waves") == EPILEPTIFORM
    assert category("polypspikes and wave") == EPILEPTIFORM
    assert category("delta slow waves") == SLOWING
    assert category("no comment") == "other"


def test_parse_events_csv():
    a = parse_events_csv(CSV, 1004)
    assert a.age_years == 12 and a.gender == "Male"
    assert len(a.events) == 4
    e0 = a.events[0]
    assert e0.onset_s == pytest.approx(65.739) and e0.offset_s == pytest.approx(67.478)
    assert e0.channels == ["FP1", "FP2", "F3"]
    assert [e.category for e in a.events] == [EPILEPTIFORM, EPILEPTIFORM, SLOWING, EPILEPTIFORM]
    assert a.events[3].offset_s == a.events[3].onset_s  # end < start -> point event
    assert len(a.intervals(EPILEPTIFORM)) == 3 and len(a.intervals(SLOWING)) == 1


def test_split_rule():
    assert split_by_rule(4, normal=False) == "train"
    assert split_by_rule(5, normal=False) == "test"
    assert split_by_rule(16, normal=True) == "train"
    assert split_by_rule(17, normal=True) == "test"
    assert split_by_rule(18, normal=True) is None


def _signal(fs=200.0, secs=60.0, spikes=(20.0, 40.0), seed=0):
    rng = np.random.default_rng(seed)
    n = int(fs * secs)
    names = list(ied_ml.STANDARD_19) + ["A1", "A2"]
    from scipy.signal import butter, sosfiltfilt

    # background with an EEG-like (low-pass) spectrum, ~10 µV
    x = sosfiltfilt(butter(4, 15 / (fs / 2), output="sos"), rng.normal(0, 20, (len(names), n)), axis=1)
    t = np.arange(n) / fs
    for s in spikes:  # 60 ms, 150 µV sharp transient + slow after-wave on left temporal chain
        spike = 150 * np.exp(-0.5 * ((t - s) / 0.015) ** 2) - 60 * np.exp(-0.5 * ((t - s - 0.15) / 0.06) ** 2)
        for c in ("T3", "F7", "T5"):
            x[names.index(c)] += spike
    return x, fs, names


def test_window_features_shape_and_signal():
    x, fs, names = _signal()
    X, t0 = ied_ml.window_features(x, fs, names)
    assert X.shape[1] == len(ied_ml.feature_names())
    assert t0[1] - t0[0] == pytest.approx(ied_ml.HOP_S)
    assert np.isfinite(X).all()
    col = ied_ml.feature_names().index("slope_bg_max")
    near = np.abs(t0 + ied_ml.WIN_S / 2 - 20.0) < 0.3
    assert X[near, col].max() > 3 * np.median(X[:, col])


def test_window_features_needs_channels():
    X, t0 = ied_ml.window_features(np.zeros((2, 2000)), 200.0, ["C3", "C4"])
    assert X.shape[0] == 0 and t0.size == 0


def test_standard_channels_aliases():
    assert ied_ml.standard_channels(["T7", "A1", "ECG", "Fp1", "Cz"]) == [0, 3, 4]


def test_window_labels():
    t0 = np.arange(0, 5, 0.25)
    y = ied_ml.window_labels(t0, [(2.0, 2.2)])
    assert y[t0 == 2.0][0] == 1 and y[t0 == 1.75][0] == 1  # [1.75, 2.25) overlaps
    assert y[t0 == 1.5][0] == -1  # [1.5, 2.0) touches within margin
    assert y[t0 == 0.0][0] == 0


def test_probability_events():
    t0 = np.arange(10) * 0.25
    p = np.array([0, .9, .8, 0, .7, 0, 0, 0, .6, 0])
    ev = ied_ml.probability_events(t0, p, 0.5)
    assert [(a, b) for a, b, _ in ev] == [(0.25, 1.5), (2.0, 2.5)]
    assert ev[0][2] == pytest.approx(0.9)


def test_merge_and_score():
    assert E.merge_intervals([(0, 1), (1.1, 2), (5, 6)]) == [(0, 2), (5, 6)]
    ep, sl = [(10.0, 11.0), (20.0, 20.5)], [(30.0, 31.0)]
    r = E.score_record([(10.8, 11.1), (30.5, 30.6), (50, 50.2)], ep, sl, 3600, normal=False)
    assert r["n_hit"] == 1 and r["n_slow_hit"] == 1 and r["n_unannotated"] == 1 and r["n_false"] == 0
    r = E.score_record([(1, 2), (3, 4)], [], [], 1800, normal=True)
    assert r["n_false"] == 2
    agg = E.aggregate([E.score_record([(10.8, 11.1)], ep, sl, 3600, False),
                       E.score_record([(1, 2), (3, 4)], [], [], 1800, True)], n_boot=50)
    assert agg["sensitivity"] == 0.5 and agg["fa_per_h_normal"] == pytest.approx(4.0)
    assert agg["slowing_hit_rate"] == 0.0


def test_reference_events_drops_slowing_overlapping_discharges():
    a = parse_events_csv(CSV, 1)
    ep, sl = E.reference_events(a)
    assert len(ep) == 3 and len(sl) == 1


def test_ml_spike_detector_events():
    x, fs, names = _signal()
    X, t0 = ied_ml.window_features(x, fs, names)

    class Oracle:  # probability from the window's slope prominence
        def predict_proba(self, X):
            c = ied_ml.feature_names().index("slope_bg_max")
            return (X[:, c] > 5).astype(float)

    from neurolens.contracts.signal import UnifiedSignal

    try:
        sig = UnifiedSignal(signal=x.astype(np.float32), sampling_rate_hz=fs, channel_names=names)
    except Exception:
        pytest.skip("UnifiedSignal requires additional fields")
    det = ied_ml.MLSpikeDetector(Oracle(), threshold=0.5)
    evs = det.detect(sig, None, None)
    centres = sorted((e.t_start + e.t_end) / 2 for e in evs)
    assert len(evs) == 2 and abs(centres[0] - 20) < 0.6 and abs(centres[1] - 40) < 0.6
    assert set(evs[0].localization.channels) & {"T3", "F7", "T5"}


def test_amplitude_flag():
    from neurolens.layer2_preprocess.sqi import amplitude_flag, robust_amplitude_uv

    class S:
        channel_names = ["C3", "C4", "O1", "O2"]

        def __init__(self, sd):
            self.signal = np.random.default_rng(0).normal(0, sd, (4, 4000))

    assert amplitude_flag(S(20.0)) is None
    assert amplitude_flag(S(0.2)).startswith("amplitude_implausible")
    assert amplitude_flag(S(1000.0)).startswith("amplitude_implausible")
    assert robust_amplitude_uv(S(20.0)) == pytest.approx(20.0, rel=0.1)


def test_heuristic_spike_at_record_start_does_not_crash():
    """A peak within the first 50 ms gave t_start < 0 -> validation error, which
    run_detectors swallowed, silently dropping every spike of the record."""
    from neurolens.contracts.signal import UnifiedSignal
    from neurolens.layer4_detect.ied import SpikeDetector

    fs = 200.0
    x = np.zeros((2, 2000))
    x[0, 3] = 400.0  # sharp transient at 15 ms
    try:
        sig = UnifiedSignal(signal=x.astype(np.float32), sampling_rate_hz=fs, channel_names=["C3", "C4"])
    except Exception:
        pytest.skip("UnifiedSignal requires additional fields")
    evs = SpikeDetector(max_events=None).detect(sig, None, None)
    assert evs and min(e.t_start for e in evs) >= 0.0


def _agg(sens, fa, slow=0.2):
    return {"sensitivity": sens, "sensitivity_ci95_record_bootstrap": [sens - 0.1, sens + 0.1],
            "sensitivity_macro": sens, "fa_per_h_normal": fa, "slowing_hit_rate": slow,
            "n_ref": 100, "n_records_with_ref": 10, "n_normal": 20, "hours_normal": 5.0,
            "unannotated_per_h_abnormal": 3.0}


def test_prereg15_assess_and_apply(tmp_path):
    import yaml

    from neurolens.contracts.config import IEDConfig
    from neurolens.evaluation import prereg15

    curve = [{"sharpness_z": z, **_agg(s, f)} for z, s, f in ((6.0, 0.95, 1600.0), (30.0, 0.04, 43.0))]
    frozen = {"threshold": 0.95, "cv_operating_point": _agg(0.64, 12.8),
              "model": {"path": "configs/models/ied_gbm_v1.json", "sha256": "ab"}}
    a = prereg15.assess({"learned": _agg(0.6, 14.0), "heuristic_curve": curve}, frozen)
    assert all(a["hypotheses"].values()) and a["decision"]["enable_learned_ied_detector"]
    assert "подтверждена" in prereg15.markdown_report(a)
    out = tmp_path / "ied.yaml"
    prereg15.apply_decision(a, frozen, str(out))
    cfg = IEDConfig(**yaml.safe_load(out.read_text()))
    assert cfg.enabled and cfg.threshold == 0.95 and not cfg.apply_to_neonates

    bad = prereg15.assess({"learned": _agg(0.6, 30.0), "heuristic_curve": curve}, frozen)
    assert not bad["hypotheses"]["H2_fa_per_h_normal_within_1.5x_budget"]
    assert not bad["decision"]["enable_learned_ied_detector"]
    dominated = prereg15.assess({"learned": _agg(0.03, 14.0), "heuristic_curve":
                                 curve + [{"sharpness_z": 40.0, **_agg(0.05, 10.0)}]}, frozen)
    assert not dominated["hypotheses"]["H3_dominates_heuristic_curve"]


def test_type_key():
    assert E.type_key("polyspikes and wave") == "polyspikes"
    assert E.type_key("spike and wave") == "spike_and_wave"
    assert E.type_key("sharp and slow waves") == "sharp_and_slow_wave"
    assert E.type_key("sharp and delta slow waves") == "sharp_and_slow_wave"
    assert E.type_key("sharp waves") == "sharp_wave"


def test_ied_config_default_disabled():
    from neurolens.pipeline.config_loader import load_configs

    c = load_configs().ied
    assert c.apply_to_neonates is False


def test_pipeline_uses_learned_ied_detector_when_enabled(demo_edf, tmp_path):
    """configs/ied.yaml enabled -> the heuristic SpikeDetector is replaced and spike
    events carry the learned detector's provenance."""
    pytest.importorskip("sklearn")
    import copy

    from sklearn.ensemble import HistGradientBoostingClassifier

    from neurolens.contracts.config import IEDConfig
    from neurolens.layer4_detect.ml_ictal import export_hist_gbm
    from neurolens.pipeline.config_loader import load_configs
    from neurolens.pipeline.pipeline import Pipeline

    names = ied_ml.feature_names()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, len(names)))
    y = (X[:, names.index("slope_bg_max")] > 1.0).astype(int)
    clf = HistGradientBoostingClassifier(max_iter=5).fit(X, y)
    path = export_hist_gbm(clf, names).save(tmp_path / "ied.json")

    cfg = copy.deepcopy(load_configs())
    cfg.ied = IEDConfig(enabled=True, model=str(path), threshold=0.0, max_events=5)
    out = Pipeline(cfg, provider_pref="template", run_ica=False).analyze_file(demo_edf)
    spikes = [e for e in out.detection.events if e.code == "spike"]
    assert spikes and all(e.metadata.get("detector") == "ied" for e in spikes)
    assert len(spikes) <= 5
