"""Learned ictal detector: featurizer, models, post-processing, streaming twin."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from neurolens.layer4_detect.ml_ictal import (
    FEATURE_NAMES,
    LogisticModel,
    MLDetectorConfig,
    MLIctalDetector,
    StreamingFeaturizer,
    TreeEnsembleModel,
    featurize,
    fit_logistic,
    load_model,
    probability_runs,
)


def _fake_features(n=400, n_ch=6, seizure=(200, 260), seed=0):
    rng = np.random.default_rng(seed)
    rms = 20 * np.exp(rng.normal(0, 0.1, size=(n, n_ch)))
    conc = np.clip(rng.normal(0.3, 0.05, size=(n, n_ch)), 0, 1)
    domf = rng.normal(9, 1, size=(n, n_ch))
    a, b = seizure
    rms[a:b] *= 4.0
    conc[a:b] = 0.8
    domf[a:b] = np.linspace(7, 3, b - a)[:, None]
    rp = {k: np.full((n, n_ch), 0.2) for k in ("delta", "theta", "alpha", "beta", "gamma")}
    return SimpleNamespace(epoch_times=np.arange(1, n + 1, dtype=float), epoch_rms=rms,
                           epoch_band_conc=conc, epoch_domfreq=domf, epoch_relpow=rp,
                           eeg_channels=[f"C{i}" for i in range(n_ch)])


def test_featurize_shape_and_causality():
    f = _fake_features()
    X = featurize(f)
    assert X.shape == (400, len(FEATURE_NAMES)) and np.isfinite(X).all()
    # causal: changing the future must not change past rows
    g = _fake_features()
    g.epoch_rms[300:] *= 10
    np.testing.assert_allclose(featurize(g)[:300], X[:300])
    z = FEATURE_NAMES.index("z_top3")
    assert X[230, z] > X[150, z] + 1.0  # seizure amplitude stands out vs own baseline


def test_streaming_featurizer_matches_offline():
    f = _fake_features(n=777)
    X = featurize(f)
    sf = StreamingFeaturizer()
    parts = []
    for a in range(0, 777, 7):
        b = min(777, a + 7)
        parts.append(sf.push(f.epoch_rms[a:b], f.epoch_band_conc[a:b], f.epoch_domfreq[a:b],
                             {k: v[a:b] for k, v in f.epoch_relpow.items()}))
    np.testing.assert_allclose(np.vstack(parts), X, atol=1e-10)


def test_probability_runs():
    p = np.zeros(100)
    p[10:30] = 0.95
    p[50:53] = 0.99  # too short after smoothing
    runs = probability_runs(p, threshold=0.8, min_epochs=5, smooth=5)
    assert len(runs) == 1 and runs[0][0] >= 10 and runs[0][1] <= 31


def test_logistic_fit_and_json_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4000, len(FEATURE_NAMES)))
    y = (X[:, 0] + 0.5 * X[:, 3] > 1.5).astype(float)
    m = fit_logistic(X, y)
    assert m.meta["converged"]
    p = m.predict_proba(X)
    assert p[y == 1].mean() > 0.8 and p[y == 0].mean() < 0.3
    path = m.save(tmp_path / "lr.json")
    m2 = load_model(path)
    assert isinstance(m2, LogisticModel)
    np.testing.assert_allclose(m2.predict_proba(X), p, atol=1e-6)


def test_tree_export_matches_sklearn(tmp_path):
    sk = pytest.importorskip("sklearn.ensemble")
    from neurolens.layer4_detect.ml_ictal import export_hist_gbm

    rng = np.random.default_rng(2)
    X = rng.normal(size=(5000, len(FEATURE_NAMES)))
    X[rng.random(X.shape) < 0.02] = np.nan
    y = (np.nan_to_num(X[:, 0]) ** 2 + np.nan_to_num(X[:, 2]) > 1.2).astype(int)
    clf = sk.HistGradientBoostingClassifier(max_iter=30, random_state=0).fit(X, y)
    m = export_hist_gbm(clf, FEATURE_NAMES)
    path = m.save(tmp_path / "gbm.json")
    m2 = load_model(path)
    assert isinstance(m2, TreeEnsembleModel)
    np.testing.assert_allclose(m2.predict_proba(X), clf.predict_proba(X)[:, 1], atol=1e-12)
    assert json.loads(path.read_text())["type"] == "tree_ensemble"


def test_ml_detector_finds_injected_seizure():
    train = [_fake_features(seed=s, seizure=(100 + 20 * s, 160 + 20 * s)) for s in range(4)]
    Xs, ys = [], []
    for s, f in enumerate(train):
        X = featurize(f)
        y = np.zeros(len(X))
        y[100 + 20 * s:160 + 20 * s] = 1
        Xs.append(X)
        ys.append(y)
    model = fit_logistic(np.vstack(Xs), np.concatenate(ys))
    det = MLIctalDetector(model, MLDetectorConfig(threshold=0.8, min_epochs=10))
    events = det.detect(None, _fake_features(seed=9, seizure=(250, 320)), None)
    assert len(events) == 1
    e = events[0]
    assert 240 <= e.t_start <= 275 and e.group == "ictal" and e.code == "ictal_ml"


def test_prereg10_assessment_logic():
    from neurolens.evaluation.prereg10 import assess, macro_sensitivity

    def rec(file, n, tp, fp, hours=5.0):
        return {"file": file, "hours": hours, "n_seizures": n, "tp": tp, "fp": fp, "latencies_s": [5.0] * tp}

    def mode(name, big_tp, small_tp, fp):
        # one patient with 40 seizures, one with 4: macro and micro differ
        return {"mode": name, "records": [rec("chb12/a.edf", 40, big_tp, fp), rec("chb04/b.edf", 4, small_tp, fp)]}

    metrics = {"results": [
        mode("offline-threshold-default", 30, 4, 40), mode("offline-ml-A", 36, 4, 30), mode("offline-ml-B", 20, 2, 4),
        mode("realtime-threshold-default", 30, 4, 20), mode("realtime-ml-A", 36, 2, 18), mode("realtime-ml-B", 30, 3, 4),
    ]}
    a = assess(metrics)
    assert macro_sensitivity(metrics["results"][4]["records"]) == round((36 / 40 + 2 / 4) / 2, 4)
    h = a["hypotheses"]
    assert h["H1_realtime_mlA_noninferior_sens_and_no_more_FA"] is True   # 38/44 vs 34/44, 3.6 vs 4.0 FA/h
    assert h["H2_offline_mlA_noninferior_sens_and_no_more_FA"] is True
    assert h["H3_realtime_mlB_quiet_FA_le_1_and_sens_ge_0.6"] is True      # 33/44, 0.8 FA/h
    assert h["H4_realtime_mlA_macro_sens_noninferior"] is False            # macro 0.70 vs 0.875
    assert a["decision"]["ml_becomes_default_realtime_seizure_detector"] is True


class _StubModel:
    """Fires when amplitude vs own baseline is high (z_top3 feature)."""

    meta = {"name": "stub"}

    def predict_proba(self, X):
        z = X[:, FEATURE_NAMES.index("z_top3")]
        return 1.0 / (1.0 + np.exp(-(z - 0.7) * 20))


def _windows(f, win=30, step=5):
    """Split a stream of epochs (1 s step, centres 1..n) into overlapping monitor windows."""
    n = f.epoch_rms.shape[0]
    out = []
    for t0 in range(0, n - win + 1, step):
        sl = slice(t0, t0 + win - 1)
        out.append((float(t0), SimpleNamespace(
            epoch_times=np.arange(1, win, dtype=float), epoch_rms=f.epoch_rms[sl],
            epoch_band_conc=f.epoch_band_conc[sl], epoch_domfreq=f.epoch_domfreq[sl],
            epoch_relpow={k: v[sl] for k, v in f.epoch_relpow.items()}, eeg_channels=f.eeg_channels)))
    return out


def test_streaming_ml_matches_offline_runs():
    from neurolens.realtime.ml_stream import StreamingIctalML

    f = _fake_features(n=600, seizure=(300, 380))
    model = _StubModel()
    stream = StreamingIctalML(model, threshold=0.8, min_epochs=10)
    first_alarm = None
    active_windows = 0
    for t0, w in _windows(f):
        ev = stream.update(t0, w)
        if ev:
            active_windows += 1
            first_alarm = first_alarm or (t0 + ev[0].t_start, t0 + ev[0].t_end)
            assert ev[0].group == "ictal" and ev[0].code == "ictal_ml"
    # offline reference on the same epoch stream (window epochs cover times 1..n-1)
    seen = SimpleNamespace(**{k: getattr(f, k) for k in ("epoch_rms", "epoch_band_conc", "epoch_domfreq",
                                                          "epoch_relpow", "eeg_channels")})
    m = (600 // 5) * 5 - 1
    seen.epoch_rms, seen.epoch_band_conc, seen.epoch_domfreq = (f.epoch_rms[:m], f.epoch_band_conc[:m],
                                                                f.epoch_domfreq[:m])
    seen.epoch_relpow = {k: v[:m] for k, v in f.epoch_relpow.items()}
    runs = probability_runs(model.predict_proba(featurize(seen)), 0.8, 10)
    assert len(runs) == 1 and active_windows > 0
    a, b, _ = runs[0]
    assert first_alarm[0] == float(a + 1)  # epoch index a has centre time a+1


def test_pipeline_and_monitor_with_learned_detector(demo_edf, configs):
    import copy

    from neurolens.pipeline.pipeline import Pipeline
    from neurolens.realtime.monitor import RealtimeMonitor
    from neurolens.realtime.stream import EdfReplaySource

    cfg = copy.deepcopy(configs)
    assert cfg.ml.offline is not None and cfg.ml.realtime is not None
    out = Pipeline(config=cfg, provider_pref="deterministic", run_ica=False, learned=True).analyze_file(demo_edf)
    assert "ictal_ml" in out.detection.detectors_run and "ictal_rhythm" not in out.detection.detectors_run
    mon = RealtimeMonitor(cfg, learned=True)
    assert mon._ml is not None and not any(type(d).__name__ == "IctalRhythmDetector" for d in mon.detectors)
    summary = mon.run(EdfReplaySource(demo_edf, chunk_s=cfg.realtime.step_s))
    assert summary.n_windows > 0
    # default follows configs/ml.yaml (increment 11 decision: learned v2 in the monitor);
    # the threshold detector stays available explicitly
    assert RealtimeMonitor(cfg)._ml is not None
    assert cfg.ml.realtime.model.endswith("ictal_gbm_realtime_v2.json")
    assert RealtimeMonitor(cfg, learned=False)._ml is None


def _with_morph(f, seed=0):
    rng = np.random.default_rng(seed)
    ll = f.epoch_rms * 0.3 * np.exp(rng.normal(0, 0.05, size=f.epoch_rms.shape))
    te = (f.epoch_rms ** 2) * 0.05
    f.extra = {"epoch_linelen": ll, "epoch_teager": te}
    return f


def test_feature_set_v2_morphology_and_streaming():
    from neurolens.layer4_detect.ml_ictal import FEATURE_NAMES_V2, StreamingFeaturizerV2, feature_names

    f = _with_morph(_fake_features(n=500))
    X1, X2 = featurize(f), featurize(f, version=2)
    assert X2.shape == (500, len(FEATURE_NAMES_V2)) == (500, len(feature_names(2)))
    np.testing.assert_allclose(X2[:, :len(FEATURE_NAMES)], X1)  # v1 columns unchanged
    sf = StreamingFeaturizerV2()
    parts = [sf.push(f.epoch_rms[a:a + 5], f.epoch_band_conc[a:a + 5], f.epoch_domfreq[a:a + 5],
                     {k: v[a:a + 5] for k, v in f.epoch_relpow.items()},
                     f.extra["epoch_linelen"][a:a + 5], f.extra["epoch_teager"][a:a + 5])
             for a in range(0, 500, 5)]
    np.testing.assert_allclose(np.vstack(parts), X2, atol=1e-10)
    ll = FEATURE_NAMES_V2.index("ll_z_top3")
    assert X2[230, ll] > X2[150, ll] + 0.5  # seizure line length stands out vs own baseline


def test_feature_set_v2_requires_layer3_morphology():
    with pytest.raises(ValueError):
        featurize(_fake_features(n=100), version=2)


def test_layer3_emits_morphology(demo_edf, configs):
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.layer2_preprocess.filters import apply_filters
    from neurolens.layer2_preprocess.reref import rereference
    from neurolens.layer3_features.feature_set import FEATURE_VERSION, compute_features

    sig = rereference(apply_filters(ingest(demo_edf), configs.filters), "average")
    f = compute_features(sig, configs.filters)
    assert f.extra["feature_version"] == FEATURE_VERSION >= 2
    assert f.extra["epoch_linelen"].shape == f.epoch_rms.shape
    assert np.all(f.extra["epoch_linelen"] > 0)


def test_platt_scaling_recovers_low_precision(tmp_path):
    from neurolens.calibration.calibrator import ConfidenceCalibrator
    from neurolens.calibration.metrics import expected_calibration_error
    from neurolens.calibration.temperature import PlattScaler, TemperatureScaler

    rng = np.random.default_rng(3)
    conf = rng.uniform(0.5, 0.99, size=3000)            # detector never says < 0.5
    y = (rng.random(3000) < 0.05 + 0.15 * (conf - 0.5) / 0.49).astype(float)  # ~12% true
    t = TemperatureScaler().fit(conf, y).transform(conf)
    pl = PlattScaler().fit(conf, y)
    p = pl.transform(conf)
    assert abs(p.mean() - y.mean()) < 0.02 and t.min() >= 0.5   # temperature cannot go below 0.5
    assert expected_calibration_error(p, y) < 0.05 < expected_calibration_error(t, y)
    cal = ConfidenceCalibrator(platt={"ictal_ml": [pl.a, pl.b]})
    path = cal.save(tmp_path / "c.json")
    cal2 = ConfidenceCalibrator.load(path)
    assert abs(cal2.calibrate("ictal_ml", 0.8) - float(pl.transform([0.8])[0])) < 1e-9
    assert cal2.calibrate("ictal_rhythm", 0.8) == pytest.approx(0.8)   # other codes untouched
    assert not cal2.is_identity


def test_monitor_learned_alarm_not_regated_on_confidence(demo_edf, configs, monkeypatch):
    """Regression: a run above threshold (raw p 0.35 at threshold 0.3) must alarm even though
    the generic seizure rule gates at min_confidence 0.6 for the threshold detector."""
    import copy

    import neurolens.layer4_detect.ml_ictal as mli
    from neurolens.contracts.alarms import AlarmType
    from neurolens.realtime.monitor import RealtimeMonitor
    from neurolens.realtime.stream import EdfReplaySource

    class Const:
        meta = {}

        def predict_proba(self, X):
            return np.full(len(X), 0.35)

    monkeypatch.setattr(mli, "load_model", lambda path: Const())
    cfg = copy.deepcopy(configs)
    cfg.ml.realtime.threshold = 0.3  # pin: the scenario needs raw p just above threshold, below 0.6
    mon = RealtimeMonitor(cfg, learned=True)
    assert mon._rt.rule_for("seizure").min_confidence == 0.0
    summary = mon.run(EdfReplaySource(demo_edf, chunk_s=cfg.realtime.step_s))
    seiz = [a for a in summary.alarms if a.type == AlarmType.SEIZURE]
    assert seiz, "learned-detector run must raise a seizure alarm"
    if cfg.ml.realtime.calibration:
        assert seiz[0].confidence < 0.2   # shown as calibrated probability


def test_prereg10_report_and_apply_decision(tmp_path):
    import shutil

    import yaml

    from neurolens.evaluation.prereg10 import apply_decision, assess, markdown_report

    def rec(file, n, tp, fp, hours=5.0):
        return {"file": file, "hours": hours, "n_seizures": n, "tp": tp, "fp": fp, "latencies_s": [5.0] * tp}

    def mode(name, tp, fp):
        return {"mode": name, "records": [rec("chb04/a.edf", 10, tp, fp), rec("PN07/b.edf", 2, 2, fp)]}

    metrics = {"results": [mode("offline-threshold-default", 9, 40), mode("offline-ml-A", 6, 30),
                           mode("offline-ml-B", 5, 3), mode("realtime-threshold-default", 8, 20),
                           mode("realtime-ml-A", 8, 15), mode("realtime-ml-B", 7, 2)]}
    a = assess(metrics)
    md = markdown_report(a)
    assert "| real-time | обученный, точка A | 10/12 |" in md and "PN07" in md and "Гипотезы" in md
    cfg = tmp_path / "ml.yaml"
    shutil.copy("configs/ml.yaml", cfg)
    flags = apply_decision(a, str(cfg))
    assert flags == {"realtime": True, "offline": False}   # H1 true (8+2 vs 8+2, fewer FA); H2 false
    doc = yaml.safe_load(cfg.read_text())
    assert doc["realtime"]["enabled"] is True and doc["offline"]["enabled"] is False
    assert "# Learned ictal detector" in cfg.read_text()  # comments preserved


def test_critical_findings_merged_per_code():
    from neurolens.contracts.events import DetectionResult, Event, Localization
    from neurolens.critical.safety import scan_critical_findings

    def ev(code, a, b):
        return Event(code=code, label_ru=code, label_uz=code, group="ictal", localization=Localization(),
                     t_start=a, t_end=b, confidence=0.7)

    det = DetectionResult(events=[ev("ictal_ml", 10, 50), ev("ictal_ml", 100, 190), ev("ictal_rhythm", 300, 310)])
    found = scan_critical_findings(det)
    codes = [f.code for f in found]
    assert codes.count("status_epilepticus_suspected") == 1 and "ongoing_seizure" not in codes
    se = next(f for f in found if f.code == "status_epilepticus_suspected")
    assert "90" in se.text.ru  # longest event's duration
    assert set(se.grounding_refs) == {"ictal_ml"}


def test_feature_version4_appends_age():
    from neurolens.layer4_detect.ml_ictal import FEATURE_NAMES_V4, feature_names, names_for_width

    f = _fake_features(n=200, seizure=(100, 140))
    X4 = featurize(f, version=4, age_years=3.0)
    assert X4.shape == (200, len(FEATURE_NAMES_V4)) and feature_names(4)[-1] == "age_years"
    np.testing.assert_allclose(X4[:, :-1], featurize(f))
    assert np.all(X4[:, -1] == 3.0)
    assert np.isnan(featurize(f, version=4)[:, -1]).all()  # unknown age -> missing
    assert names_for_width(len(FEATURE_NAMES_V4)) == FEATURE_NAMES_V4


def test_subject_ages_parsing(tmp_path):
    from neurolens.datasets import PhysioNetClient

    texts = {"SUBJECT-INFO": b"Case\tGender\tAge (years)\n\nchb01\tF\t11\nchb06\tF\t 1.5\n",
             "subject_info.csv": b"patient_id, age_years, gender\nPN00,55,Male\nPN10,25,Male\n"}
    fetch = lambda url: texts[url.rsplit("/", 1)[1]]
    assert PhysioNetClient("chbmit", tmp_path, fetch_text=fetch).subject_ages() == {"chb01": 11.0, "chb06": 1.5}
    assert PhysioNetClient("siena", tmp_path, fetch_text=fetch).subject_ages() == {"PN00": 55.0, "PN10": 25.0}


def test_prereg11_assessment_decisions_and_apply(tmp_path):
    import shutil

    import yaml

    from neurolens.evaluation.prereg11 import apply_decision, assess, markdown_report

    def rec(file, n, tp, fp):
        return {"file": file, "hours": 10.0, "n_seizures": n, "tp": tp, "fp": fp, "latencies_s": [5.0] * tp}

    def mode(name, tp, fp):
        return {"mode": name, "records": [rec("chb06/a.edf", 10, tp, fp), rec("PN10/b.edf", 10, 10, fp)]}

    def metrics(rt_ml_fp, off_ml_tp):
        return {"results": [
            mode("offline-threshold-default", 9, 50), mode("offline-cmp-A", 9, 45), mode("offline-ml-A", off_ml_tp, 40),
            mode("realtime-threshold-default", 8, 20), mode("realtime-cmp-A", 9, 25), mode("realtime-ml-A", 9, rt_ml_fp)]}

    a = assess(metrics(rt_ml_fp=18, off_ml_tp=9))
    assert a["decision"] == {"realtime_default": "learned_v2", "offline_default": "learned_v2"}
    assert "обученный v2 (31 пациент), A" in markdown_report(a)
    b = assess(metrics(rt_ml_fp=22, off_ml_tp=5))   # v2 offline much worse -> fall back to v1
    assert b["decision"] == {"realtime_default": "threshold", "offline_default": "learned_v1"}
    fz11 = {m: {"A_replacement": {"params": {"threshold": 0.35, "min_epochs": 12}},
                "model": {"path": f"configs/models/ictal_gbm_{m}_v2.json"},
                "calibration": {"path": f"configs/calibration.ml_{m}_v2.json"}} for m in ("offline", "realtime")}
    fz10 = {m: {"A_replacement": {"params": {"threshold": 0.4, "min_epochs": 5}},
                "model": {"path": f"configs/models/ictal_gbm_{m}_v1.json"}} for m in ("offline", "realtime")}
    cfg = tmp_path / "ml.yaml"
    shutil.copy("configs/ml.yaml", cfg)
    apply_decision(a, fz11, fz10, str(cfg))
    doc = yaml.safe_load(cfg.read_text())
    assert doc["realtime"]["enabled"] is True and doc["realtime"]["model"].endswith("realtime_v2.json")
    assert doc["offline"]["threshold"] == 0.35 and doc["offline"]["min_epochs"] == 12
    assert doc["offline"]["calibration"] == "configs/calibration.ml_offline_v2.json"  # follows the model
    assert "# Platt" in cfg.read_text()  # comments survive
    apply_decision(b, fz11, fz10, str(cfg))
    doc = yaml.safe_load(cfg.read_text())
    assert doc["realtime"]["enabled"] is False and doc["offline"]["model"].endswith("offline_v1.json")
    assert doc["offline"]["calibration"] == "configs/calibration.ml_offline.json"  # v1 keeps its own Platt file
    assert doc["realtime"]["alarm_persistence_windows"] == 1  # untouched keys survive


def test_ml_eval_stream_cache_resumes(demo_edf, configs, tmp_path, monkeypatch):
    """An interrupted held-out run resumes from per-record replay results and
    gives the same metrics as an uninterrupted one."""
    from neurolens.datasets.annotations import RecordAnnotation, SeizureInterval
    from neurolens.evaluation import ml_eval

    frozen = json.loads(open("docs/preregistration_increment10_frozen.json", encoding="utf-8").read())
    ann = RecordAnnotation(database="chbmit", subject="demo", file="demo/demo.edf",
                           seizures=[SeizureInterval(60.0, 90.0)])
    items = [(ann, demo_edf)]
    cache = tmp_path / "streams"
    first = ml_eval.evaluate_ml_test(items, configs, frozen, feature_cache=tmp_path / "feat",
                                     workers=1, stream_cache=cache)
    assert len(list(cache.glob("stream_demo_*.pkl"))) == 1

    def no_replay(fn, tasks, *a, **k):
        assert tasks == [], "cached records must not be replayed again"

    monkeypatch.setattr(ml_eval, "run_memory_bounded", no_replay)
    second = ml_eval.evaluate_ml_test(items, configs, frozen, feature_cache=tmp_path / "feat",
                                      workers=1, stream_cache=cache)
    assert [r.mode for r in first] == [r.mode for r in second]
    for a, b in zip(first, second):
        assert a.total.as_dict() == b.total.as_dict()
