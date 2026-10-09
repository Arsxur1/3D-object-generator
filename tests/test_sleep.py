"""Sleep staging: Sleep-EDF parsing/split, features, multiclass JSON model,
metrics, signal-level staging, pipeline integration, pre-registered decision."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from neurolens.datasets.sleepedf import (
    analysis_window,
    epochs_from_annotations,
    pair_records,
    split_by_rule,
    stage_label,
)
from neurolens.layer4_detect.sleep_staging import (
    STAGES,
    confusion,
    derivations,
    epoch_features,
    export_multiclass_gbm,
    hypnogram_summary,
    rule_based_stages,
    scores_from_confusion,
    stage_signal,
    standardise_and_context,
)


def test_pairing_split_and_stage_mapping():
    paths = ["sleep-cassette/SC4001E0-PSG.edf", "sleep-cassette/SC4001EC-Hypnogram.edf",
             "sleep-cassette/SC4012E0-PSG.edf", "sleep-cassette/SC4012EC-Hypnogram.edf",
             "sleep-cassette/SC4021E0-PSG.edf"]  # no hypnogram -> dropped
    recs = pair_records(paths)
    assert [r.key for r in recs] == ["SC4001", "SC4012"]
    s = split_by_rule(recs)
    assert [r.key for r in s["dev"]] == ["SC4001"] and [r.key for r in s["test"]] == ["SC4012"]
    assert stage_label("Sleep stage 4") == STAGES.index("N3") == stage_label("Sleep stage 3")
    assert stage_label("Sleep stage R") == STAGES.index("REM")
    assert stage_label("Sleep stage ?") is None and stage_label("Movement time") is None


def test_epoch_expansion_and_window():
    st = epochs_from_annotations([0, 3600, 3660, 3690], [3600, 60, 30, 30],
                                 ["Sleep stage W", "Sleep stage 1", "Sleep stage 2", "Sleep stage W"], 200)
    assert st[0] == 0 and st[120] == 1 and st[122] == 2 and st[123] == 0 and st[150] is None
    a, b = analysis_window(st)
    assert (a, b) == (120 - 60, 123 + 60)


def _night(n_ep=200, fs=100.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(n_ep * 30 * fs)) / fs
    y = np.repeat(rng.integers(0, 5, n_ep // 10), 10)
    amp = np.repeat(np.array([5, 10, 15, 60, 8])[y], int(30 * fs))
    freq = np.repeat(np.array([10.0, 6.0, 13.0, 1.0, 5.0])[y], int(30 * fs))
    x = amp * np.sin(2 * np.pi * freq * t) + rng.normal(0, 3, t.size)
    return x, x * 0.8 + rng.normal(0, 3, t.size), rng.normal(0, 5, t.size), y, fs


def test_features_model_roundtrip_and_metrics(tmp_path):
    from sklearn.ensemble import HistGradientBoostingClassifier

    from neurolens.layer4_detect.sleep_staging import MulticlassTreeModel

    fx, px, eog, y, fs = _night()
    F, names = epoch_features(fx, px, eog, fs, 0, len(y))
    assert F.shape == (len(y), len(names)) and np.isfinite(F).all()
    X, xn = standardise_and_context(F, names)
    assert X.shape[1] == 5 * len(names) and "fc_rel_alpha@+0" in xn
    clf = HistGradientBoostingClassifier(max_iter=20, random_state=0).fit(X, y)
    m = export_multiclass_gbm(clf, xn, list(STAGES))
    m2 = MulticlassTreeModel.load(m.save(tmp_path / "m.json"))
    assert np.allclose(m2.predict_proba(X), clf.predict_proba(X), atol=1e-9)
    sc = scores_from_confusion(confusion(y, m2.predict_proba(X).argmax(1)))
    assert sc["kappa"] > 0.9 and set(sc["f1"]) == set(STAGES)
    assert rule_based_stages(F, names).shape == y.shape


def test_hypnogram_summary():
    st = np.array([0] * 20 + [1] * 2 + [2] * 30 + [3] * 20 + [4] * 10 + [0] * 8)
    s = hypnogram_summary(st)
    assert s["sleep_latency_min"] == 10.0 and s["rem_latency_min"] == 26.0
    assert s["total_sleep_min"] == 31.0 and s["stage_percent_of_sleep"]["N3"] == pytest.approx(32.3, abs=0.1)


def test_derivations_and_stage_signal():
    from sklearn.ensemble import HistGradientBoostingClassifier

    fx, px, eog, y, fs = _night(n_ep=60)
    names = ["Fp1", "Fp2", "Cz", "Pz", "O1", "O2", "F7", "F8"]
    zeros = np.zeros_like(fx)
    data = np.vstack([fx, fx, zeros, px, zeros, zeros, eog, zeros])
    d = derivations(names, data)
    assert np.allclose(d["frontal"], fx) and np.allclose(d["parietal"], px) and np.allclose(d["eog"], eog)
    assert derivations(["Fp1", "Cz"], data[:2]) is None
    F, fn = epoch_features(fx, px, eog, fs, 0, len(y))
    X, xn = standardise_and_context(F, fn)
    m = export_multiclass_gbm(HistGradientBoostingClassifier(max_iter=10, random_state=0).fit(X, y), xn, list(STAGES))
    out = stage_signal(names, data.astype(np.float32), 200.0 if False else fs, m)
    assert len(out["hypnogram"]) == len(y) and out["eog_source"] == "F7-F8"
    # 256 Hz input is resampled to the model's 100 Hz
    from scipy.signal import resample_poly

    up = resample_poly(data, 64, 25, axis=1)
    assert len(stage_signal(names, up, 256.0, m)["hypnogram"]) == len(y)


def test_pipeline_reports_sleep_staging_when_enabled(demo_edf, configs, tmp_path, monkeypatch):
    from neurolens.pipeline.pipeline import Pipeline

    cfg = copy.deepcopy(configs)
    assert "sleep_staging" not in Pipeline(config=cfg, provider_pref="deterministic",
                                           run_ica=False).analyze_file(demo_edf).result_json
    import neurolens.layer4_detect.sleep_staging as ss

    calls = {}

    class Fake:
        def predict_proba(self, X):
            calls["n"] = len(X)
            p = np.zeros((len(X), 5))
            p[:, 2] = 1.0
            return p

    monkeypatch.setattr(ss.MulticlassTreeModel, "load", classmethod(lambda cls, path: Fake()))
    cfg.sleep = cfg.sleep.model_copy(update={"enabled": True, "model": "x.json", "min_duration_h": 0.0})
    out = Pipeline(config=cfg, provider_pref="deterministic", run_ica=False).analyze_file(demo_edf)
    sl = out.result_json.get("sleep_staging")
    if sl is None:
        pytest.skip("demo EDF lacks the electrodes for the sleep derivations")
    assert set(sl["hypnogram"]) == {"N2"} and calls["n"] == len(sl["hypnogram"])


def test_prereg13_decision_and_apply(tmp_path):
    import shutil

    import yaml

    from neurolens.evaluation.prereg13 import apply_decision, assess, markdown_report

    def sc(k, f1=0.7):
        return {"accuracy": 0.8, "kappa": k, "macro_f1": f1, "f1": {s: f1 for s in STAGES},
                "per_night_kappa_median": k, "n_epochs": 1000, "n_nights": 10, "n_subjects": 5}

    frozen = {"dev_cv": {"kappa": 0.74}, "model": {"path": "configs/models/sleep_stager_v1.json"}}
    a = assess({"learned": sc(0.73), "rule_based": sc(0.40)}, frozen)
    assert all(a["hypotheses"].values()) and a["decision"]["sleep_staging_enabled_by_default"]
    b = assess({"learned": sc(0.66), "rule_based": sc(0.40)}, frozen)
    assert not b["decision"]["sleep_staging_enabled_by_default"]
    assert not b["hypotheses"]["H3_transfer_within_0.05_of_dev_cv"]
    assert "обученный" in markdown_report(a)
    cfg = tmp_path / "sleep.yaml"
    shutil.copy("configs/sleep.yaml", cfg)
    apply_decision(a, frozen, str(cfg))
    d = yaml.safe_load(cfg.read_text())
    assert d["enabled"] is True and d["model"] == "configs/models/sleep_stager_v1.json"
    apply_decision(b, frozen, str(cfg))
    assert yaml.safe_load(cfg.read_text())["enabled"] is False
