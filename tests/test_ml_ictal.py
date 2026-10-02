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
