"""Learned ictal detector: causal epoch features + logistic regression (TZ §7.3, §13).

Increment 9 showed that hand-set global thresholds (rhythmicity, amplitude
factor, channel count, duration) do not transfer between patients. This module
learns how to *combine* the same physiological evidence from many patients:

* **Features** — per 2 s epoch (1 s step), computed *causally* (only the past),
  so the identical code serves offline analysis and the streaming monitor.
  Amplitude is normalised to each channel's own slow baseline (log-RMS minus an
  exponential moving average, τ ≈ 5 min), which removes the between-patient
  amplitude differences that broke the fixed ``amplitude_factor``. Channel
  values are aggregated (max / top-3 mean / mean / fraction of channels), so the
  model is montage-independent; short temporal context captures persistence and
  frequency evolution.
* **Model** — L2-regularised logistic regression (numpy/scipy, no extra
  dependency), class-balanced; stored as plain JSON (feature names, scaling,
  weights) so it is inspectable and versionable — no pickled code.
* **Events** — causal moving average of the probability; an event is a run of
  ≥ ``min_epochs`` epochs with smoothed probability ≥ ``threshold``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

from ..contracts.events import Event, EventEvidence, Localization
from .base import Detector

EMA_TAU_EPOCHS = 300.0       # baseline time constant (epochs at 1 s step ≈ 5 min)
INIT_EPOCHS = 30             # baseline initialised from the median of the first 30 epochs
LOG2 = float(np.log(2.0))
LOG15 = float(np.log(1.5))

BASE_FEATURES = [
    "z_max", "z_top3", "z_mean", "z_frac_gt2x",
    "conc_max", "conc_top3", "conc_mean",
    "coinvolved_frac",
    "domf_median", "domf_iqr",
    "rp_delta", "rp_theta", "rp_alpha", "rp_beta", "rp_gamma",
]
CONTEXT_FEATURES = [
    "z_top3_m5", "z_top3_m15", "conc_top3_m5", "conc_top3_m15",
    "coinv_m5", "coinv_m15", "domf_evolution10", "z_top3_trend",
]
FEATURE_NAMES = BASE_FEATURES + CONTEXT_FEATURES


def _top3_mean(a: np.ndarray) -> np.ndarray:
    k = min(3, a.shape[1])
    return np.mean(np.sort(a, axis=1)[:, -k:], axis=1)


def _causal_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Mean over the current and previous n-1 samples (shorter at the start)."""
    c = np.cumsum(np.concatenate([[0.0], x]))
    idx = np.arange(1, x.size + 1)
    lo = np.maximum(0, idx - n)
    return (c[idx] - c[lo]) / (idx - lo)


def _lag(x: np.ndarray, k: int) -> np.ndarray:
    out = np.empty_like(x)
    out[:k] = x[0]
    out[k:] = x[:-k] if k else x
    return out


def causal_baseline(log_rms: np.ndarray, init: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-channel EMA of log-RMS, value *before* each epoch's update."""
    a = 1.0 / EMA_TAU_EPOCHS
    if init is None:
        init = np.median(log_rms[: min(INIT_EPOCHS, len(log_rms))], axis=0)
    # y[t] = (1-a) y[t-1] + a x[t]; baseline for epoch t is y[t-1]
    y, _ = lfilter([a], [1.0, -(1.0 - a)], log_rms, axis=0, zi=(1.0 - a) * init[None, :])
    base = np.empty_like(log_rms)
    base[0] = init
    base[1:] = y[:-1]
    return base


def causal_baseline_gated(log_rms: np.ndarray, init: Optional[np.ndarray] = None,
                          gate: float = LOG15) -> np.ndarray:
    """EMA baseline that does **not** learn from elevated epochs: channel c is
    updated at epoch t only if log_rms[t,c] - baseline[t,c] < ``gate`` (1.5×).
    Prevents a long seizure from being absorbed into "normal" (experimental, v3)."""
    a = 1.0 / EMA_TAU_EPOCHS
    if init is None:
        init = np.median(log_rms[: min(INIT_EPOCHS, len(log_rms))], axis=0)
    ema = np.array(init, dtype=float)
    base = np.empty_like(log_rms)
    for t in range(log_rms.shape[0]):
        base[t] = ema
        upd = (log_rms[t] - ema) < gate
        ema = np.where(upd, (1.0 - a) * ema + a * log_rms[t], ema)
    return base


def base_features(rms: np.ndarray, conc: np.ndarray, domf: np.ndarray,
                  relpow: dict[str, np.ndarray], baseline: np.ndarray) -> np.ndarray:
    z = np.log(np.maximum(rms, 1e-6)) - baseline
    co = ((z > LOG15) & (conc > 0.45)).mean(axis=1)
    q75, q25 = np.percentile(domf, [75, 25], axis=1)
    cols = [
        z.max(axis=1), _top3_mean(z), z.mean(axis=1), (z > LOG2).mean(axis=1),
        conc.max(axis=1), _top3_mean(conc), conc.mean(axis=1),
        co,
        np.median(domf, axis=1), q75 - q25,
    ] + [relpow[b].mean(axis=1) for b in ("delta", "theta", "alpha", "beta", "gamma")]
    return np.column_stack(cols)


def context_features(B: np.ndarray) -> np.ndarray:
    i = {n: k for k, n in enumerate(BASE_FEATURES)}
    zt, ct, co, df = B[:, i["z_top3"]], B[:, i["conc_top3"]], B[:, i["coinvolved_frac"]], B[:, i["domf_median"]]
    ref = _lag(_causal_mean(zt, 20), 10)  # mean of epochs t-29..t-10
    cols = [
        _causal_mean(zt, 5), _causal_mean(zt, 15), _causal_mean(ct, 5), _causal_mean(ct, 15),
        _causal_mean(co, 5), _causal_mean(co, 15), np.abs(df - _lag(df, 10)), zt - ref,
    ]
    return np.column_stack(cols)


# --- feature set v2: + waveform morphology (line length, Teager energy) ---------
MORPH_BASE = ["ll_z_max", "ll_z_top3", "ll_z_mean", "te_z_max", "te_z_top3", "shape_z_top3"]
MORPH_CONTEXT = ["ll_z_top3_m5", "ll_z_top3_m15"]
FEATURE_NAMES_V2 = FEATURE_NAMES + MORPH_BASE + MORPH_CONTEXT


FEATURE_NAMES_V4 = FEATURE_NAMES + ["age_years"]


def feature_names(version: int = 1) -> list[str]:
    return {2: FEATURE_NAMES_V2, 4: FEATURE_NAMES_V4}.get(version, FEATURE_NAMES)


def names_for_width(n: int) -> list[str]:
    for names in (FEATURE_NAMES, FEATURE_NAMES_V4, FEATURE_NAMES_V2):
        if len(names) == n:
            return names
    return [f"f{i}" for i in range(n)]


def _morph_logs(rms: np.ndarray, linelen: np.ndarray, teager: np.ndarray) -> list[np.ndarray]:
    ll = np.log(np.maximum(linelen, 1e-6))
    te = np.log(np.maximum(teager, 1e-6))
    shape = ll - np.log(np.maximum(rms, 1e-6))  # sharpness: line length per unit amplitude
    return [ll, te, shape]


def morph_base(logs: list[np.ndarray], bases: list[np.ndarray]) -> np.ndarray:
    zl, zt, zs = (x - b for x, b in zip(logs, bases))
    return np.column_stack([zl.max(axis=1), _top3_mean(zl), zl.mean(axis=1),
                            zt.max(axis=1), _top3_mean(zt), _top3_mean(zs)])


def morph_context(M: np.ndarray) -> np.ndarray:
    zl = M[:, MORPH_BASE.index("ll_z_top3")]
    return np.column_stack([_causal_mean(zl, 5), _causal_mean(zl, 15)])


def _morph_arrays(features):
    extra = getattr(features, "extra", None) or {}
    if "epoch_linelen" not in extra:
        raise ValueError("feature set v2 needs epoch_linelen/epoch_teager (Layer-3 FEATURE_VERSION >= 2)")
    return np.asarray(extra["epoch_linelen"], float), np.asarray(extra["epoch_teager"], float)


def featurize(features, version: int = 1, age_years: Optional[float] = None) -> np.ndarray:
    """Causal per-epoch feature matrix [n_epochs × len(feature_names(version))].

    version 1: frozen increment-10 features; 2: + morphology (rejected, §11);
    3: v1 with the gated baseline (not adopted, §14); 4: v1 + patient age in
    years (NaN when unknown — the trees route missing values)."""
    if version == 3:
        return _featurize_v1(features, gated=True)
    if version == 4:
        X = _featurize_v1(features)
        age = np.nan if age_years is None else float(age_years)
        return np.column_stack([X, np.full(X.shape[0], age)])
    X = _featurize_v1(features)
    if version == 1 or X.shape[0] == 0:
        return X if version == 1 else np.zeros((0, len(FEATURE_NAMES_V2)))
    ll, te = _morph_arrays(features)
    logs = _morph_logs(np.asarray(features.epoch_rms, float), ll, te)
    M = morph_base(logs, [causal_baseline(x) for x in logs])
    return np.column_stack([X, M, morph_context(M)])


def _featurize_v1(features, gated: bool = False) -> np.ndarray:
    """Causal per-epoch feature matrix [n_epochs × len(FEATURE_NAMES)] for a FeatureSet."""
    rms = np.asarray(features.epoch_rms, dtype=float)
    if rms.size == 0:
        return np.zeros((0, len(FEATURE_NAMES)))
    log_rms = np.log(np.maximum(rms, 1e-6))
    base = causal_baseline_gated(log_rms) if gated else causal_baseline(log_rms)
    B = base_features(rms, np.asarray(features.epoch_band_conc, float),
                      np.asarray(features.epoch_domfreq, float),
                      {k: np.asarray(v, float) for k, v in features.epoch_relpow.items()}, base)
    return np.column_stack([B, context_features(B)])


class StreamingFeaturizer:
    """Incremental twin of :func:`featurize` for the real-time monitor.

    Feed epochs as they arrive (``push``); returns the feature rows for epochs
    that became computable. Output concatenated over a stream is identical to
    ``featurize`` on the whole stream (tested): the baseline is initialised from
    the median of the first ``INIT_EPOCHS`` epochs (a 30 s warm-up, emitted at
    once), then updated as the same EMA; context features are computed on a
    bounded tail of base rows long enough to be exact for new rows.
    """

    _KEEP = 256
    _TRIM_TO = 128

    def __init__(self) -> None:
        self._pending: list[tuple] = []
        self._n_pending = 0
        self._ema: Optional[np.ndarray] = None
        self._B: Optional[np.ndarray] = None

    def push(self, rms, conc, domf, relpow: dict) -> np.ndarray:
        rms = np.atleast_2d(np.asarray(rms, float))
        if rms.shape[0] == 0:
            return np.zeros((0, len(FEATURE_NAMES)))
        chunk = (rms, np.atleast_2d(np.asarray(conc, float)), np.atleast_2d(np.asarray(domf, float)),
                 {k: np.atleast_2d(np.asarray(v, float)) for k, v in relpow.items()})
        if self._ema is None:
            self._pending.append(chunk)
            self._n_pending += rms.shape[0]
            if self._n_pending < INIT_EPOCHS:
                return np.zeros((0, len(FEATURE_NAMES)))
            rms = np.vstack([c[0] for c in self._pending])
            chunk = (rms, np.vstack([c[1] for c in self._pending]), np.vstack([c[2] for c in self._pending]),
                     {k: np.vstack([c[3][k] for c in self._pending]) for k in self._pending[0][3]})
            self._pending = []
            log_rms = np.log(np.maximum(rms, 1e-6))
            self._ema = np.median(log_rms[:INIT_EPOCHS], axis=0)
        rms, conc, domf, rp = chunk
        log_rms = np.log(np.maximum(rms, 1e-6))
        base = causal_baseline(log_rms, init=self._ema)
        a = 1.0 / EMA_TAU_EPOCHS
        self._ema = (1.0 - a) * base[-1] + a * log_rms[-1]
        B_new = base_features(rms, conc, domf, rp, base)
        self._B = B_new if self._B is None else np.vstack([self._B, B_new])
        X = np.column_stack([self._B, context_features(self._B)])[-B_new.shape[0]:]
        if self._B.shape[0] > self._KEEP:
            self._B = self._B[-self._TRIM_TO:]
        return X


class StreamingFeaturizerV2:
    """Streaming twin of ``featurize(version=2)``: the unchanged v1 streaming
    featurizer plus morphology columns with the same warm-up and EMA logic."""

    _KEEP = 256
    _TRIM_TO = 128

    def __init__(self) -> None:
        self._v1 = StreamingFeaturizer()
        self._pending: list[tuple] = []
        self._n_pending = 0
        self._ema: Optional[list[np.ndarray]] = None
        self._M: Optional[np.ndarray] = None

    def push(self, rms, conc, domf, relpow: dict, linelen, teager) -> np.ndarray:
        X1 = self._v1.push(rms, conc, domf, relpow)
        chunk = tuple(np.atleast_2d(np.asarray(a, float)) for a in (rms, linelen, teager))
        if chunk[0].shape[0] == 0:
            return np.zeros((0, len(FEATURE_NAMES_V2)))
        if self._ema is None:
            self._pending.append(chunk)
            self._n_pending += chunk[0].shape[0]
            if self._n_pending < INIT_EPOCHS:
                return np.zeros((0, len(FEATURE_NAMES_V2)))
            chunk = tuple(np.vstack([c[i] for c in self._pending]) for i in range(3))
            self._pending = []
            self._ema = [np.median(x[:INIT_EPOCHS], axis=0) for x in _morph_logs(*chunk)]
        logs = _morph_logs(*chunk)
        bases = [causal_baseline(x, init=e) for x, e in zip(logs, self._ema)]
        a = 1.0 / EMA_TAU_EPOCHS
        self._ema = [(1.0 - a) * b[-1] + a * x[-1] for b, x in zip(bases, logs)]
        M_new = morph_base(logs, bases)
        self._M = M_new if self._M is None else np.vstack([self._M, M_new])
        Mx = np.column_stack([self._M, morph_context(self._M)])[-M_new.shape[0]:]
        if self._M.shape[0] > self._KEEP:
            self._M = self._M[-self._TRIM_TO:]
        if X1.shape[0] != Mx.shape[0]:
            raise RuntimeError("v1/v2 streaming featurizers out of step")
        return np.column_stack([X1, Mx])


# ---------------------------------------------------------------------------
# logistic regression
# ---------------------------------------------------------------------------

@dataclass
class LogisticModel:
    feature_names: list[str]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    l2: float = 1.0
    meta: dict = field(default_factory=dict)

    def decision(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mean) / self.scale
        return Z @ self.weights + self.bias

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self.decision(X), -40, 40)))

    def to_json(self) -> dict:
        return {
            "type": "logistic_regression", "feature_names": self.feature_names,
            "mean": self.mean.round(8).tolist(), "scale": self.scale.round(8).tolist(),
            "weights": self.weights.round(8).tolist(), "bias": round(float(self.bias), 8),
            "l2": self.l2, "meta": self.meta,
        }

    @classmethod
    def from_json(cls, d: dict) -> "LogisticModel":
        if d.get("type") != "logistic_regression":
            raise ValueError("not a logistic_regression model")
        return cls(list(d["feature_names"]), np.asarray(d["mean"]), np.asarray(d["scale"]),
                   np.asarray(d["weights"]), float(d["bias"]), float(d.get("l2", 1.0)),
                   dict(d.get("meta", {})))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "LogisticModel":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0,
                 sample_weight: Optional[np.ndarray] = None) -> LogisticModel:
    """Class-balanced L2 logistic regression on standardised features (L-BFGS)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-9] = 1.0
    Z = (X - mean) / scale
    n_pos, n_neg = max(1.0, y.sum()), max(1.0, (1 - y).sum())
    w_cls = np.where(y > 0.5, 0.5 * len(y) / n_pos, 0.5 * len(y) / n_neg)
    sw = w_cls * (sample_weight if sample_weight is not None else 1.0)
    sw = sw / sw.mean()
    n, d = Z.shape

    def loss_grad(theta):
        w, b = theta[:d], theta[d]
        m = Z @ w + b
        # stable log(1+exp(-y'm)) with y' in {-1, 1}
        ys = 2 * y - 1
        t = -ys * m
        lse = np.logaddexp(0.0, t)
        loss = np.sum(sw * lse) / n + 0.5 * l2 * np.dot(w, w) / n
        g_m = sw * (-ys) * (1.0 / (1.0 + np.exp(-np.clip(t, -40, 40)))) / n
        return loss, np.concatenate([Z.T @ g_m + l2 * w / n, [g_m.sum()]])

    res = minimize(loss_grad, np.zeros(d + 1), jac=True, method="L-BFGS-B",
                   options={"maxiter": 500})
    return LogisticModel(list(FEATURE_NAMES[:d]) if d == len(FEATURE_NAMES) else [f"f{i}" for i in range(d)],
                         mean, scale, res.x[:d], float(res.x[d]), l2,
                         {"converged": bool(res.success), "n_train": int(n), "n_pos": int(y.sum())})


# ---------------------------------------------------------------------------
# gradient-boosted trees (trained with scikit-learn, evaluated in numpy)
# ---------------------------------------------------------------------------

@dataclass
class TreeEnsembleModel:
    """Additive tree ensemble (binary log-loss) stored as plain arrays.

    Training uses scikit-learn's HistGradientBoostingClassifier (optional ``ml``
    extra); :func:`export_hist_gbm` copies its trees into this JSON-serialisable
    form, and prediction here is pure numpy — identical to scikit-learn's
    ``predict_proba`` (tested) without a runtime dependency or pickled code.
    """

    feature_names: list[str]
    baseline: float
    trees: list[dict]  # each: feature, threshold, left, right, value, is_leaf, missing_left
    meta: dict = field(default_factory=dict)

    def decision(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        raw = np.full(X.shape[0], self.baseline)
        rows = np.arange(X.shape[0])
        for t in self.trees:
            feat, thr = t["_feature"], t["_threshold"]
            left, right, leaf = t["_left"], t["_right"], t["_is_leaf"]
            miss_left = t["_missing_left"]
            node = np.zeros(X.shape[0], dtype=np.int64)
            active = ~leaf[node]
            while active.any():
                n = node[active]
                x = X[rows[active], feat[n]]
                go_left = np.where(np.isnan(x), miss_left[n], x <= thr[n])
                node[active] = np.where(go_left, left[n], right[n])
                active = ~leaf[node]
            raw += t["_value"][node]
        return raw

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self.decision(X), -40, 40)))

    def __post_init__(self) -> None:
        for t in self.trees:  # cache numpy views for fast traversal
            t["_feature"] = np.asarray(t["feature"], dtype=np.int64)
            t["_threshold"] = np.asarray(t["threshold"], dtype=float)
            t["_left"] = np.asarray(t["left"], dtype=np.int64)
            t["_right"] = np.asarray(t["right"], dtype=np.int64)
            t["_value"] = np.asarray(t["value"], dtype=float)
            t["_is_leaf"] = np.asarray(t["is_leaf"], dtype=bool)
            t["_missing_left"] = np.asarray(t["missing_left"], dtype=bool)

    def to_json(self) -> dict:
        trees = [{k: v for k, v in t.items() if not k.startswith("_")} for t in self.trees]
        return {"type": "tree_ensemble", "feature_names": self.feature_names,
                "baseline": self.baseline, "trees": trees, "meta": self.meta}

    @classmethod
    def from_json(cls, d: dict) -> "TreeEnsembleModel":
        if d.get("type") != "tree_ensemble":
            raise ValueError("not a tree_ensemble model")
        return cls(list(d["feature_names"]), float(d["baseline"]),
                   [dict(t) for t in d["trees"]], dict(d.get("meta", {})))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), separators=(",", ":")) + "\n", encoding="utf-8")
        return path


def export_hist_gbm(clf, feature_names: list[str], meta: dict | None = None) -> TreeEnsembleModel:
    """Copy a fitted binary HistGradientBoostingClassifier into a TreeEnsembleModel."""
    trees = []
    for (pred,) in clf._predictors:
        nd = pred.nodes
        if np.any(nd["is_categorical"]):
            raise ValueError("categorical splits are not supported")
        trees.append({
            "feature": nd["feature_idx"].astype(int).tolist(),
            "threshold": [float(v) for v in nd["num_threshold"]],
            "left": nd["left"].astype(int).tolist(),
            "right": nd["right"].astype(int).tolist(),
            "value": [float(v) for v in nd["value"]],
            "is_leaf": nd["is_leaf"].astype(bool).tolist(),
            "missing_left": nd["missing_go_to_left"].astype(bool).tolist(),
        })
    base = float(np.ravel(clf._baseline_prediction)[0])
    return TreeEnsembleModel(list(feature_names), base, trees, dict(meta or {}))


def load_model(path: str | Path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("type") == "tree_ensemble":
        return TreeEnsembleModel.from_json(d)
    return LogisticModel.from_json(d)


# ---------------------------------------------------------------------------
# probabilities -> events
# ---------------------------------------------------------------------------

SMOOTH_EPOCHS = 5


def probability_runs(p: np.ndarray, threshold: float, min_epochs: int,
                     smooth: int = SMOOTH_EPOCHS) -> list[tuple[int, int, float]]:
    """Runs [a, b) of epochs whose causally smoothed probability ≥ threshold,
    at least ``min_epochs`` long; returns (a, b, mean_probability)."""
    if p.size == 0:
        return []
    ps = _causal_mean(p, smooth)
    on = ps >= threshold
    d = np.diff(on.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    stops = list(np.where(d == -1)[0] + 1)
    if on[0]:
        starts = [0] + starts
    if on[-1]:
        stops = stops + [on.size]
    return [(int(a), int(b), float(p[a:b].mean())) for a, b in zip(starts, stops) if b - a >= min_epochs]


@dataclass
class MLDetectorConfig:
    threshold: float = 0.9
    min_epochs: int = 5


class MLIctalDetector(Detector):
    """Layer-4 detector wrapping a trained :class:`LogisticModel`."""

    code = "ictal_ml"
    group = "ictal"

    def __init__(self, model, cfg: MLDetectorConfig | None = None):
        self.model = model
        self.cfg = cfg or MLDetectorConfig()

    def detect(self, sig, features, thresholds, artifacts=None) -> list[Event]:
        X = featurize(features)
        if X.shape[0] == 0:
            return []
        p = self.model.predict_proba(X)
        times = np.asarray(features.epoch_times, float)
        z_idx = FEATURE_NAMES.index("z_top3")
        out = []
        for a, b, pm in probability_runs(p, self.cfg.threshold, self.cfg.min_epochs):
            rms = np.asarray(features.epoch_rms)[a:b]
            base = np.median(np.asarray(features.epoch_rms), axis=0) + 1e-9
            involved = [features.eeg_channels[i] for i in np.where((rms / base).mean(axis=0) > 1.5)[0]]
            ru, uz = self.labels(self.code)
            out.append(Event(
                code=self.code, label_ru=ru, label_uz=uz, group=self.group,
                localization=Localization(channels=involved),
                t_start=float(times[a]), t_end=float(times[b - 1]),
                confidence=float(np.clip(pm, 0.5, 0.99)),
                evidence=[
                    EventEvidence(feature="ml_probability", value=round(pm, 3),
                                  reference=f">= {self.cfg.threshold} for >= {self.cfg.min_epochs} epochs"),
                    EventEvidence(feature="amplitude_vs_own_baseline_log", value=round(float(X[a:b, z_idx].mean()), 3),
                                  note="log RMS above the channel's 5-min baseline (top-3 channels)"),
                ],
                metadata={"model": self.model.meta.get("name", "ictal_lr"), "n_epochs": b - a},
            ))
        return out
