"""Sleep staging (AASM: W, N1, N2, N3, REM) from two EEG derivations + EOG.

Per 30-s epoch, spectral and Hjorth features of a frontal-central (Fpz-Cz) and a
parietal-occipital (Pz-Oz) derivation and the horizontal EOG are computed,
standardised per night (offline staging: the whole night is available), and
stacked with the two neighbouring epochs on each side (sleep scorers use
context). A multiclass gradient-boosted tree model (trained with scikit-learn,
stored as JSON, predicted with numpy - same approach as the ictal detector)
gives stage probabilities; a rule-based AASM-style stager without training is
kept as a reference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import welch

STAGES = ("W", "N1", "N2", "N3", "REM")
EPOCH_S = 30.0
BANDS = ((0.5, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, 30.0))
BAND_NAMES = ("sdelta", "delta", "theta", "alpha", "sigma", "beta")
CONTEXT = 2  # epochs on each side


def _eeg_features(x: np.ndarray, fs: float, prefix: str) -> tuple[np.ndarray, list[str]]:
    """x: (n_epochs, n_samples) -> features per epoch."""
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    sel = (f >= 0.5) & (f <= 30.0)
    total = p[:, sel].sum(axis=1) + 1e-12
    feats, names = [np.log(total)], [f"{prefix}_logpow"]
    rel = []
    for (lo, hi), nm in zip(BANDS, BAND_NAMES):
        b = p[:, (f >= lo) & (f < hi)].sum(axis=1)
        rel.append(b / total)
        feats += [np.log(b + 1e-12), b / total]
        names += [f"{prefix}_log_{nm}", f"{prefix}_rel_{nm}"]
    rel = np.array(rel)
    pn = p[:, sel] / total[:, None]
    feats.append(-(pn * np.log(pn + 1e-12)).sum(axis=1))
    names.append(f"{prefix}_entropy")
    cum = np.cumsum(p[:, sel], axis=1) / total[:, None]
    feats.append(f[sel][np.argmax(cum >= 0.95, axis=1)])
    names.append(f"{prefix}_sef95")
    d1 = np.diff(x, axis=1)
    d2 = np.diff(d1, axis=1)
    v0, v1, v2 = x.var(axis=1) + 1e-12, d1.var(axis=1) + 1e-12, d2.var(axis=1) + 1e-12
    mob = np.sqrt(v1 / v0)
    feats += [mob, np.sqrt(v2 / v1) / mob]
    names += [f"{prefix}_hjorth_mob", f"{prefix}_hjorth_comp"]
    feats += [np.log((rel[0] + rel[1]) / (rel[5] + 1e-6) + 1e-6), np.log(rel[2] / (rel[3] + 1e-6) + 1e-6),
              np.log(rel[4] / (rel[2] + rel[3] + 1e-6) + 1e-6)]
    names += [f"{prefix}_delta_beta", f"{prefix}_theta_alpha", f"{prefix}_sigma_ratio"]
    return np.column_stack(feats), names


def _eog_features(x: np.ndarray, fs: float) -> tuple[np.ndarray, list[str]]:
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    slow = p[:, (f >= 0.3) & (f < 2.0)].sum(axis=1)
    fast = p[:, (f >= 2.0) & (f < 8.0)].sum(axis=1)
    return (np.column_stack([np.log(slow + 1e-12), np.log(fast + 1e-12), np.log(x.var(axis=1) + 1e-12)]),
            ["eog_log_slow", "eog_log_fast", "eog_logvar"])


def epoch_matrix(sig: np.ndarray, fs: float, start: int, end: int) -> np.ndarray:
    n = int(EPOCH_S * fs)
    return np.asarray(sig[start * n:end * n], dtype=np.float64).reshape(end - start, n)


def epoch_features(frontal: np.ndarray, parietal: np.ndarray, eog: np.ndarray | None, fs: float,
                   start: int, end: int) -> tuple[np.ndarray, list[str]]:
    """Raw (unstandardised) features for epochs [start, end)."""
    a, na = _eeg_features(epoch_matrix(frontal, fs, start, end), fs, "fc")
    b, nb = _eeg_features(epoch_matrix(parietal, fs, start, end), fs, "po")
    blocks, names = [a, b], na + nb
    if eog is not None:
        c, nc = _eog_features(epoch_matrix(eog, fs, start, end), fs)
        blocks.append(c)
        names += nc
    return np.column_stack(blocks), names


def standardise_and_context(F: np.ndarray, names: list[str], context: int = CONTEXT) -> tuple[np.ndarray, list[str]]:
    """Per-night robust z-score (median / IQR) + neighbouring epochs (edge-padded)."""
    med = np.median(F, axis=0)
    iqr = np.subtract(*np.percentile(F, [75, 25], axis=0))
    Z = (F - med) / np.where(iqr > 1e-9, iqr, 1.0)
    cols, out_names = [], []
    for k in range(-context, context + 1):
        idx = np.clip(np.arange(len(Z)) + k, 0, len(Z) - 1)
        cols.append(Z[idx])
        out_names += [f"{n}@{k:+d}" for n in names]
    return np.hstack(cols), out_names


# ---------------------------------------------------------------------------
# multiclass tree ensemble (JSON, numpy prediction)
# ---------------------------------------------------------------------------

@dataclass
class MulticlassTreeModel:
    feature_names: list[str]
    classes: list[str]
    baseline: list[float]
    trees: list[list[dict]]  # per boosting iteration: one tree per class
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        from .ml_ictal import TreeEnsembleModel

        # reuse the binary ensemble's traversal: one additive model per class
        self._per_class = [TreeEnsembleModel(self.feature_names, b, [it[k] for it in self.trees], {})
                           for k, b in enumerate(self.baseline)]

    def decision(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([m.decision(X) for m in self._per_class])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = self.decision(X)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def to_json(self) -> dict:
        strip = lambda t: {k: v for k, v in t.items() if not k.startswith("_")}
        return {"type": "multiclass_tree_ensemble", "feature_names": self.feature_names,
                "classes": self.classes, "baseline": self.baseline,
                "trees": [[strip(t) for t in it] for it in self.trees], "meta": self.meta}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), separators=(",", ":")), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MulticlassTreeModel":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["feature_names"], d["classes"], d["baseline"], d["trees"], d.get("meta", {}))


def export_multiclass_gbm(clf, feature_names: list[str], classes: list[str], meta: dict | None = None):
    trees = []
    for it in clf._predictors:
        row = []
        for pred in it:
            nd = pred.nodes
            row.append({"feature": nd["feature_idx"].astype(int).tolist(),
                        "threshold": [float(v) for v in nd["num_threshold"]],
                        "left": nd["left"].astype(int).tolist(), "right": nd["right"].astype(int).tolist(),
                        "value": [float(v) for v in nd["value"]],
                        "is_leaf": nd["is_leaf"].astype(bool).tolist(),
                        "missing_left": nd["missing_go_to_left"].astype(bool).tolist()})
        trees.append(row)
    base = [float(v) for v in np.ravel(clf._baseline_prediction)]
    return MulticlassTreeModel(list(feature_names), list(classes), base, trees, dict(meta or {}))


GBM_PARAMS = {"max_iter": 200, "learning_rate": 0.1, "max_leaf_nodes": 31, "random_state": 0}


def fit_stager(X: np.ndarray, y: np.ndarray, feature_names: list[str], meta: dict | None = None):
    from sklearn.ensemble import HistGradientBoostingClassifier

    clf = HistGradientBoostingClassifier(class_weight="balanced", **GBM_PARAMS).fit(X, y)
    assert list(clf.classes_) == list(range(len(STAGES))), clf.classes_
    return export_multiclass_gbm(clf, feature_names, list(STAGES), {**(meta or {}), "params": GBM_PARAMS})


# ---------------------------------------------------------------------------
# rule-based reference (no training)
# ---------------------------------------------------------------------------

def rule_based_stages(F: np.ndarray, names: list[str]) -> np.ndarray:
    """AASM-style heuristic on raw epoch features (one night): slow-wave dominance
    -> N3; occipital alpha or high beta with eye movements -> W; sigma (spindle band)
    excess -> N2; theta-dominant with eye movements and no sigma -> REM; else N1."""
    col = {n: i for i, n in enumerate(names)}
    g = lambda n: F[:, col[n]]
    z = lambda v: (v - np.median(v)) / (np.subtract(*np.percentile(v, [75, 25])) + 1e-9)
    sdelta, alpha_po, beta = g("fc_rel_sdelta"), g("po_rel_alpha"), g("fc_rel_beta")
    sigma = z(g("fc_rel_sigma")) + z(g("po_rel_sigma"))
    eog = z(g("eog_log_fast")) if "eog_log_fast" in col else np.zeros(len(F))
    amp = z(g("fc_logpow"))
    out = np.full(len(F), STAGES.index("N1"))
    out[sigma > 0.5] = STAGES.index("N2")
    out[(sigma <= 0.0) & (eog > 0.5) & (alpha_po < np.percentile(alpha_po, 60))] = STAGES.index("REM")
    out[(sdelta > 0.5) & (amp > 0.5)] = STAGES.index("N3")
    out[(z(alpha_po) > 1.0) | (z(beta) > 1.0)] = STAGES.index("W")
    return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int = len(STAGES)) -> np.ndarray:
    m = np.zeros((k, k), dtype=int)
    np.add.at(m, (y_true, y_pred), 1)
    return m


def scores_from_confusion(m: np.ndarray) -> dict:
    n = m.sum()
    acc = np.trace(m) / n if n else float("nan")
    pe = (m.sum(0) * m.sum(1)).sum() / n ** 2 if n else float("nan")
    kappa = (acc - pe) / (1 - pe) if n and pe < 1 else float("nan")
    f1 = {}
    for i, s in enumerate(STAGES):
        tp, fp, fn = m[i, i], m[:, i].sum() - m[i, i], m[i, :].sum() - m[i, i]
        f1[s] = round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) else float("nan")
    return {"n_epochs": int(n), "accuracy": round(float(acc), 4), "kappa": round(float(kappa), 4),
            "macro_f1": round(float(np.nanmean(list(f1.values()))), 4), "f1": f1,
            "confusion": m.tolist()}


REM_MIN_RUN = 3  # epochs (1.5 min)


def hypnogram_summary(stages: np.ndarray) -> dict:
    """Sleep metrics from a stage sequence (indices into STAGES), 30-s epochs."""
    stages = np.asarray(stages)
    sleep = stages != STAGES.index("W")
    tib_min = len(stages) * EPOCH_S / 60
    tst = sleep.sum() * EPOCH_S / 60
    first = int(np.argmax(sleep)) if sleep.any() else None
    # REM latency to the first *sustained* REM (>= REM_MIN_RUN consecutive epochs): a single
    # misstaged REM epoch early in the night would otherwise mimic a sleep-onset REM period
    # (a diagnostic sign of narcolepsy)
    is_rem = (stages == STAGES.index("REM")).astype(int)
    run = np.convolve(is_rem, np.ones(REM_MIN_RUN, dtype=int), mode="valid") if len(is_rem) >= REM_MIN_RUN else np.array([])
    rem = np.where(run == REM_MIN_RUN)[0]
    return {
        "time_in_bed_min": round(tib_min, 1), "total_sleep_min": round(tst, 1),
        "sleep_efficiency": round(tst / tib_min, 3) if tib_min else None,
        "sleep_latency_min": None if first is None else round(first * EPOCH_S / 60, 1),
        "rem_latency_min": None if (first is None or not len(rem)) else round((rem[0] - first) * EPOCH_S / 60, 1),
        "stage_percent_of_sleep": {s: round(100 * float((stages == i).sum()) / max(1, sleep.sum()), 1)
                                   for i, s in enumerate(STAGES) if s != "W"},
    }


# ---------------------------------------------------------------------------
# staging a NeuroLens signal (clinical 10-20 recording)
# ---------------------------------------------------------------------------

MODEL_FS = 100.0  # Sleep-EDF EEG sampling rate the model was trained on


def derivations(names: list[str], data: np.ndarray, parasagittal_fallback: bool = False) -> dict | None:
    """Sleep-EDF-like derivations from a 10-20 recording: Fpz-Cz (Fp1/Fp2 mean when
    Fpz is absent), Pz-Oz (O1/O2 mean when Oz is absent) and horizontal EOG (a
    dedicated EOG channel, else F7-F8). Without midline electrodes and with
    ``parasagittal_fallback``: Fp2-C4 / P4-O2 (left side if the right is missing),
    as validated on clinical PSG in increment 13b. None when electrodes are missing."""
    idx = {n.upper(): i for i, n in enumerate(names)}

    def ch(*opts):
        got = [data[idx[o.upper()]] for o in opts if o.upper() in idx]
        return np.mean(got, axis=0) if got else None

    fpz = ch("Fpz") if "FPZ" in idx else ch("Fp1", "Fp2")
    oz = ch("Oz") if "OZ" in idx else ch("O1", "O2")
    cz, pz = ch("Cz"), ch("Pz")
    if fpz is None or oz is None or cz is None or pz is None:
        side = next((s for s in (("FP2", "C4", "P4", "O2"), ("FP1", "C3", "P3", "O1"))
                     if all(e in idx for e in s)), None) if parasagittal_fallback else None
        if side is None:
            return None
        fp, c, pp, o = (data[idx[e]] for e in side)
        frontal, parietal = fp - c, pp - o
    else:
        frontal, parietal = fpz - cz, pz - oz
    eog_name = next((n for n in names if n.upper().startswith("EOG") or n.upper().startswith("LOC")), None)
    if eog_name is not None:
        eog = data[names.index(eog_name)]
    elif "F7" in idx and "F8" in idx:
        eog = data[idx["F7"]] - data[idx["F8"]]
    else:
        eog = None
    return {"frontal": frontal, "parietal": parietal, "eog": eog}


def stage_signal(names: list[str], data: np.ndarray, fs: float, model: MulticlassTreeModel,
                 parasagittal_fallback: bool = False,
                 parasagittal_model: MulticlassTreeModel | None = None) -> dict | None:
    """Hypnogram + sleep summary for a recording, or None if derivations are missing.
    ``parasagittal_model`` (a stager trained on clinical derivations) replaces ``model``
    when the parasagittal fallback is used."""
    from scipy.signal import resample_poly

    midline = derivations(names, data[:, :1]) is not None
    der = derivations(names, data, parasagittal_fallback)
    if der is None:
        return None
    if not midline and parasagittal_model is not None:
        model = parasagittal_model
    if abs(fs - MODEL_FS) > 1e-6:
        from fractions import Fraction

        fr = Fraction(MODEL_FS / fs).limit_denominator(1000)
        der = {k: (None if v is None else resample_poly(v, fr.numerator, fr.denominator)) for k, v in der.items()}
    n_ep = int(len(der["frontal"]) / MODEL_FS // EPOCH_S)
    if n_ep < 2 * CONTEXT + 1:
        return None
    F, names_f = epoch_features(der["frontal"], der["parietal"], der["eog"], MODEL_FS, 0, n_ep)
    if der["eog"] is None:  # model expects EOG features: neutral (median) values
        F = np.column_stack([F, np.zeros((n_ep, 3))])
        names_f = names_f + ["eog_log_slow", "eog_log_fast", "eog_logvar"]
    X, _ = standardise_and_context(F, names_f)
    proba = model.predict_proba(X)
    stages = proba.argmax(axis=1)
    return {
        "epoch_s": EPOCH_S,
        "hypnogram": [STAGES[i] for i in stages],
        "summary": hypnogram_summary(stages),
        "mean_confidence": round(float(proba.max(axis=1).mean()), 3),
        "derivations": "midline" if midline else "parasagittal",
        "eog_source": "eog_channel" if any(n.upper().startswith(("EOG", "LOC")) for n in names)
                      else ("F7-F8" if der["eog"] is not None else "none"),
    }
