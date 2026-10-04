"""Training and patient-wise cross-validation of the learned ictal detector.

* Epoch labels: 1 inside an expert seizure; epochs within the scoring tolerance
  around a seizure (onset − pre_s … onset, offset … offset + post_s) are left
  out of training (ambiguous), everything else is 0.
* Leave-one-subject-out (LOSO): every subject's records are predicted by a model
  that never saw that subject — out-of-fold probabilities estimate performance
  on new patients without touching any test set.
* Event post-processing (threshold, minimum run length) is chosen on the
  out-of-fold probabilities with the same selection rules as threshold tuning
  (``TuneResult.select``), so all choices are made on training data only.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from ..layer4_detect.ml_ictal import (
    FEATURE_NAMES,
    FEATURE_NAMES_V2,
    LogisticModel,
    TreeEnsembleModel,
    export_hist_gbm,
    featurize,
    fit_logistic,
    names_for_width,
    probability_runs,
)
from .matching import Detection, aggregate, score_record
from .runner import PreparedRecord, ScoringRules
from .tune import Trial, TuneResult

POSTPROC_GRID: dict[str, list[Any]] = {
    "threshold": [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99],
    "min_epochs": [3, 5, 8, 12, 20],
}


def subject_key(rec: PreparedRecord) -> str:
    return f"{rec.annotation.database}:{rec.annotation.subject}"


def record_xy(rec: PreparedRecord, rules: ScoringRules | None = None, version: int = 1,
              age_years: float | None = None):
    """(X, y, train_mask, epoch_times) for one record."""
    rules = rules or ScoringRules()
    X = featurize(rec.features, version, age_years=age_years) if version == 4 else featurize(rec.features, version)
    t = np.asarray(rec.features.epoch_times, float)
    y = np.zeros(t.size)
    keep = np.ones(t.size, bool)
    for s in rec.annotation.seizures:
        y[(t >= s.onset_s) & (t <= s.offset_s)] = 1.0
        keep[((t >= s.onset_s - rules.pre_s) & (t < s.onset_s)) |
             ((t > s.offset_s) & (t <= s.offset_s + rules.post_s))] = False
    return X, y, keep, t


@dataclass
class Dataset:
    records: list[PreparedRecord]
    X: list[np.ndarray]
    y: list[np.ndarray]
    keep: list[np.ndarray]
    times: list[np.ndarray]

    @classmethod
    def build(cls, records: list[PreparedRecord], rules: ScoringRules | None = None,
              version: int = 1, ages: dict[str, float] | None = None) -> "Dataset":
        """``ages``: subject key ("db:subject") -> age in years (feature version 4)."""
        ages = ages or {}
        parts = [record_xy(r, rules, version, ages.get(subject_key(r))) for r in records]
        return cls(records, [p[0] for p in parts], [p[1] for p in parts],
                   [p[2] for p in parts], [p[3] for p in parts])

    def stacked(self, idx: list[int]):
        X = np.vstack([self.X[i][self.keep[i]] for i in idx])
        y = np.concatenate([self.y[i][self.keep[i]] for i in idx])
        return X, y


GBM_PARAMS = {"max_iter": 200, "learning_rate": 0.1, "max_leaf_nodes": 31, "random_state": 0}


def fit_gbm(X: np.ndarray, y: np.ndarray, params: dict | None = None) -> TreeEnsembleModel:
    """Gradient-boosted trees (requires scikit-learn: ``pip install -e '.[ml]'``)."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    p = {**GBM_PARAMS, **(params or {})}
    clf = HistGradientBoostingClassifier(class_weight="balanced", **p).fit(X, y)
    import sklearn

    names = names_for_width(X.shape[1])
    return export_hist_gbm(clf, names, {
        "trainer": f"sklearn {sklearn.__version__} HistGradientBoostingClassifier",
        "params": p, "n_train": int(len(y)), "n_pos": int(np.sum(y)),
    })


def train_model(ds: Dataset, idx: list[int] | None = None, l2: float = 1.0, kind: str = "gbm"):
    idx = list(range(len(ds.records))) if idx is None else idx
    X, y = ds.stacked(idx)
    if kind == "gbm":
        return fit_gbm(X, y)
    if kind == "lr":
        return fit_logistic(X, y, l2=l2)
    raise ValueError(f"unknown model kind {kind!r}")


def loso_probabilities(ds: Dataset, l2: float = 1.0, progress=None, kind: str = "gbm") -> list[np.ndarray]:
    """Out-of-fold epoch probabilities, one model per held-out subject."""
    subjects = sorted({subject_key(r) for r in ds.records})
    probs: list[np.ndarray | None] = [None] * len(ds.records)
    for s in subjects:
        test = [i for i, r in enumerate(ds.records) if subject_key(r) == s]
        train = [i for i, r in enumerate(ds.records) if subject_key(r) != s]
        model = train_model(ds, train, l2, kind)
        for i in test:
            probs[i] = model.predict_proba(ds.X[i])
        if progress:
            progress(f"fold {s}: {len(test)} records")
    return probs  # type: ignore[return-value]


def detections_from_probs(times: np.ndarray, p: np.ndarray, threshold: float,
                          min_epochs: int, causal: bool = False) -> list[Detection]:
    """Runs -> detections. ``causal``: the system only *knows* once the run has
    lasted ``min_epochs`` (streaming latency); offline reports run start."""
    return [Detection(float(times[a]), float(times[b - 1]), pm,
                      t_known=float(times[min(b - 1, a + min_epochs - 1)]) if causal else None)
            for a, b, pm in probability_runs(p, threshold, min_epochs)]


def score_probs(ds: Dataset, probs: list[np.ndarray], threshold: float, min_epochs: int,
                rules: ScoringRules | None = None, causal: bool = False):
    rules = rules or ScoringRules()
    scores = []
    for rec, t, p in zip(ds.records, ds.times, probs):
        scores.append(score_record(
            rec.annotation.file, rec.annotation.seizures,
            detections_from_probs(t, p, threshold, min_epochs, causal),
            rec.duration_s, rules.pre_s, rules.post_s, rules.merge_gap_s,
        ))
    return scores


def postproc_search(ds: Dataset, probs: list[np.ndarray], fa_target: float = 1.0,
                    grid: dict[str, list[Any]] | None = None,
                    rules: ScoringRules | None = None, causal: bool = False) -> TuneResult:
    grid = grid or POSTPROC_GRID
    names = list(grid)
    trials = []
    for combo in itertools.product(*(grid[k] for k in names)):
        params = dict(zip(names, combo))
        trials.append(Trial(params, aggregate(score_probs(ds, probs, rules=rules, causal=causal, **params))))
    best = max(trials, key=lambda t: t.key(fa_target))
    return TuneResult(best=best, baseline=best, trials=trials, fa_target=fa_target, grid=grid)


# ---------------------------------------------------------------------------
# real-time path: epochs as the streaming monitor sees them
# ---------------------------------------------------------------------------

def stream_from_trace(trace) -> SimpleNamespace:
    """Concatenate the *newest* epochs of each monitor window (those the stream
    had not seen before) into one causal epoch stream with absolute times —
    exactly what :class:`StreamingFeaturizer` receives in deployment."""
    times, rms, conc, domf, rp = [], [], [], [], {}
    last = -np.inf
    channels = None
    for t0, _t1, feats, _baseline in trace.windows:
        et = t0 + np.asarray(feats.epoch_times, float)
        new = et > last + 1e-6
        if not new.any():
            continue
        channels = channels or list(feats.eeg_channels)
        times.append(et[new])
        rms.append(np.asarray(feats.epoch_rms)[new])
        conc.append(np.asarray(feats.epoch_band_conc)[new])
        domf.append(np.asarray(feats.epoch_domfreq)[new])
        for k, v in feats.epoch_relpow.items():
            rp.setdefault(k, []).append(np.asarray(v)[new])
        last = et[new][-1]
    return SimpleNamespace(
        epoch_times=np.concatenate(times), epoch_rms=np.vstack(rms),
        epoch_band_conc=np.vstack(conc), epoch_domfreq=np.vstack(domf),
        epoch_relpow={k: np.vstack(v) for k, v in rp.items()}, eeg_channels=channels or [],
    )


def realtime_records(traces) -> list[PreparedRecord]:
    """PreparedRecord views whose features are the monitor's epoch stream."""
    return [PreparedRecord(annotation=tr.annotation, path=Path(tr.annotation.file),
                           duration_s=tr.duration_s, features=stream_from_trace(tr))
            for tr in traces]


def dataset_ages() -> dict[str, float]:
    """Subject key -> age for both PhysioNet databases (feature version 4)."""
    from ..datasets.physionet import PhysioNetClient

    out = {}
    for db in ("chbmit", "siena"):
        for sub, age in PhysioNetClient(db).subject_ages().items():
            out[f"{db}:{sub}"] = age
    return out
