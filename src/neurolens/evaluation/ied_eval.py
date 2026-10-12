"""Evaluation of interictal epileptiform discharge (IED) detection on NMT-Events
(increment 15).

Reference standard: the "epileptiform" annotations of an abnormal recording
(spike-and-wave, sharp-and-slow, sharp waves, polyspikes), across channels merged
into discharge events when they overlap or are < ``MERGE_GAP_S`` apart.

* **Sensitivity** - share of reference discharge events overlapped by a detection
  extended by ``TOL_S`` on each side (pooled over records; per-record macro and a
  record-level bootstrap interval are reported as well).
* **False detections per hour** - every detection in a *normal* recording (the
  dataset's normal recordings contain no epileptiform events by definition). In
  abnormal recordings the completeness of the annotation is unknown, so detections
  outside any annotation are reported as "unannotated", not as false.
* **Slowing hits** - share of non-epileptiform slowing events ("delta slow waves")
  that are overlapped by a detection (specificity against the commonest mimic).

Detectors: the heuristic :class:`~neurolens.layer4_detect.ied.SpikeDetector`
(uncapped; peaks on any channel within ``MERGE_GAP_S`` merged into one detection),
evaluated over a grid of sharpness thresholds, and the learned window model
(:mod:`neurolens.layer4_detect.ied_ml`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..datasets.nmt_events import AMPLITUDE_SCALE, EPILEPTIFORM, SLOWING, NMTAnnotation
from ..layer4_detect import ied_ml

MERGE_GAP_S = 0.25
TOL_S = 0.25
HEURISTIC_Z_GRID = (4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0)
FEATURE_VERSION = "ied-f1"


def merge_intervals(iv: list[tuple[float, float]], gap: float = MERGE_GAP_S) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def reference_events(ann: Optional[NMTAnnotation]) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """(epileptiform discharge events, slowing-only events)."""
    if ann is None:
        return [], []
    ep = merge_intervals(ann.intervals(EPILEPTIFORM))
    sl = [iv for iv in merge_intervals(ann.intervals(SLOWING))
          if not any(a < iv[1] + TOL_S and b > iv[0] - TOL_S for a, b in ep)]
    return ep, sl


def _overlaps(a: float, b: float, dets: np.ndarray, tol: float) -> bool:
    if dets.size == 0:
        return False
    return bool(np.any((dets[:, 0] < b + tol) & (dets[:, 1] > a - tol)))


def score_record(dets: list[tuple[float, float]], ep: list[tuple[float, float]],
                 sl: list[tuple[float, float]], duration_s: float, normal: bool,
                 tol: float = TOL_S) -> dict:
    D = np.asarray(dets, float).reshape(-1, 2)
    hit = sum(_overlaps(a, b, D, tol) for a, b in ep)
    sl_hit = sum(_overlaps(a, b, D, tol) for a, b in sl)
    allref = np.asarray(ep + sl, float).reshape(-1, 2)
    unannotated = sum(not _overlaps(a, b, allref, tol) for a, b in D) if not normal else 0
    return {"n_ref": len(ep), "n_hit": int(hit), "n_slow": len(sl), "n_slow_hit": int(sl_hit),
            "n_det": int(D.shape[0]), "n_false": int(D.shape[0]) if normal else 0,
            "n_unannotated": int(unannotated), "hours": duration_s / 3600.0, "normal": normal}


def aggregate(rows: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    ab = [r for r in rows if not r["normal"] and r["n_ref"] > 0]
    nm = [r for r in rows if r["normal"]]
    n_ref = sum(r["n_ref"] for r in ab)
    n_hit = sum(r["n_hit"] for r in ab)
    sens = n_hit / n_ref if n_ref else float("nan")
    macro = float(np.mean([r["n_hit"] / r["n_ref"] for r in ab])) if ab else float("nan")
    lo = hi = float("nan")
    if ab:
        rng = np.random.default_rng(seed)
        refs = np.array([r["n_ref"] for r in ab], float)
        hits = np.array([r["n_hit"] for r in ab], float)
        bs = []
        for _ in range(n_boot):
            i = rng.integers(0, len(ab), len(ab))
            bs.append(hits[i].sum() / refs[i].sum())
        lo, hi = (float(v) for v in np.percentile(bs, [2.5, 97.5]))
    nh = sum(r["hours"] for r in nm)
    nf = sum(r["n_false"] for r in nm)
    n_slow = sum(r["n_slow"] for r in rows if not r["normal"])
    ah = sum(r["hours"] for r in rows if not r["normal"])
    return {
        "sensitivity": sens, "sensitivity_ci95_record_bootstrap": [lo, hi],
        "sensitivity_macro": macro, "n_ref": n_ref, "n_hit": n_hit, "n_records_with_ref": len(ab),
        "fa_per_h_normal": nf / nh if nh else float("nan"), "n_false_normal": nf,
        "hours_normal": nh, "n_normal": len(nm),
        "normal_records_with_any_detection": sum(1 for r in nm if r["n_det"] > 0),
        "slowing_hit_rate": (sum(r["n_slow_hit"] for r in rows if not r["normal"]) / n_slow) if n_slow else float("nan"),
        "n_slowing": n_slow,
        "unannotated_per_h_abnormal": (sum(r["n_unannotated"] for r in rows if not r["normal"]) / ah) if ah else float("nan"),
    }


# ---------------------------------------------------------------------------
# record preparation (cached)
# ---------------------------------------------------------------------------

@dataclass
class IEDRecord:
    record_id: int
    normal: bool
    duration_s: float
    t0: np.ndarray
    X: np.ndarray
    heuristic: dict[float, list[tuple[float, float]]] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


def _cfg_hash(cfg) -> str:
    blob = json.dumps({"filters": cfg.filters.model_dump(mode="json"), "v": FEATURE_VERSION,
                       "z": HEURISTIC_Z_GRID, "scale": AMPLITUDE_SCALE}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def heuristic_detections(sig, z: float) -> list[tuple[float, float]]:
    from ..layer4_detect.ied import SpikeDetector

    evs = SpikeDetector(sharpness_z=z, max_events=None).detect(sig, None, None)
    return merge_intervals([(e.t_start, e.t_end) for e in evs])


def prepare_record(path: Path, record_id: int, normal: bool, cfg, cache_dir: Optional[Path] = None) -> IEDRecord:
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"{record_id:07d}_{'n' if normal else 'a'}_{_cfg_hash(cfg)}.npz"
        if cache.exists():
            d = np.load(cache, allow_pickle=False)
            heur = json.loads(str(d["heuristic"]))
            return IEDRecord(record_id, normal, float(d["duration_s"]), d["t0"], d["X"],
                             {float(k): [tuple(x) for x in v] for k, v in heur.items()},
                             json.loads(str(d["flags"])))
    from ..layer1_ingest.registry import ingest
    from ..layer2_preprocess.filters import apply_filters
    from ..layer2_preprocess.reref import rereference

    sig = ingest(path)
    sig = sig.with_signal((sig.signal * AMPLITUDE_SCALE).astype(np.float32))
    flags = list(sig.quality.flags)
    analysis = rereference(apply_filters(sig, cfg.filters), "average")
    X, t0 = ied_ml.window_features(analysis.signal, analysis.sampling_rate_hz, analysis.channel_names)
    heur = {z: heuristic_detections(analysis, z) for z in HEURISTIC_Z_GRID}
    rec = IEDRecord(record_id, normal, float(analysis.duration_s), t0, X.astype(np.float32), heur, flags)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, t0=t0, X=rec.X, duration_s=rec.duration_s,
                            heuristic=json.dumps({str(k): v for k, v in heur.items()}),
                            flags=json.dumps(flags))
    return rec


def training_xy(rec: IEDRecord, ann: Optional[NMTAnnotation]) -> tuple[np.ndarray, np.ndarray]:
    ep, _ = reference_events(ann)
    y = ied_ml.window_labels(rec.t0, ep)
    keep = y >= 0
    return rec.X[keep], y[keep].astype(int)


def learned_detections(rec: IEDRecord, p: np.ndarray, threshold: float) -> list[tuple[float, float]]:
    return [(a, b) for a, b, _ in ied_ml.probability_events(rec.t0, p, threshold)]


def evaluate(records: list[IEDRecord], anns: dict[int, NMTAnnotation], dets_of) -> dict:
    """``dets_of(rec) -> list[(start, end)]``; returns aggregate + per-record rows."""
    rows = []
    for rec in records:
        ep, sl = reference_events(None if rec.normal else anns.get(rec.record_id))
        r = score_record(dets_of(rec), ep, sl, rec.duration_s, rec.normal)
        r["record_id"] = rec.record_id
        rows.append(r)
    return {"aggregate": aggregate(rows), "records": rows}


# ---------------------------------------------------------------------------
# training-data cross-validation and operating point
# ---------------------------------------------------------------------------

IED_GBM_PARAMS = {"max_iter": 300, "learning_rate": 0.1, "max_leaf_nodes": 31,
                  "min_samples_leaf": 40, "l2_regularization": 1.0, "random_state": 0}
THRESHOLD_GRID = tuple(sorted({round(float(v), 3) for v in np.concatenate(
    [np.arange(0.30, 0.90, 0.05), np.arange(0.90, 0.991, 0.01)])}))


def fit_ied_gbm(X: np.ndarray, y: np.ndarray, params: dict | None = None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    import sklearn

    from ..layer4_detect.ml_ictal import export_hist_gbm

    p = {**IED_GBM_PARAMS, **(params or {})}
    clf = HistGradientBoostingClassifier(class_weight="balanced", **p).fit(X, y)
    return export_hist_gbm(clf, ied_ml.feature_names(), {
        "trainer": f"sklearn {sklearn.__version__} HistGradientBoostingClassifier",
        "params": p, "n_train": int(len(y)), "n_pos": int(np.sum(y)),
        "window_s": ied_ml.WIN_S, "hop_s": ied_ml.HOP_S, "features": FEATURE_VERSION,
    })


def fold_of(record_id: int, normal: bool, k: int = 5) -> int:
    """Deterministic grouped folds (one record = one person in NMT-Events)."""
    return (record_id * 7 + (3 if normal else 0)) % k


def cross_val_probabilities(records: list[IEDRecord], anns: dict[int, NMTAnnotation],
                            k: int = 5, params: dict | None = None, progress=None) -> list[np.ndarray]:
    xy = [training_xy(r, None if r.normal else anns.get(r.record_id)) for r in records]
    folds = [fold_of(r.record_id, r.normal, k) for r in records]
    out: list[Optional[np.ndarray]] = [None] * len(records)
    for f in range(k):
        tr = [i for i, g in enumerate(folds) if g != f]
        X = np.vstack([xy[i][0] for i in tr])
        y = np.concatenate([xy[i][1] for i in tr])
        model = fit_ied_gbm(X, y, params)
        for i, g in enumerate(folds):
            if g == f:
                out[i] = model.predict_proba(records[i].X) if records[i].X.shape[0] else np.zeros(0)
        if progress:
            progress(f)
    return out  # type: ignore[return-value]


def learned_curve(records, anns, probs, grid=THRESHOLD_GRID) -> list[dict]:
    pos = {id(r): p for r, p in zip(records, probs)}
    return [{"threshold": t, **evaluate(records, anns,
                                        lambda r, t=t: learned_detections(r, pos[id(r)], t))["aggregate"]}
            for t in grid]


def heuristic_curve(records, anns, grid=HEURISTIC_Z_GRID) -> list[dict]:
    return [{"sharpness_z": z, **evaluate(records, anns, lambda r, z=z: r.heuristic[z])["aggregate"]}
            for z in grid]


def operating_point(curve: list[dict], fa_target: float, key: str) -> Optional[dict]:
    """Highest-sensitivity point with FA/h on normal recordings <= fa_target."""
    ok = [c for c in curve if c["fa_per_h_normal"] <= fa_target]
    if not ok:
        return None
    return max(ok, key=lambda c: (c["sensitivity"], -c["fa_per_h_normal"], c[key] if key == "threshold" else -c[key]))
