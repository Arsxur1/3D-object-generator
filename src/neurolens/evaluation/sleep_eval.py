"""Sleep-staging evaluation on Sleep-EDF (epoch-level, AASM stages).

``prepare_night`` -> raw features + expert stages for the analysis window
(cached on disk); ``evaluate`` scores a frozen model and the rule-based
reference on the same nights: pooled confusion matrix, accuracy, Cohen's kappa,
macro-F1, per-stage F1 and per-night kappa. Epochs without a scorable expert
label ("?", movement) are excluded.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..datasets.sleepedf import SleepEDFClient, SleepRecord
from ..layer4_detect.sleep_staging import (
    MulticlassTreeModel,
    confusion,
    epoch_features,
    rule_based_stages,
    scores_from_confusion,
    standardise_and_context,
)

FEATURE_VERSION = 1


@dataclass
class Night:
    key: str
    subject: int
    F: np.ndarray            # raw epoch features (analysis window)
    names: list[str]
    y: np.ndarray            # stage index or -1 (not scored)


def prepare_night(client: SleepEDFClient, rec: SleepRecord, cache_dir: str | Path | None = None) -> Night:
    cp = None
    if cache_dir:
        path = client.local_path(rec.psg)
        st = path.stat()
        key = hashlib.sha1(f"{path.name}:{st.st_size}:{st.st_mtime_ns}:fv{FEATURE_VERSION}".encode()).hexdigest()[:12]
        cp = Path(cache_dir) / f"sleep_{rec.key}_{key}.pkl"
        if cp.exists():
            return pickle.loads(cp.read_bytes())
    sig, fs, rec = client.load(rec)
    a, b = rec.start_epoch, rec.end_epoch
    F, names = epoch_features(sig["EEG Fpz-Cz"], sig["EEG Pz-Oz"], sig.get("EOG horizontal"), fs, a, b)
    y = np.array([-1 if s is None else s for s in rec.stages[a:b]], dtype=int)
    night = Night(rec.key, rec.subject, F, names, y)
    if cp is not None:
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_bytes(pickle.dumps(night))
    return night


def _prepare_in_worker(data_dir, rec: SleepRecord, cache_dir) -> Night:
    # clients hold download callables (not picklable): each worker builds its own
    return prepare_night(SleepEDFClient(data_dir), rec, cache_dir)


def prepare_nights(client, recs, cache_dir=None, workers: int = 3, progress: Optional[Callable] = None) -> list[Night]:
    from concurrent.futures import ProcessPoolExecutor

    data_dir = client.cache.parent
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = [pool.submit(_prepare_in_worker, data_dir, r, cache_dir) for r in recs]
        out = []
        for r, f in zip(recs, futs):
            out.append(f.result())
            if progress:
                progress(f"prepared {r.key}")
    return out


def design(night: Night) -> tuple[np.ndarray, list[str]]:
    return standardise_and_context(night.F, night.names)


def training_matrix(nights: list[Night]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    Xs, ys, names = [], [], None
    for n in nights:
        X, names = design(n)
        keep = n.y >= 0
        Xs.append(X[keep])
        ys.append(n.y[keep])
    return np.vstack(Xs), np.concatenate(ys), names


def predict_night(model: MulticlassTreeModel, night: Night) -> np.ndarray:
    X, _ = design(night)
    return model.predict_proba(X).argmax(axis=1)


def score(nights: list[Night], preds: list[np.ndarray]) -> dict:
    total = np.zeros((5, 5), dtype=int)
    per = {}
    for n, p in zip(nights, preds):
        keep = n.y >= 0
        m = confusion(n.y[keep], p[keep])
        total += m
        per[n.key] = scores_from_confusion(m)["kappa"]
    out = scores_from_confusion(total)
    vals = np.array(list(per.values()), dtype=float)
    out["per_night_kappa"] = per
    out["per_night_kappa_median"] = round(float(np.nanmedian(vals)), 4)
    out["n_nights"] = len(nights)
    out["n_subjects"] = len({n.subject for n in nights})
    return out


def evaluate(nights: list[Night], model: MulticlassTreeModel) -> dict:
    return {
        "learned": score(nights, [predict_night(model, n) for n in nights]),
        "rule_based": score(nights, [rule_based_stages(n.F, n.names) for n in nights]),
    }


def grouped_cv(nights: list[Night], folds: int = 5, progress: Optional[Callable] = None) -> dict:
    """Subject-grouped cross-validation (fold = subject number mod folds)."""
    from ..layer4_detect.sleep_staging import fit_stager

    preds: dict[str, np.ndarray] = {}
    for k in range(folds):
        train = [n for n in nights if n.subject % folds != k]
        test = [n for n in nights if n.subject % folds == k]
        X, y, names = training_matrix(train)
        m = fit_stager(X, y, names)
        for n in test:
            preds[n.key] = predict_night(m, n)
        if progress:
            progress(f"fold {k}: {len(test)} nights")
    return score(nights, [preds[n.key] for n in nights])


def save(result: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "sleep_metrics.json"
    p.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# CAP Sleep Database (clinical transfer test, increment 13b)
# ---------------------------------------------------------------------------

def cap_headers(client) -> dict[str, list[str]]:
    """Channel labels of every CAP recording from its EDF header (HTTP range
    request, no signal data), cached as JSON."""
    import time
    import urllib.request

    path = client.cache / "headers.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for rec in [r for r in client.records() if r.endswith(".edf")]:
        for attempt in range(6):
            try:
                req = urllib.request.Request(f"{client.base}/{rec}",
                                             headers={"Range": "bytes=0-16383", "User-Agent": "neurolens/0.1"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    h = resp.read()
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(5 * (attempt + 1))
        ns = int(h[252:256])
        out[rec] = [h[256 + 16 * i: 256 + 16 * (i + 1)].decode("latin-1").strip() for i in range(ns)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def prepare_cap_night(client, rec_name: str, cache_dir: str | Path, keep_edf: bool = False) -> Night:
    """Download one CAP recording (+ scoring), compute features, delete the EDF."""
    import zlib

    from ..datasets.capsleep import read_cap

    cp = Path(cache_dir) / f"cap_{rec_name.removesuffix('.edf')}_fv{FEATURE_VERSION}.pkl"
    if cp.exists():
        return pickle.loads(cp.read_bytes())
    edf = client.fetch(rec_name)
    txt = client.text(rec_name.removesuffix(".edf") + ".txt")
    sig, rec = read_cap(edf, txt)
    a, b = rec.start_epoch, rec.end_epoch
    F, names = epoch_features(sig["frontal"], sig["parietal"], sig["eog"], 100.0, a, b)
    y = np.array([-1 if s is None else s for s in rec.stages[a:b]], dtype=int)
    night = Night(rec.name, zlib.crc32(rec.name.encode()), F, names, y)
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_bytes(pickle.dumps(night))
    if not keep_edf:
        edf.unlink(missing_ok=True)
    return night
