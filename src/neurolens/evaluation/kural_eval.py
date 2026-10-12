"""External validation of IED detectors on the IFCN criteria dataset (Kural et al.
2020; increment 16, docs/preregistration_increment16.md).

Each of the 100 segments (one per patient, 11-14 s, 500 Hz, 25 electrodes) contains
one marked sharp transient ("Marker: Annotation"); the gold standard says whether the
patient has epilepsy (transient = IED) or not (benign sharp transient). The question
is the one that matters for over-reading: does the detector's score at the marked
transient separate the two groups?

* learned score - maximal window probability of the frozen ``ied_gbm_v1`` over windows
  overlapping [t* - 0.5 s, t* + 0.5 s];
* heuristic score - maximal sharpness z of :class:`SpikeDetector` peaks (z >= 4, the
  30 µV gate kept) within the same interval; 0 when none;
* binary calls - learned: a detection at the frozen threshold overlaps the interval;
  heuristic: a default (z = 6) peak lies in it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..datasets.kural_ied import KuralCase
from ..layer4_detect import ied_ml

HALF_WINDOW_S = 0.5
HEURISTIC_MIN_Z = 4.0
HEURISTIC_DEFAULT_Z = 6.0


def transient_time(path: Path) -> float:
    """Seconds from the first sample to the "Marker: Annotation" marker."""
    from ..layer1_ingest.edf import PlainEdfReader

    r = PlainEdfReader(path)
    on, _, tx = r.readAnnotations()
    marks = [o for o, t in zip(on, tx) if t.strip().lower() == "marker: annotation"]
    if len(marks) != 1:
        raise ValueError(f"{path.name}: expected one 'Marker: Annotation', found {len(marks)}")
    first = r.record_onsets[0] if r.record_onsets else 0.0
    return float(marks[0] - first)


@dataclass
class CaseScore:
    sample: str
    epileptic: bool
    age_years: Optional[float]
    t_marker: float
    learned_score: float
    learned_call: bool
    heuristic_score: float
    heuristic_call: bool
    n_channels: int
    flags: list[str]


def score_case(case: KuralCase, path: Path, cfg, model, threshold: float) -> CaseScore:
    from ..layer1_ingest.registry import ingest
    from ..layer2_preprocess.filters import apply_filters
    from ..layer2_preprocess.reref import rereference
    from ..layer2_preprocess.sqi import amplitude_flag
    from ..layer4_detect.ied import SpikeDetector

    t = transient_time(path)
    sig = ingest(path)
    filt = apply_filters(sig, cfg.filters)
    flags = list(sig.quality.flags)
    af = amplitude_flag(filt)
    if af:
        flags.append(af)
    a = rereference(filt, "average")
    lo, hi = t - HALF_WINDOW_S, t + HALF_WINDOW_S
    X, t0 = ied_ml.window_features(a.signal, a.sampling_rate_hz, a.channel_names)
    p = model.predict_proba(X) if X.shape[0] else np.zeros(0)
    near = (t0 < hi) & (t0 + ied_ml.WIN_S > lo)
    learned = float(p[near].max()) if near.any() else 0.0
    call = any(s < hi and e > lo for s, e, _ in ied_ml.probability_events(t0, p, threshold))
    zs = [ev.evidence[0].value for ev in SpikeDetector(sharpness_z=HEURISTIC_MIN_Z, max_events=None)
          .detect(a, None, None) if lo <= (ev.t_start + ev.t_end) / 2 <= hi]
    hz = float(max(zs)) if zs else 0.0
    return CaseScore(case.sample, case.epileptic, case.age_years, round(t, 3), learned, bool(call),
                     hz, hz >= HEURISTIC_DEFAULT_Z, len(ied_ml.standard_channels(a.channel_names)), flags)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC (ties count 1/2)."""
    pos, neg = scores[labels], scores[~labels]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (pos.size * neg.size))


def bootstrap_auc(a: np.ndarray, b: Optional[np.ndarray], y: np.ndarray, n: int = 2000,
                  seed: int = 0) -> dict:
    """Stratified bootstrap CI for AUC(a) and, with ``b``, for AUC(a) - AUC(b) (paired)."""
    rng = np.random.default_rng(seed)
    ip, ineg = np.where(y)[0], np.where(~y)[0]
    va, vd = [], []
    for _ in range(n):
        i = np.concatenate([rng.choice(ip, ip.size), rng.choice(ineg, ineg.size)])
        va.append(auc(a[i], y[i]))
        if b is not None:
            vd.append(auc(a[i], y[i]) - auc(b[i], y[i]))
    out = {"ci95": [float(v) for v in np.percentile(va, [2.5, 97.5])]}
    if b is not None:
        out["diff_ci95"] = [float(v) for v in np.percentile(vd, [2.5, 97.5])]
    return out


def summarize(scores: list[CaseScore]) -> dict:
    from .prereg import clopper_pearson

    y = np.array([s.epileptic for s in scores])
    L = np.array([s.learned_score for s in scores])
    H = np.array([s.heuristic_score for s in scores])

    def binary(calls):
        c = np.array(calls)
        tp, fn = int((c & y).sum()), int((~c & y).sum())
        tn, fp = int((~c & ~y).sum()), int((c & ~y).sum())
        sens, spec = tp / (tp + fn), tn / (tn + fp)
        return {"tp": tp, "fn": fn, "tn": tn, "fp": fp, "sensitivity": sens,
                "sensitivity_ci95": list(clopper_pearson(tp, tp + fn)),
                "specificity": spec, "specificity_ci95": list(clopper_pearson(tn, tn + fp)),
                "accuracy": (tp + tn) / len(c), "balanced_accuracy": (sens + spec) / 2}

    bl = bootstrap_auc(L, H, y)
    bh = bootstrap_auc(H, None, y)
    return {
        "n": len(scores), "n_epileptic": int(y.sum()), "n_non_epileptic": int((~y).sum()),
        "learned": {"auc": auc(L, y), "auc_ci95": bl["ci95"],
                    "at_frozen_threshold": binary([s.learned_call for s in scores])},
        "heuristic": {"auc": auc(H, y), "auc_ci95": bh["ci95"],
                      "at_default_z": binary([s.heuristic_call for s in scores])},
        "auc_difference_learned_minus_heuristic": auc(L, y) - auc(H, y),
        "auc_difference_ci95": bl["diff_ci95"],
        "flags": sorted({f for s in scores for f in s.flags}),
    }


def save(scores: list[CaseScore], summary: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "cases": [s.__dict__ for s in scores]},
                              indent=1, default=float) + "\n", encoding="utf-8")
