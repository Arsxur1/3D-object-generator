"""Data-driven threshold tuning + confidence calibration on annotated EEG.

"Learning" here is deliberately transparent: an exhaustive grid search over a
handful of physiologically meaningful ictal-detector thresholds, selected by an
explicit clinical objective (TZ §16) —

    maximise event sensitivity  subject to  false alarms/hour <= target,
    then fewer FA/h, then shorter median latency —

on *training patients only*. The selected thresholds are then reported on
held-out patients (record-wise / patient-wise split, no leakage). Detector
confidences are calibrated by temperature scaling on the training events
(label = event overlaps an expert seizure), reusing ``neurolens.calibration``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..calibration.calibrator import ConfidenceCalibrator
from ..calibration.metrics import brier_score, expected_calibration_error
from ..calibration.temperature import TemperatureScaler
from ..contracts.config import Thresholds
from ..layer4_detect.ictal import IctalRhythmDetector
from .matching import AggregateScore
from .runner import PreparedRecord, ScoringRules, evaluate_offline

DEFAULT_GRID: dict[str, list[Any]] = {
    "ictal_min_channels": [1, 2, 3, 4, 5, 6],
    "ictal_min_duration_s": [6.0, 8.0, 10.0, 12.0, 16.0],
    "ictal_rhythmicity": [0.45, 0.55, 0.65],
    "ictal_amplitude_factor": [1.5, 2.0, 2.5, 3.0],
}


@dataclass
class Trial:
    params: dict[str, Any]
    score: AggregateScore

    def key(self, fa_target: float) -> tuple:
        s = self.score
        meets = s.fa_per_hour <= fa_target
        lat = s.latency_median_s if s.latency_median_s is not None else 1e9
        # feasible first; then sensitivity; then fewer FA; then faster
        return (meets, round(s.sensitivity, 6), -round(s.fa_per_hour, 6), -lat)


SELECTORS = ("strict", "robust", "margin")


@dataclass
class TuneResult:
    best: Trial
    baseline: Trial
    trials: list[Trial] = field(default_factory=list)
    fa_target: float = 1.0
    grid: dict[str, list[Any]] = field(default_factory=dict)

    def top(self, n: int = 10) -> list[Trial]:
        return sorted(self.trials, key=lambda t: t.key(self.fa_target), reverse=True)[:n]

    def select(self, rule: str = "strict") -> Trial:
        """Pick the operating point by a named rule.

        * ``strict``  — max training sensitivity s.t. FA/h <= target, then fewest FA,
          then shortest latency (increment 8; tends to pick the sharpest corner of
          the grid and over-fit, see docs/validation_physionet.md).
        * ``robust``  — the same objective evaluated on each setting's *worst grid
          neighbour* (every parameter moved one step either way): a setting must
          keep its sensitivity and FA budget under small threshold perturbations.
          Ties: own sensitivity, fewer own FA, shorter latency. (On training data this
          trades sensitivity for FA robustness — see docs/preregistration_increment9.md.)
        * ``margin``  — "spend the FA budget on a sensitivity margin": own FA/h <=
          target, max own sensitivity, then max *worst-neighbour sensitivity* (how
          far thresholds can drift before a seizure is missed), then fewer FA,
          shorter latency.
        """
        if rule == "strict":
            return max(self.trials, key=lambda t: t.key(self.fa_target))
        if rule in ("robust", "margin"):
            rob = robust_scores(self)
            keyf = _robust_key if rule == "robust" else _margin_key
            return max(self.trials, key=lambda t: keyf(t, rob[id(t)], self.fa_target))
        raise ValueError(f"unknown selector {rule!r}; known: {SELECTORS}")


def _grid_neighbours(trials: list[Trial], grid: dict[str, list[Any]]) -> dict[int, list[Trial]]:
    names = list(grid)
    pos = {k: {v: i for i, v in enumerate(vals)} for k, vals in grid.items()}
    by_idx = {tuple(pos[k][t.params[k]] for k in names): t for t in trials}
    out: dict[int, list[Trial]] = {}
    for idx, t in by_idx.items():
        nb = [t]
        for d in range(len(names)):
            for step in (-1, 1):
                j = list(idx)
                j[d] += step
                if tuple(j) in by_idx:
                    nb.append(by_idx[tuple(j)])
        out[id(t)] = nb
    return out


def robust_scores(res: "TuneResult") -> dict[int, tuple[float, float]]:
    """id(trial) -> (worst-neighbour sensitivity, worst-neighbour FA/h)."""
    nbs = _grid_neighbours(res.trials, res.grid)
    return {
        tid: (min(n.score.sensitivity for n in nb), max(n.score.fa_per_hour for n in nb))
        for tid, nb in nbs.items()
    }


def _margin_key(t: Trial, rob: tuple[float, float], fa_target: float) -> tuple:
    r_sens, _ = rob
    lat = t.score.latency_median_s if t.score.latency_median_s is not None else 1e9
    return (t.score.fa_per_hour <= fa_target, round(t.score.sensitivity, 6), round(r_sens, 6),
            -round(t.score.fa_per_hour, 6), -lat)


def _robust_key(t: Trial, rob: tuple[float, float], fa_target: float) -> tuple:
    r_sens, r_fa = rob
    lat = t.score.latency_median_s if t.score.latency_median_s is not None else 1e9
    return (r_fa <= fa_target, round(r_sens, 6), round(t.score.sensitivity, 6),
            -round(t.score.fa_per_hour, 6), -lat)


def grid_search(
    train: list[PreparedRecord],
    base: Thresholds,
    grid: dict[str, list[Any]] | None = None,
    fa_target: float = 1.0,
    rules: ScoringRules | None = None,
) -> TuneResult:
    grid = grid or DEFAULT_GRID
    names = list(grid)
    baseline = Trial({k: getattr(base, k) for k in names},
                     evaluate_offline(train, base, rules).total)
    trials = []
    for combo in itertools.product(*(grid[k] for k in names)):
        params = dict(zip(names, combo))
        th = base.model_copy(update=params)
        trials.append(Trial(params, evaluate_offline(train, th, rules).total))
    best = max(trials, key=lambda t: t.key(fa_target))
    return TuneResult(best=best, baseline=baseline, trials=trials, fa_target=fa_target, grid=grid)


def event_confidences(
    records: list[PreparedRecord], thresholds: Thresholds, rules: ScoringRules | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(confidence, is_true_seizure) for every ictal event emitted on records."""
    rules = rules or ScoringRules()
    conf, lab = [], []
    for rec in records:
        for e in IctalRhythmDetector().detect(rec.signal, rec.features, thresholds):
            hit = any(
                e.t_end >= s.onset_s - rules.pre_s and e.t_start <= s.offset_s + rules.post_s
                for s in rec.annotation.seizures
            )
            conf.append(e.confidence)
            lab.append(1.0 if hit else 0.0)
    return np.asarray(conf), np.asarray(lab)


def fit_ictal_calibration(
    train: list[PreparedRecord], thresholds: Thresholds, rules: ScoringRules | None = None,
    base: ConfidenceCalibrator | None = None,
) -> tuple[ConfidenceCalibrator, dict[str, Any]]:
    conf, lab = event_confidences(train, thresholds, rules)
    cal = base or ConfidenceCalibrator.identity()
    if conf.size < 5 or lab.min() == lab.max():
        info = {"n": int(conf.size), "positives": int(lab.sum()) if lab.size else 0,
                "skipped": "too few or single-class events (identity kept)"}
        cal.metrics[IctalRhythmDetector.code] = info
        return cal, info
    scaler = TemperatureScaler().fit(conf, lab)
    after = scaler.transform(conf)
    metrics = {
        "n": int(conf.size),
        "positives": int(lab.sum()),
        "temperature": round(float(scaler.temperature), 4),
        "ece_before": round(expected_calibration_error(conf, lab), 4),
        "ece_after": round(expected_calibration_error(after, lab), 4),
        "brier_before": round(brier_score(conf, lab), 4),
        "brier_after": round(brier_score(after, lab), 4),
        "source": "physionet event-level (training patients; see thresholds provenance)",
    }
    cal.temperatures[IctalRhythmDetector.code] = round(float(scaler.temperature), 4)
    cal.metrics[IctalRhythmDetector.code] = metrics
    return cal, metrics


def write_overrides(
    path: str | Path, params: dict[str, Any], provenance: dict[str, Any],
    realtime_params: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: (float(v) if isinstance(v, float) else int(v) if isinstance(v, (int, np.integer)) else v)
             for k, v in params.items()}
    header = (
        "# Ictal-detector thresholds tuned on PhysioNet data (TZ §16); see provenance.\n"
        "# Generated by `neurolens tune` — do not edit by hand; re-run to update.\n"
        "# Use with: neurolens analyze/monitor/evaluate --overrides <this file>\n"
        "# WARNING: thresholds tuned on few patients may not transfer; validate on fresh\n"
        "# patients first (increment 9: 33-45% sensitivity, see docs/validation_physionet.md).\n"
    )
    doc: dict[str, Any] = {"thresholds": clean}
    if realtime_params:
        rp = {k: (float(v) if isinstance(v, float) else int(v)) for k, v in realtime_params.items()}
        pers = rp.pop("persistence_windows", None)
        rt_doc: dict[str, Any] = {"threshold_overrides": rp}
        if pers is not None:
            rt_doc["alarms"] = {"seizure": {"persistence_windows": int(pers)}}
        doc["realtime"] = rt_doc
    doc["provenance"] = provenance
    body = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
    path.write_text(header + body, encoding="utf-8")
    return path


REALTIME_GRID: dict[str, list[Any]] = {
    "ictal_min_channels": [1, 2, 3],
    "ictal_min_duration_s": [6.0, 8.0],
    "ictal_rhythmicity": [0.45, 0.55],
    "ictal_amplitude_factor": [1.5, 2.0, 2.5],
    "persistence_windows": [1, 2],
}


def grid_search_realtime(
    traces: list,
    base: Thresholds,
    rt,
    grid: dict[str, list[Any]] | None = None,
    fa_target: float = 1.0,
    rules: ScoringRules | None = None,
) -> TuneResult:
    """Same objective as :func:`grid_search`, scored on replayed monitor traces.

    Threshold params become ``realtime.threshold_overrides``;
    ``persistence_windows`` sets the seizure alarm rule.
    """
    from .realtime_replay import evaluate_replay, with_realtime_params

    grid = grid or REALTIME_GRID
    names = list(grid)
    seizure_rule = rt.rule_for("seizure")
    base_params = {k: (seizure_rule.persistence_windows if k == "persistence_windows"
                       else rt.threshold_overrides.get(k, getattr(base, k))) for k in names}
    baseline = Trial(base_params, evaluate_replay(traces, base, rt, rules).total)
    trials = []
    for combo in itertools.product(*(grid[k] for k in names)):
        params = dict(zip(names, combo))
        trials.append(Trial(params, evaluate_replay(traces, base, with_realtime_params(rt, params), rules).total))
    best = max(trials, key=lambda t: t.key(fa_target))
    return TuneResult(best=best, baseline=baseline, trials=trials, fa_target=fa_target, grid=grid)
