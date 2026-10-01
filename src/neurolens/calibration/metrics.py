"""Calibration metrics (TZ §16: ECE below threshold)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    avg_confidence: float
    accuracy: float


def reliability_bins(confidences, labels, n_bins: int = 10) -> list[ReliabilityBin]:
    """Bin predictions by confidence; per bin report avg confidence vs accuracy."""
    conf = np.asarray(confidences, dtype=float)
    y = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[ReliabilityBin] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            out.append(ReliabilityBin(lo, hi, 0, 0.0, 0.0))
        else:
            out.append(ReliabilityBin(lo, hi, cnt, float(conf[mask].mean()), float(y[mask].mean())))
    return out


def expected_calibration_error(confidences, labels, n_bins: int = 10) -> float:
    """ECE = sum_b (n_b/N) * |acc_b - conf_b| (lower is better)."""
    conf = np.asarray(confidences, dtype=float)
    n = len(conf)
    if n == 0:
        return 0.0
    ece = 0.0
    for b in reliability_bins(conf, labels, n_bins):
        if b.count:
            ece += (b.count / n) * abs(b.accuracy - b.avg_confidence)
    return float(ece)


def max_calibration_error(confidences, labels, n_bins: int = 10) -> float:
    """MCE = max_b |acc_b - conf_b|."""
    gaps = [abs(b.accuracy - b.avg_confidence) for b in reliability_bins(confidences, labels, n_bins) if b.count]
    return float(max(gaps)) if gaps else 0.0


def brier_score(confidences, labels) -> float:
    """Mean squared error between confidence and outcome (lower is better)."""
    conf = np.asarray(confidences, dtype=float)
    y = np.asarray(labels, dtype=float)
    if conf.size == 0:
        return 0.0
    return float(np.mean((conf - y) ** 2))
