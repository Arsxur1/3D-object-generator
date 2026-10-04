"""Temperature scaling (TZ §13, §16).

Calibrates probabilities by a single scalar temperature T:
    z = logit(p);  p' = sigmoid(z / T)
T > 1 softens over-confident probabilities toward 0.5; T < 1 sharpens. T is fit
by minimizing negative log-likelihood on held-out (confidence, label) pairs.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def apply_temperature(confidences, temperature: float) -> np.ndarray:
    """Return calibrated probabilities for a given temperature."""
    z = _logit(np.asarray(confidences, dtype=float))
    return _sigmoid(z / max(temperature, _EPS))


class TemperatureScaler:
    """Fits and applies a single temperature."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def fit(self, confidences, labels) -> "TemperatureScaler":
        conf = np.asarray(confidences, dtype=float)
        y = np.asarray(labels, dtype=float)
        # need both classes present and >=2 samples to calibrate
        if conf.size < 2 or len(np.unique(y)) < 2:
            self.temperature = 1.0
            return self
        z = _logit(conf)

        def nll(t: float) -> float:
            p = _sigmoid(z / max(t, _EPS))
            p = np.clip(p, _EPS, 1 - _EPS)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

        res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
        self.temperature = float(res.x) if res.success else 1.0
        return self

    def transform(self, confidences) -> np.ndarray:
        return apply_temperature(confidences, self.temperature)


def apply_platt(confidences, a: float, b: float) -> np.ndarray:
    """Platt scaling: p' = sigmoid(a * logit(p) + b)."""
    return _sigmoid(a * _logit(np.asarray(confidences, dtype=float)) + b)


class PlattScaler:
    """Affine logit calibration (Platt 1999).

    Unlike temperature scaling — which only rescales around 0.5 and so cannot
    map confidences that are all >= 0.5 down to a low true precision — Platt
    scaling also shifts the logit (bias ``b``). Needed when a detector runs at a
    high-sensitivity operating point where most events are false (e.g. the
    learned ictal detector at point A: ~10% of events are true seizures).
    """

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a, self.b = float(a), float(b)

    def fit(self, confidences, labels) -> "PlattScaler":
        from scipy.optimize import minimize

        conf = np.asarray(confidences, dtype=float)
        y = np.asarray(labels, dtype=float)
        if conf.size < 2 or len(np.unique(y)) < 2:
            self.a, self.b = 1.0, 0.0
            return self
        z = _logit(conf)

        def nll(theta):
            p = np.clip(_sigmoid(theta[0] * z + theta[1]), _EPS, 1 - _EPS)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

        res = minimize(nll, np.array([1.0, 0.0]), method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 2000})
        self.a, self.b = (float(res.x[0]), float(res.x[1])) if res.success else (1.0, 0.0)
        return self

    def transform(self, confidences) -> np.ndarray:
        return apply_platt(confidences, self.a, self.b)
