"""Confidence calibration (TZ §13, §16).

Detector confidences are heuristic and uncalibrated. For mode-B gates (§2) and
alarm thresholds (§16) to be trustworthy, confidences must be *calibrated* — a
reported 0.8 should mean ~80% correct. This package provides calibration metrics
(ECE / MCE / Brier), per-code temperature scaling, and a builder that fits
calibration from the neurophysiologist feedback log (§13).
"""

from __future__ import annotations

from .metrics import expected_calibration_error, max_calibration_error, brier_score, reliability_bins
from .temperature import TemperatureScaler
from .calibrator import ConfidenceCalibrator
from .from_feedback import build_calibration

__all__ = [
    "expected_calibration_error",
    "max_calibration_error",
    "brier_score",
    "reliability_bins",
    "TemperatureScaler",
    "ConfidenceCalibrator",
    "build_calibration",
]
