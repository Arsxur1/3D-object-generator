"""Build calibration from the neurophysiologist feedback log (TZ §13).

Turns confirm/reject corrections (with the model's confidence at the time) into
labeled (confidence, correct) pairs, fits a pooled default temperature and
per-code temperatures where enough samples exist, and reports ECE before/after.
This is the concrete link between expert feedback (§13) and calibrated
confidences that make mode-B gates and alarms reliable (§16).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..feedback.log import FeedbackLogger
from .calibrator import ConfidenceCalibrator
from .metrics import expected_calibration_error
from .temperature import TemperatureScaler


def build_calibration(
    feedback_path: str | Path,
    out_path: str | Path | None = None,
    min_samples: int = 10,
) -> ConfidenceCalibrator:
    records = FeedbackLogger(feedback_path).read_all()

    per_code: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    pooled_p: list[float] = []
    pooled_y: list[float] = []
    for r in records:
        if r.model_confidence is None:
            continue
        if r.action == "confirm":
            y = 1.0
        elif r.action == "reject":
            y = 0.0
        else:  # 'edit' is ambiguous for a binary correctness label
            continue
        per_code[r.target_ref][0].append(float(r.model_confidence))
        per_code[r.target_ref][1].append(y)
        pooled_p.append(float(r.model_confidence))
        pooled_y.append(y)

    default_scaler = TemperatureScaler().fit(pooled_p, pooled_y)
    temperatures: dict[str, float] = {}
    for code, (ps, ys) in per_code.items():
        if len(ps) >= min_samples:
            temperatures[code] = round(TemperatureScaler().fit(ps, ys).temperature, 4)

    metrics = {}
    if pooled_p:
        metrics = {
            "n_samples": len(pooled_p),
            "ece_before": round(expected_calibration_error(pooled_p, pooled_y), 4),
            "ece_after": round(
                expected_calibration_error(default_scaler.transform(pooled_p), pooled_y), 4
            ),
        }

    cal = ConfidenceCalibrator(
        temperatures=temperatures,
        default=round(default_scaler.temperature, 4),
        metrics=metrics,
    )
    if out_path is not None:
        cal.save(out_path)
    return cal
