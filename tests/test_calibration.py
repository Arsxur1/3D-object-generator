"""Confidence calibration tests (TZ §13, §16)."""

from __future__ import annotations

import numpy as np

from neurolens.calibration.calibrator import ConfidenceCalibrator
from neurolens.calibration.from_feedback import build_calibration
from neurolens.calibration.metrics import (
    brier_score,
    expected_calibration_error,
)
from neurolens.calibration.temperature import TemperatureScaler
from neurolens.contracts.events import DetectionResult, Event, Localization
from neurolens.feedback.log import Correction, FeedbackLogger


def _overconfident(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    conf = rng.uniform(0.8, 0.99, n)          # claims high confidence
    y = (rng.uniform(size=n) < 0.6).astype(float)  # but only ~60% correct
    return conf, y


def test_ece_detects_miscalibration():
    conf, y = _overconfident()
    assert expected_calibration_error(conf, y) > 0.15
    # well-calibrated: p == P(correct)
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 3000)
    yy = (rng.uniform(size=p.size) < p).astype(float)
    assert expected_calibration_error(p, yy) < 0.06


def test_temperature_scaling_reduces_ece():
    conf, y = _overconfident()
    ts = TemperatureScaler().fit(conf, y)
    assert ts.temperature > 1.0  # softens over-confidence
    before = expected_calibration_error(conf, y)
    after = expected_calibration_error(ts.transform(conf), y)
    assert after < before


def test_temperature_identity_single_class():
    ts = TemperatureScaler().fit([0.9, 0.8, 0.7], [1, 1, 1])
    assert ts.temperature == 1.0  # cannot calibrate without both classes


def test_brier_bounds():
    assert brier_score([1.0, 0.0], [1.0, 0.0]) == 0.0
    assert brier_score([0.0, 1.0], [1.0, 0.0]) == 1.0


def test_calibrator_per_code_and_fallback(tmp_path):
    cal = ConfidenceCalibrator(temperatures={"spike": 4.0}, default=2.0)
    assert cal.temperature_for("spike") == 4.0
    assert cal.temperature_for("unknown_code") == 2.0
    # softening pulls a high confidence toward 0.5
    assert cal.calibrate("spike", 0.95) < 0.95
    # round-trip
    path = cal.save(tmp_path / "cal.json")
    loaded = ConfidenceCalibrator.load(path)
    assert loaded.temperatures == {"spike": 4.0} and loaded.default == 2.0


def test_apply_to_detection_preserves_raw():
    det = DetectionResult(events=[
        Event(code="ictal_rhythm", label_ru="x", label_uz="x", group="ictal",
              localization=Localization(), t_start=0, t_end=10, confidence=0.95),
    ])
    ConfidenceCalibrator(default=3.0).apply_to_detection(det)
    e = det.events[0]
    assert e.metadata["confidence_raw"] == 0.95
    assert e.confidence < 0.95


def test_build_calibration_from_feedback(tmp_path):
    fb = tmp_path / "fb.jsonl"
    log = FeedbackLogger(fb)
    rng = np.random.default_rng(3)
    for _ in range(60):  # spike @0.9 but ~50% correct -> needs T>1
        log.record(Correction(
            recording_id="r", target_kind="event", target_ref="spike",
            action="confirm" if rng.uniform() < 0.5 else "reject", model_confidence=0.9,
        ))
    cal = build_calibration(fb, out_path=tmp_path / "cal.json", min_samples=10)
    assert "spike" in cal.temperatures and cal.temperatures["spike"] > 1.0
    assert cal.metrics["ece_after"] < cal.metrics["ece_before"]


def test_pipeline_applies_calibration(demo_edf, configs, tmp_path):
    from neurolens.pipeline.pipeline import Pipeline

    calf = ConfidenceCalibrator(default=3.0).save(tmp_path / "cal.json")
    base = Pipeline(config=configs, provider_pref="deterministic").analyze_file(demo_edf)
    cal = Pipeline(config=configs, provider_pref="deterministic", calibration_file=calf).analyze_file(demo_edf)

    assert cal.result_json["calibration"]["applied"] is True
    top_raw = max(e.confidence for e in base.detection.events)
    top_cal = max(e.confidence for e in cal.detection.events)
    assert top_cal < top_raw  # softened
