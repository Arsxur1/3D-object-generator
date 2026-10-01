"""Seizure-burden aggregation tests (TZ §7.3)."""

from __future__ import annotations

from neurolens.contracts.events import DetectionResult, Event, Localization
from neurolens.layer4_detect.seizure_burden import compute_seizure_burden


def _ictal(t0, t1):
    return Event(code="ictal_rhythm", label_ru="i", label_uz="i", group="ictal",
                 localization=Localization(), t_start=t0, t_end=t1, confidence=0.9)


def test_burden_aggregates_multiple_seizures():
    det = DetectionResult(events=[_ictal(10, 25), _ictal(100, 110)])
    sb = compute_seizure_burden(det, duration_s=600.0)
    assert sb.n_seizures == 2
    assert sb.total_seizure_time_s == 25.0
    assert sb.longest_seizure_s == 15.0
    assert abs(sb.seizure_fraction - 25.0 / 600.0) < 1e-3
    assert abs(sb.seizures_per_hour - 12.0) < 1e-6
    assert sb.status_epilepticus_suspected is False


def test_burden_flags_status():
    det = DetectionResult(events=[_ictal(0, 45)])
    sb = compute_seizure_burden(det, duration_s=300.0)
    assert sb.status_epilepticus_suspected is True


def test_burden_empty():
    sb = compute_seizure_burden(DetectionResult(events=[]), duration_s=300.0)
    assert sb.n_seizures == 0
    assert sb.total_seizure_time_s == 0.0
