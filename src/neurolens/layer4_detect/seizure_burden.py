"""Seizure-burden accounting for cEEG (TZ §7.3).

Aggregates ictal events into total seizure time, fraction of the record, and
seizures/hour — a strong clinical signal in continuous EEG monitoring.
"""

from __future__ import annotations

from ..contracts.events import DetectionResult, SeizureBurden

# Continuous ictal activity beyond this suggests status epilepticus (lenient demo value).
STATUS_DURATION_S = 30.0


def compute_seizure_burden(detection: DetectionResult, duration_s: float) -> SeizureBurden:
    ictal = detection.by_group("ictal")
    total = sum(e.duration_s for e in ictal)
    longest = max((e.duration_s for e in ictal), default=0.0)
    frac = (total / duration_s) if duration_s > 0 else 0.0
    per_hour = (len(ictal) / duration_s * 3600.0) if duration_s > 0 else 0.0
    return SeizureBurden(
        n_seizures=len(ictal),
        total_seizure_time_s=round(total, 1),
        recording_duration_s=round(duration_s, 1),
        seizure_fraction=round(min(1.0, frac), 4),
        seizures_per_hour=round(per_hour, 2),
        longest_seizure_s=round(longest, 1),
        status_epilepticus_suspected=longest >= STATUS_DURATION_S,
    )
