"""Electrographic status epilepticus (ESE) criteria — one definition for the offline
report, the critical findings and the real-time monitor.

* Adults and children — ACNS Standardized Critical Care EEG Terminology 2021
  (Hirsch LJ et al., J Clin Neurophysiol 38:1-29): electrographic seizure activity
  for >= 10 continuous minutes, or for a total duration of >= 20% of any 60-minute
  period of recording.
* Neonates — ACNS neonatal EEG terminology (Tsuchida TN et al., J Clin
  Neurophysiol 30:161-173, 2013): summed seizure duration >= 50% of any 1-hour epoch.

The hourly criterion is applied as an absolute amount of seizure time inside any
60-minute window (12 min / 30 min), so a short recording cannot reach "20%" with a
few minutes of seizures. Replaces a 30-second "demo" threshold that declared status
epilepticus after half a minute of seizure activity (found in an end-to-end run).
"""

from __future__ import annotations

ESE_CONTINUOUS_S = 600.0
HOUR_S = 3600.0
ESE_HOURLY_FRACTION = 0.20
NEONATAL_HOURLY_FRACTION = 0.50


def merge(intervals: list[tuple[float, float]], gap_s: float = 0.0) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1] + gap_s:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def max_seizure_time_in_window(intervals: list[tuple[float, float]], window_s: float = HOUR_S) -> float:
    """Largest total seizure time inside any window of ``window_s`` (window starts at
    an interval onset - the maximum is always attained there)."""
    iv = merge(intervals)
    best = 0.0
    for start, _ in iv:
        end = start + window_s
        best = max(best, sum(max(0.0, min(b, end) - max(a, start)) for a, b in iv))
    return best


def status_epilepticus(intervals: list[tuple[float, float]], neonate: bool = False) -> dict:
    """Whether the seizure intervals meet ESE criteria, and which one."""
    iv = merge(intervals)
    longest = max((b - a for a, b in iv), default=0.0)
    hourly = max_seizure_time_in_window(iv)
    frac = NEONATAL_HOURLY_FRACTION if neonate else ESE_HOURLY_FRACTION
    continuous = (not neonate) and longest >= ESE_CONTINUOUS_S
    burden = hourly >= frac * HOUR_S
    return {
        "suspected": bool(continuous or burden),
        "criterion": "continuous_10min" if continuous else ("hourly_burden" if burden else None),
        "longest_s": round(longest, 1),
        "max_seizure_s_per_hour": round(hourly, 1),
        "hourly_threshold_s": frac * HOUR_S,
        "population": "neonate" if neonate else "adult_child",
    }
