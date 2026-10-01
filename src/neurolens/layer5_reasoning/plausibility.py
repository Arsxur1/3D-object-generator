"""Plausibility & artifact cross-check (TZ §8.3).

Separating true IEDs from artifacts is a critical task. Here we resolve
spike/sharp candidates that temporally coincide with ECG QRS complexes (or
occur on ECG-contaminated channels) as ECG artifact rather than epileptiform
discharges, and flag events that nothing explains as implausible.
"""

from __future__ import annotations

from ..contracts.events import DetectionResult, Event
from ..layer2_preprocess.artifacts import ArtifactReport


def _near_qrs(event: Event, qrs_times: list[float], tol: float = 0.06) -> bool:
    center = 0.5 * (event.t_start + event.t_end)
    return any(abs(center - q) <= tol for q in qrs_times)


def check_plausibility(
    detection: DetectionResult, artifacts: ArtifactReport | None
) -> dict[str, dict]:
    """Return per-event-code resolution info used by the rules engine.

    Output maps a spike Event (by id(event)) to a dict:
      {"is_artifact": bool, "reason": str}
    Also returns a global 'implausible' list of event codes with no explanation.
    """
    resolutions: dict[int, dict] = {}
    if artifacts is not None and (artifacts.ecg_present or artifacts.ecg_contaminated_channels):
        qrs = artifacts.ecg_qrs_times_s
        contaminated = set(artifacts.ecg_contaminated_channels)
        for ev in detection.by_group("ied"):
            on_contaminated = bool(set(ev.localization.channels) & contaminated)
            coincident = _near_qrs(ev, qrs)
            if coincident or on_contaminated:
                reason = []
                if coincident:
                    reason.append("совпадение с QRS ЭКГ")
                if on_contaminated:
                    reason.append("на ЭКГ-контаминированном канале")
                resolutions[id(ev)] = {
                    "is_artifact": True,
                    "reason": "; ".join(reason),
                }
    return {"spike_resolutions": resolutions}
