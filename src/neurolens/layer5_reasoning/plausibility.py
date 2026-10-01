"""Plausibility & artifact/variant cross-check (TZ §8.3, §16).

Separating true IEDs from artifacts AND from benign variants is critical.
A spike candidate is resolved as:
  - ECG artifact — coincides with QRS or sits on an ECG-contaminated channel;
  - benign variant (wicket) — coincides in time+channel with a detected wicket;
otherwise it stays a genuine epileptiform candidate. Events that nothing
explains are flagged implausible upstream.
"""

from __future__ import annotations

from ..contracts.events import DetectionResult, Event
from ..layer2_preprocess.artifacts import ArtifactReport


def _near_qrs(event: Event, qrs_times: list[float], tol: float = 0.06) -> bool:
    center = 0.5 * (event.t_start + event.t_end)
    return any(abs(center - q) <= tol for q in qrs_times)


def _overlaps_time(ev: Event, other: Event) -> bool:
    return not (ev.t_end < other.t_start or other.t_end < ev.t_start)


def check_plausibility(
    detection: DetectionResult, artifacts: ArtifactReport | None
) -> dict[str, dict]:
    """Resolve spike candidates. Returns {"spike_resolutions": {id(ev): {...}}}.

    Each resolution: {"kind": "artifact"|"variant", "reason": str}.
    """
    resolutions: dict[int, dict] = {}
    spikes = detection.by_group("ied")

    # 1) ECG-artifact resolution
    if artifacts is not None and (artifacts.ecg_present or artifacts.ecg_contaminated_channels):
        qrs = artifacts.ecg_qrs_times_s
        contaminated = set(artifacts.ecg_contaminated_channels)
        for ev in spikes:
            on_contaminated = bool(set(ev.localization.channels) & contaminated)
            coincident = _near_qrs(ev, qrs)
            if coincident or on_contaminated:
                reason = []
                if coincident:
                    reason.append("совпадение с QRS ЭКГ")
                if on_contaminated:
                    reason.append("на ЭКГ-контаминированном канале")
                resolutions[id(ev)] = {"kind": "artifact", "reason": "; ".join(reason)}

    # 2) benign-variant (wicket) resolution — only for still-unresolved spikes
    wickets = detection.by_group("variant")
    if wickets:
        for ev in spikes:
            if id(ev) in resolutions:
                continue
            ev_chans = set(ev.localization.channels)
            for w in wickets:
                if ev_chans & set(w.localization.channels) and _overlaps_time(ev, w):
                    resolutions[id(ev)] = {
                        "kind": "variant",
                        "reason": "совпадает с доброкачественным вариантом (wicket)",
                    }
                    break

    return {"spike_resolutions": resolutions}
