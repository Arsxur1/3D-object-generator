"""Benign variant detection (TZ §8.2, §16).

Benign variants are frequently misread as epileptiform. Here we detect
**wicket spikes/rhythms**: temporal arciform ~6-11 Hz rhythms. Emitted as a
``variant`` event so Layer 5 can down-weight spike candidates that coincide
with them (improving IED specificity). Age-dependent variants (posterior slow
waves of youth, hypnagogic hypersynchrony) are handled in Layer-5 physiology.
"""

from __future__ import annotations

import numpy as np

from ..contracts.events import Event, EventEvidence, Localization
from .base import Detector

# Mid/anterior-temporal chain where wicket is typically seen.
_TEMPORAL = ["T3", "T4", "T5", "T6", "F7", "F8"]
_WICKET_BAND = (6.0, 11.0)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.size == 0:
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    stops = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        stops = stops + [mask.size]
    return list(zip(starts, stops))


class BenignVariantDetector(Detector):
    code = "wicket"
    group = "variant"

    def detect(self, sig, features, thresholds, artifacts=None) -> list[Event]:
        temporal = [c for c in _TEMPORAL if c in features.eeg_channels]
        if len(temporal) < 1 or features.epoch_times.size == 0:
            return []
        idx = [features.eeg_channels.index(c) for c in temporal]
        domf = features.epoch_domfreq[:, idx]
        conc = features.epoch_band_conc[:, idx]
        lo, hi = _WICKET_BAND

        events: list[Event] = []
        times = features.epoch_times
        dt = float(np.median(np.diff(times))) if times.size > 1 else features.epoch_s
        min_epochs = max(2, int(round(3.0 / max(dt, 1e-6))))  # sustained >= ~3 s
        for k, ci in enumerate(idx):
            active = (domf[:, k] >= lo) & (domf[:, k] <= hi) & (conc[:, k] >= thresholds.rda_rhythmicity)
            best = None
            for a, b in _runs(active):
                if (b - a) >= min_epochs and (best is None or (b - a) > (best[1] - best[0])):
                    best = (a, b)
            if best is None:
                continue
            a, b = best
            chan = temporal[k]
            t0 = float(times[a])
            t1 = float(times[min(b, len(times)) - 1])
            mean_freq = float(np.median(domf[a:b, k]))
            active_frac = (b - a) / len(times)
            ru, uz = self.labels(self.code)
            events.append(
                Event(
                    code=self.code,
                    label_ru=ru,
                    label_uz=uz,
                    group=self.group,
                    localization=Localization(channels=[chan], region="temporal"),
                    t_start=round(t0, 1),
                    t_end=round(t1, 1),
                    confidence=float(np.clip(0.4 + active_frac, 0.4, 0.8)),
                    evidence=[
                        EventEvidence(feature="wicket_freq_hz", value=round(mean_freq, 2),
                                      unit="Hz", reference=f"{lo}-{hi} Hz temporal arciform",
                                      channels=[chan]),
                    ],
                    metadata={"variant": "wicket"},
                )
            )
        return events
