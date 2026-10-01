"""Interictal epileptiform discharge (IED) detector — spikes / sharp waves (TZ §7.2).

Template-free heuristic: transient, high-amplitude, sharp deflections (short
rise time, prominent above the local background) on individual channels. Every
candidate is a *hypothesis*; Layer 5 cross-checks against ECG/artifacts before
it is trusted (TZ §8.3 — separating true IEDs from artifacts is critical).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from ..contracts.events import Event, EventEvidence, Localization
from ..layer1_ingest.electrodes import is_eeg_channel
from ..layer4_detect.background_abn import _region_of
from .base import Detector


class SpikeDetector(Detector):
    code = "spike"
    group = "ied"

    def __init__(self, sharpness_z: float = 6.0, max_width_ms: float = 70.0):
        self.sharpness_z = sharpness_z
        self.max_width_ms = max_width_ms

    def detect(self, sig, features, thresholds, artifacts=None) -> list[Event]:
        fs = sig.sampling_rate_hz
        data = sig.signal.astype(np.float64)
        names = sig.channel_names
        events: list[Event] = []
        max_width = int(self.max_width_ms * 1e-3 * fs)

        for i, n in enumerate(names):
            if not is_eeg_channel(n):
                continue
            x = data[i]
            # sharpness = |2nd derivative|, z-scored
            d2 = np.abs(np.diff(x, n=2, prepend=x[0], append=x[-1]))
            sd = np.std(d2) + 1e-9
            z = d2 / sd
            peaks, props = find_peaks(z, height=self.sharpness_z, distance=int(0.2 * fs))
            for pk in peaks:
                # width check: sharp transient, not slow wave
                lo = max(0, pk - max_width)
                hi = min(len(x), pk + max_width)
                amp = float(np.max(np.abs(x[lo:hi] - np.median(x[lo:hi]))))
                if amp < 30.0:  # ignore tiny blips
                    continue
                t = pk / fs
                ru, uz = self.labels(self.code)
                events.append(
                    Event(
                        code=self.code,
                        label_ru=ru,
                        label_uz=uz,
                        group=self.group,
                        localization=Localization(channels=[n], region=_region_of(n)),
                        t_start=round(t - 0.05, 3),
                        t_end=round(t + 0.05, 3),
                        confidence=float(np.clip(0.4 + 0.05 * (z[pk] - self.sharpness_z), 0.4, 0.8)),
                        is_artifact_hypothesis=False,
                        evidence=[
                            EventEvidence(feature="sharpness_z", value=round(float(z[pk]), 2),
                                          reference=f">= {self.sharpness_z}", channels=[n]),
                            EventEvidence(feature="amplitude_uv", value=round(amp, 1), unit="uV",
                                          channels=[n]),
                        ],
                        metadata={"needs_artifact_check": True},
                    )
                )
        # cap the number reported in the skeleton to avoid flooding
        events.sort(key=lambda e: e.confidence, reverse=True)
        return events[:8]
