"""Special patterns — triphasic waves (TZ §7.6).

Triphasic waves: generalized, frontally-predominant ~1.5-2.5 Hz periodic waves,
classically associated with metabolic encephalopathy. MVP heuristic: sustained
frontal dominant frequency in the triphasic band with high delta power. FIRDA,
extreme delta brush, etc. are added in v2.
"""

from __future__ import annotations

import numpy as np

from ..contracts.events import Event, EventEvidence, Localization
from .base import Detector

_FRONTAL = ["Fp1", "Fp2", "F3", "F4", "Fz", "F7", "F8"]
_TRIPHASIC_BAND = (1.3, 2.7)


class TriphasicWaveDetector(Detector):
    code = "triphasic_waves"
    group = "special"

    def detect(self, sig, features, thresholds, artifacts=None) -> list[Event]:
        frontal_idx = [features.eeg_channels.index(c) for c in _FRONTAL if c in features.eeg_channels]
        if len(frontal_idx) < 3 or features.epoch_times.size == 0:
            return []

        domf = features.epoch_domfreq[:, frontal_idx]   # [n_ep][n_frontal]
        delta = features.epoch_relpow["delta"][:, frontal_idx]
        lo, hi = _TRIPHASIC_BAND
        in_band = (domf >= lo) & (domf <= hi)
        # epoch qualifies if most frontal channels are in the triphasic band with high delta
        epoch_ok = (in_band.mean(axis=1) >= 0.6) & (delta.mean(axis=1) >= 0.5)
        frac = float(epoch_ok.mean())
        if frac < 0.3:
            return []

        times = features.epoch_times
        t0 = float(times[epoch_ok][0])
        t1 = float(times[epoch_ok][-1])
        mean_freq = float(np.mean(domf[in_band])) if in_band.any() else float(np.mean(domf))
        ru, uz = self.labels(self.code)
        return [
            Event(
                code=self.code,
                label_ru=ru,
                label_uz=uz,
                group=self.group,
                localization=Localization(
                    channels=_FRONTAL, region="frontal", lateralization="generalized"
                ),
                t_start=t0,
                t_end=t1,
                confidence=float(np.clip(0.4 + frac, 0.4, 0.85)),
                evidence=[
                    EventEvidence(feature="frontal_dominant_freq_hz",
                                  value=round(mean_freq, 2), unit="Hz",
                                  reference=f"{lo}-{hi} Hz (triphasic band)"),
                    EventEvidence(feature="frontal_rel_delta",
                                  value=round(float(delta.mean()), 3), reference=">= 0.5"),
                    EventEvidence(feature="triphasic_epoch_fraction", value=round(frac, 3)),
                ],
            )
        ]
