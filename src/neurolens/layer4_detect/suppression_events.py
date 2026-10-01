"""Burst-suppression / suppression detector (TZ §7.5, §10).

Localizes burst-suppression episodes (rather than reporting a global ratio) by
scanning the representative channel for contiguous regions of high suppressed
fraction. Burst-suppression is a hard-safety pattern (TZ §10): severe injury vs
deep sedation/hypothermia is disambiguated by Layer 5 using clinical context.
"""

from __future__ import annotations

import numpy as np

from ..contracts.events import Event, EventEvidence, Localization
from ..layer1_ingest.electrodes import is_eeg_channel
from ..layer3_features.suppression import burst_suppression, suppression_regions_multichannel
from .base import Detector


class BurstSuppressionDetector(Detector):
    code = "burst_suppression"
    group = "suppression"

    def detect(self, sig, features, thresholds, artifacts=None) -> list[Event]:
        fs = sig.sampling_rate_hz
        eeg_idx = [i for i, n in enumerate(sig.channel_names) if is_eeg_channel(n)]
        if not eeg_idx:
            return []
        eeg_data = sig.signal[eeg_idx].astype(np.float64)

        # Generalized burst-suppression: suppression must be across channels.
        regions = suppression_regions_multichannel(
            eeg_data, fs,
            amp_thresh_uv=thresholds.suppression_amplitude_uv,
            frac_thresh=thresholds.burst_suppression_ratio,
        )
        if not regions:
            return []
        # take the longest episode; quantify on the mean of channels within it
        t0, t1 = max(regions, key=lambda r: r[1] - r[0])
        seg = eeg_data[:, int(t0 * fs):int(t1 * fs)].mean(axis=0)
        bs = burst_suppression(seg, fs, amp_thresh_uv=thresholds.suppression_amplitude_uv)
        rep = features.representative_channel

        ru, uz = self.labels(self.code)
        conf = float(np.clip(bs.suppression_ratio, 0.5, 0.95))
        return [
            Event(
                code=self.code,
                label_ru=ru,
                label_uz=uz,
                group=self.group,
                localization=Localization(region="generalized", lateralization="generalized"),
                t_start=round(t0, 1),
                t_end=round(t1, 1),
                confidence=conf,
                evidence=[
                    EventEvidence(feature="suppression_ratio",
                                  value=round(bs.suppression_ratio, 3),
                                  reference=f">= {thresholds.burst_suppression_ratio} (в эпизоде)",
                                  channels=[rep]),
                    EventEvidence(feature="n_bursts", value=float(bs.n_bursts)),
                    EventEvidence(feature="mean_ibi_s", value=round(bs.mean_ibi_s, 2), unit="s"),
                ],
                metadata={"channel": rep, "episode_s": round(t1 - t0, 1)},
            )
        ]
