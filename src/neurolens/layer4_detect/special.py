"""Special patterns (TZ §7.6): triphasic waves, FIRDA, extreme delta brush.

- Triphasic waves: generalized, frontally-predominant ~1.5-2.5 Hz periodic waves
  (classically metabolic encephalopathy).
- FIRDA: frontal INTERMITTENT rhythmic delta (runs with gaps, not continuous).
- Extreme delta brush: delta waves with superimposed fast (beta) "brushes"
  riding on the delta crests (e.g. anti-NMDA-R encephalitis; extreme prematurity).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from ..contracts.events import AcnsModifiers, Event, EventEvidence, Localization
from ..layer1_ingest.electrodes import is_eeg_channel
from .base import Detector

_FRONTAL = ["Fp1", "Fp2", "F3", "F4", "Fz", "F7", "F8"]
_TRIPHASIC_BAND = (1.3, 2.7)


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


class FirdaDetector(Detector):
    """Frontal intermittent rhythmic delta activity (TZ §7.6).

    Frontal rhythmic ~1.5-3 Hz delta that comes in runs with gaps (intermittent),
    distinguishing it from continuous GRDA.
    """

    code = "firda"
    group = "special"

    def detect(self, sig, features, thresholds, artifacts=None) -> list[Event]:
        frontal_idx = [
            features.eeg_channels.index(c) for c in _FRONTAL if c in features.eeg_channels
        ]
        if len(frontal_idx) < 3 or features.epoch_times.size == 0:
            return []

        domf = features.epoch_domfreq[:, frontal_idx]
        conc = features.epoch_band_conc[:, frontal_idx]
        active = ((domf >= 1.5) & (domf <= 3.0) & (conc >= thresholds.rda_rhythmicity)).mean(axis=1) >= 0.5
        if not active.any():
            return []

        runs = _runs(active)
        coverage = float(active.mean())
        n_runs = len(runs)
        # must be INTERMITTENT: multiple runs, not continuously present
        if n_runs < thresholds.firda_min_runs or coverage >= thresholds.firda_max_continuous_fraction:
            return []

        times = features.epoch_times
        t0 = float(times[active][0])
        t1 = float(times[active][-1])
        mean_freq = float(np.median(domf[(domf >= 1.5) & (domf <= 3.0)])) if (domf >= 1.5).any() else 2.0
        ru, uz = self.labels(self.code)
        return [
            Event(
                code=self.code,
                label_ru=ru,
                label_uz=uz,
                group=self.group,
                localization=Localization(channels=_FRONTAL, region="frontal",
                                          lateralization="generalized"),
                t_start=round(t0, 1),
                t_end=round(t1, 1),
                confidence=float(np.clip(0.45 + 0.3 * coverage, 0.45, 0.8)),
                acns=AcnsModifiers(frequency_hz=round(mean_freq, 2)),
                evidence=[
                    EventEvidence(feature="firda_freq_hz", value=round(mean_freq, 2), unit="Hz",
                                  reference="1.5-3 Hz frontal"),
                    EventEvidence(feature="firda_n_runs", value=float(n_runs),
                                  reference=f">= {thresholds.firda_min_runs} (intermittent)"),
                    EventEvidence(feature="firda_coverage", value=round(coverage, 3),
                                  reference=f"< {thresholds.firda_max_continuous_fraction}"),
                ],
            )
        ]


class ExtremeDeltaBrushDetector(Detector):
    """Extreme delta brush (TZ §7.6): fast (beta) 'brushes' riding on delta.

    Heuristic: channels with high delta power where the beta-band amplitude
    envelope is correlated with the delta envelope (fast rides on delta crests)
    and beta power fraction exceeds a threshold.
    """

    code = "extreme_delta_brush"
    group = "special"

    def detect(self, sig, features, thresholds, artifacts=None) -> list[Event]:
        fs = sig.sampling_rate_hz
        if fs <= 70 or features.epoch_times.size == 0:
            return []
        data = sig.signal.astype(np.float64)
        times = features.epoch_times
        dt = float(np.median(np.diff(times))) if times.size > 1 else features.epoch_s
        min_epochs = max(2, int(round(thresholds.rda_min_duration_s / max(dt, 1e-6))))
        delta_rp = features.epoch_relpow["delta"]
        beta_rp = features.epoch_relpow["beta"]

        hits: list[str] = []
        corrs: list[float] = []
        span_a, span_b = None, None
        for ci, chan in enumerate(features.eeg_channels):
            active = (delta_rp[:, ci] >= 0.35) & (beta_rp[:, ci] >= thresholds.edb_beta_ratio)
            best = None
            for a, b in _runs(active):
                if (b - a) >= min_epochs and (best is None or (b - a) > (best[1] - best[0])):
                    best = (a, b)
            if best is None:
                continue
            a, b = best
            s0 = int(times[a] * fs)
            s1 = int(times[min(b, len(times)) - 1] * fs)
            seg = data[sig.channel_names.index(chan), s0:s1]
            if len(seg) < fs:
                continue
            # brushes ride on delta crests: beta amplitude tracks the delta
            # WAVEFORM (not its envelope, which is flat for a steady rhythm)
            r = float(np.corrcoef(_band_signal(seg, fs, 1.0, 4.0), _band_env(seg, fs, 18.0, 30.0))[0, 1])
            if r >= 0.3:
                hits.append(chan)
                corrs.append(r)
                span_a = a if span_a is None else min(span_a, a)
                span_b = b if span_b is None else max(span_b, b)

        if len(hits) < 2:
            return []
        ru, uz = self.labels(self.code)
        return [
            Event(
                code=self.code,
                label_ru=ru,
                label_uz=uz,
                group=self.group,
                localization=Localization(channels=sorted(hits), region="widespread"),
                t_start=round(float(times[span_a]), 1),
                t_end=round(float(times[min(span_b, len(times)) - 1]), 1),
                confidence=float(np.clip(0.4 + 0.1 * len(hits), 0.4, 0.8)),
                evidence=[
                    EventEvidence(feature="edb_delta_beta_corr",
                                  value=round(float(np.mean(corrs)), 3), reference=">= 0.3",
                                  channels=hits[:8]),
                    EventEvidence(feature="edb_channels", value=float(len(hits))),
                ],
            )
        ]


def _band_signal(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(2, [lo / nyq, min(hi, nyq * 0.99) / nyq], btype="bandpass")
    return filtfilt(b, a, x)


def _band_env(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    return np.abs(hilbert(_band_signal(x, fs, lo, hi)))
