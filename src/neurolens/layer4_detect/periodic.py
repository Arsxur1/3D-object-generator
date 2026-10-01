"""Periodic & rhythmic ACNS patterns (TZ §7.4): GPDs, LPDs, BIPDs, LRDA, GRDA.

Classic/qEEG heuristics (ML detectors are v2):

- Periodic discharges (PDs): quasi-periodic sharp transients recurring at a
  regular inter-discharge interval. Localization from the involved channels
  (generalized / lateralized / bilateral-independent). Modifiers: frequency,
  prevalence, plus-features (+F fast, +R rhythmic, +S sharp). Patterns in the
  1.5-2.5 Hz band or with +F are flagged on the ictal-interictal continuum.
- Rhythmic delta (RDA): sustained rhythmic 0.5-4 Hz activity (GRDA / LRDA),
  reusing the epoched Layer-3 features.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from ..contracts.events import AcnsModifiers, Event, EventEvidence, Localization
from ..layer1_ingest.electrodes import is_eeg_channel
from .base import Detector

_MIDLINE = {"Fz", "Cz", "Pz"}


def _side(channel: str) -> str:
    """Left (odd trailing digit) / right (even) / midline (z)."""
    c = channel.strip()
    if c in _MIDLINE:
        return "midline"
    if c and c[-1].isdigit():
        return "left" if int(c[-1]) % 2 == 1 else "right"
    return "midline"


def _prevalence(coverage: float) -> str:
    if coverage >= 0.9:
        return "continuous"
    if coverage >= 0.6:
        return "abundant"
    if coverage >= 0.3:
        return "frequent"
    if coverage >= 0.1:
        return "occasional"
    return "rare"


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


class PeriodicPatternDetector(Detector):
    code = "periodic"
    group = "periodic"

    def detect(self, sig, features, thresholds, artifacts=None) -> list[Event]:
        events: list[Event] = []
        events += self._periodic_discharges(sig, features, thresholds, artifacts)
        events += self._rhythmic_delta(sig, features, thresholds)
        return events

    # -- periodic discharges (GPD/LPD/BIPD) ------------------------------
    def _periodic_discharges(self, sig, features, thr, artifacts=None) -> list[Event]:
        fs = sig.sampling_rate_hz
        data = sig.signal.astype(np.float64)
        names = sig.channel_names
        duration = sig.duration_s or (sig.n_samples / fs)

        # ECG artifact can mimic periodic discharges — exclude contaminated
        # channels and reject a periodicity that matches the heart rate.
        ecg_channels = set(artifacts.ecg_contaminated_channels) if artifacts else set()
        hr = artifacts.heart_rate_hz if artifacts else None

        periodic: dict[str, dict] = {}
        for i, n in enumerate(names):
            if not is_eeg_channel(n) or n in ecg_channels:
                continue
            times = _discharge_times(data[i], fs)
            run = _longest_periodic_run(times, thr.pd_freq_hz_min, thr.pd_freq_hz_max)
            if run is None or run["n"] < thr.pd_min_discharges or run["cv"] > thr.pd_isi_cv_max:
                continue
            # reject periodicity that coincides with the heart rate (ECG)
            if hr is not None and abs(run["freq"] - hr) < 0.15:
                continue
            s0, s1 = int(run["t0"] * fs), int(run["t1"] * fs)
            seg = data[i][s0:s1]
            run["amp"] = float(np.percentile(np.abs(seg - np.median(seg)), 95)) if seg.size else 0.0
            periodic[n] = run

        if not periodic:
            return []

        # Amplitude prominence: keep channels where the discharge is genuinely
        # prominent (>= 60% of the max), so average-reference leakage of a focal
        # discharge into distant channels does not make it look generalized.
        max_amp = max(r["amp"] for r in periodic.values())
        dominant = max(periodic.values(), key=lambda r: r["amp"])
        periodic = {
            c: r for c, r in periodic.items()
            if r["amp"] >= 0.6 * max_amp
            and not (r["t1"] < dominant["t0"] or dominant["t1"] < r["t0"])  # overlaps in time
        }
        if not periodic:
            return []

        n_eeg = sum(1 for n in names if is_eeg_channel(n))
        chans = list(periodic)
        left = [c for c in chans if _side(c) == "left"]
        right = [c for c in chans if _side(c) == "right"]
        frac = len(chans) / max(1, n_eeg)

        both_sides = bool(left) and bool(right)
        if frac >= thr.generalized_channel_fraction and both_sides:
            code, region, lat = "gpds", "generalized", "generalized"
        elif len(left) >= 2 and len(right) >= 2:
            code, region, lat = "bipds", "bilateral", "bilateral_independent"
        else:
            majority = "left" if len(left) >= len(right) else "right"
            code, region, lat = "lpds", "hemispheric", majority

        freqs = [periodic[c]["freq"] for c in chans]
        frequency = float(np.median(freqs))
        t0 = min(periodic[c]["t0"] for c in chans)
        t1 = max(periodic[c]["t1"] for c in chans)
        coverage = float((t1 - t0) / max(duration, 1e-9))
        prevalence = _prevalence(coverage)

        # plus-features
        fast = float(np.mean([
            features.channel_summary[c]["rel_beta"] + features.channel_summary[c]["rel_gamma"]
            for c in chans if c in features.channel_summary
        ])) if chans else 0.0
        plus = ["+S"]  # periodic discharges are sharp by definition here
        if fast >= thr.pd_plus_fast_ratio:
            plus.append("+F")
        iic = (thr.iic_freq_hz_min <= frequency <= thr.iic_freq_hz_max) or ("+F" in plus)

        ru, uz = self.labels(code)
        return [
            Event(
                code=code,
                label_ru=ru,
                label_uz=uz,
                group=self.group,
                localization=Localization(channels=sorted(chans), region=region, lateralization=lat),
                t_start=round(float(t0), 1),
                t_end=round(float(t1), 1),
                confidence=float(np.clip(0.5 + 0.4 * frac, 0.5, 0.9)),
                acns=AcnsModifiers(
                    frequency_hz=round(frequency, 2), prevalence=prevalence,
                    plus_features=plus, iic=iic,
                ),
                evidence=[
                    EventEvidence(feature="pd_frequency_hz", value=round(frequency, 2), unit="Hz"),
                    EventEvidence(feature="pd_isi_cv",
                                  value=round(float(np.mean([periodic[c]["cv"] for c in chans])), 3),
                                  reference=f"<= {thr.pd_isi_cv_max}"),
                    EventEvidence(feature="pd_channel_fraction", value=round(frac, 3)),
                    EventEvidence(feature="fast_power_fraction", value=round(fast, 3),
                                  reference=f"+F if >= {thr.pd_plus_fast_ratio}"),
                ],
                metadata={"prevalence": prevalence, "iic": iic},
            )
        ]

    # -- rhythmic delta (GRDA/LRDA) --------------------------------------
    def _rhythmic_delta(self, sig, features, thr) -> list[Event]:
        times = features.epoch_times
        if times.size == 0:
            return []
        dt = float(np.median(np.diff(times))) if times.size > 1 else features.epoch_s
        min_epochs = max(2, int(round(thr.rda_min_duration_s / max(dt, 1e-6))))

        domf = features.epoch_domfreq
        conc = features.epoch_band_conc
        involved: dict[str, tuple[int, int, float]] = {}
        for ci, chan in enumerate(features.eeg_channels):
            in_delta = (domf[:, ci] >= 0.5) & (domf[:, ci] <= 4.0)
            active = in_delta & (conc[:, ci] >= thr.rda_rhythmicity)
            best = None
            for a, b in _runs(active):
                if (b - a) >= min_epochs:
                    if best is None or (b - a) > (best[1] - best[0]):
                        best = (a, b)
            if best is not None:
                a, b = best
                involved[chan] = (a, b, float(np.median(domf[a:b, ci])))

        if not involved:
            return []

        n_eeg = len(features.eeg_channels)
        chans = list(involved)
        frac = len(chans) / max(1, n_eeg)
        left = [c for c in chans if _side(c) == "left"]
        right = [c for c in chans if _side(c) == "right"]
        generalized = frac >= thr.generalized_channel_fraction and left and right
        code = "grda" if generalized else "lrda"
        lat = "generalized" if generalized else ("left" if len(left) >= len(right) else "right")

        a0 = min(v[0] for v in involved.values())
        b0 = max(v[1] for v in involved.values())
        frequency = float(np.median([v[2] for v in involved.values()]))
        ru, uz = self.labels(code)
        return [
            Event(
                code=code,
                label_ru=ru,
                label_uz=uz,
                group=self.group,
                localization=Localization(channels=sorted(chans),
                                          region="generalized" if generalized else "hemispheric",
                                          lateralization=lat),
                t_start=round(float(times[a0]), 1),
                t_end=round(float(times[min(b0, len(times)) - 1]), 1),
                confidence=float(np.clip(0.5 + 0.3 * frac, 0.5, 0.85)),
                acns=AcnsModifiers(frequency_hz=round(frequency, 2)),
                evidence=[
                    EventEvidence(feature="rda_frequency_hz", value=round(frequency, 2), unit="Hz",
                                  reference="0.5-4 Hz (delta)"),
                    EventEvidence(feature="rda_channel_fraction", value=round(frac, 3)),
                ],
            )
        ]


def _discharge_times(x: np.ndarray, fs: float, z_thr: float = 6.0, min_dist_s: float = 0.25) -> np.ndarray:
    """Times (s) of SHARP discharge peaks on one channel.

    Uses a high sharpness (2nd-derivative z-score) threshold so smooth rhythmic
    waves (rhythmic delta) are not counted as periodic discharges — the clinical
    PD-vs-RDA distinction.
    """
    d2 = np.abs(np.diff(x, n=2, prepend=x[0], append=x[-1]))
    sd = np.std(d2) + 1e-9
    z = d2 / sd
    peaks, _ = find_peaks(z, height=z_thr, distance=max(1, int(min_dist_s * fs)))
    return peaks / fs


def _longest_periodic_run(times: np.ndarray, fmin: float, fmax: float, tol: float = 0.3):
    """Longest run of consecutive discharges with a consistent inter-discharge interval.

    Localizes periodic discharges to the segment where they actually recur
    regularly (rather than requiring global periodicity over the whole record).
    Returns {"freq","t0","t1","n","cv"} or None.
    """
    if len(times) < 3:
        return None
    isi = np.diff(times)
    N = len(isi)
    best = None
    i = 0
    while i < N:
        j = i
        acc = [isi[i]]
        while j + 1 < N:
            med = float(np.median(acc))
            if abs(isi[j + 1] - med) <= tol * med:
                acc.append(isi[j + 1])
                j += 1
            else:
                break
        n_disch = (j - i) + 2
        med = float(np.median(acc))
        freq = 1.0 / med if med > 0 else 0.0
        cv = float(np.std(acc) / (np.mean(acc) + 1e-9))
        if fmin <= freq <= fmax and (best is None or n_disch > best["n"]):
            best = {"freq": freq, "t0": float(times[i]), "t1": float(times[j + 1]),
                    "n": n_disch, "cv": cv}
        i = j + 1
    return best
