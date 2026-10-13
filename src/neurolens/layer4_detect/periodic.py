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
_EAR = {"A1", "A2", "M1", "M2"}
_OCULAR = {"FP1", "FP2", "F7", "F8"}


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

        # Discharges must be TIME-LOCKED across channels (a periodic discharge has a
        # field): peaks of >= 2 channels within +-PD_LOCK_S form one discharge, and
        # periodicity is judged on these consensus discharges - per hemisphere and
        # over the whole scalp. (Increment 17: per-channel runs of 6 sharp peaks gave
        # single-channel "LPDs" in 75% of normal routine EEGs.)
        per_ch: dict[str, np.ndarray] = {}
        for i, n in enumerate(names):
            if not is_eeg_channel(n) or n in ecg_channels:
                continue
            per_ch[n] = _discharge_times(data[i], fs)

        def consensus_run(chans: list[str]):
            disch = _consensus_discharges({c: per_ch[c] for c in chans if c in per_ch})
            if len(disch) < thr.pd_min_discharges:
                return None
            times = np.array([d[0] for d in disch])
            run = _longest_periodic_run(times, thr.pd_freq_hz_min, thr.pd_freq_hz_max)
            if run is None or run["n"] < thr.pd_min_discharges or run["cv"] > thr.pd_isi_cv_max:
                return None
            if hr is not None and abs(run["freq"] - hr) < 0.15:
                return None  # periodicity of the heart rate (ECG)
            in_run = [d for d in disch if run["t0"] - 1e-6 <= d[0] <= run["t1"] + 1e-6]
            counts: dict[str, int] = {}
            for _, members in in_run:
                for c in members:
                    counts[c] = counts.get(c, 0) + 1
            run["channels"] = sorted(c for c, k in counts.items() if k >= 0.5 * len(in_run))
            return run

        # ear/mastoid electrodes are reference sites (and carry the ECG field most
        # strongly), not scalp locations of a periodic pattern
        eeg = [n for n in names if is_eeg_channel(n) and n not in ecg_channels
               and n.upper() not in _EAR]
        whole = consensus_run(eeg)
        left_run = consensus_run([c for c in eeg if _side(c) in ("left", "midline")])
        right_run = consensus_run([c for c in eeg if _side(c) in ("right", "midline")])
        if whole is None and left_run is None and right_run is None:
            return []
        periodic: dict[str, dict] = {}
        if whole is not None:
            for c in whole["channels"]:
                periodic[c] = whole
        for run in (left_run, right_run):
            if run is not None:
                for c in run["channels"]:
                    periodic.setdefault(c, run)
        if len(periodic) < 2:
            return []
        span = max(r["t1"] for r in periodic.values()) - min(r["t0"] for r in periodic.values())
        if span < thr.pd_min_duration_s:
            return []  # ACNS: < 10 s is "very brief" - not reported as a periodic pattern
        for c, run in periodic.items():
            s0, s1 = int(run["t0"] * fs), int(run["t1"] * fs)
            seg = data[names.index(c)][s0:s1]
            periodic[c] = {**run, "amp": float(np.percentile(np.abs(seg - np.median(seg)), 95)) if seg.size else 0.0}

        n_eeg = sum(1 for n in names if is_eeg_channel(n) and n.upper() not in _EAR)
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
            if len(left if majority == "left" else right) < 2:
                return []  # a lateralized pattern needs >= 2 lateral (non-midline) channels
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

        # Increment 18: the two classic mimics of periodic discharges on routine EEG -
        # ECG (cardiac-rate, almost perfectly regular) and eye movements (maximal at
        # Fp/anterior-temporal electrodes) - make the pattern an artefact hypothesis.
        cv_mean = float(np.mean([periodic[c]["cv"] for c in chans]))
        dominant = max(chans, key=lambda c: periodic[c]["amp"])
        mimic = None
        if thr.pd_ecg_freq_hz[0] <= frequency <= thr.pd_ecg_freq_hz[1] and cv_mean < thr.pd_ecg_isi_cv_max:
            mimic = "ecg"
        elif dominant.upper() in _OCULAR:
            mimic = "ocular"

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
                    EventEvidence(feature="pd_dominant_channel_amp_uv",
                                  value=round(periodic[dominant]["amp"], 1), unit="uV",
                                  channels=[dominant]),
                ],
                metadata={"prevalence": prevalence, "iic": iic, "dominant_channel": dominant,
                          **({"artifact_hypothesis": mimic} if mimic else {})},
            )
        ]

    # -- rhythmic delta (GRDA/LRDA) --------------------------------------
    def _rhythmic_delta(self, sig, features, thr) -> list[Event]:
        """ACNS rhythmic delta: channels whose delta-band autocorrelation shows a
        repeating waveform (>= ``rda_autocorr_min`` at one period, 6 s windows) *at
        the same time*. GRDA: >= ``generalized_channel_fraction`` of channels with
        both hemispheres for >= ``grda_min_windows``; LRDA: >= 2 channels of one
        hemisphere (and at least twice the other side) for >= ``lrda_min_windows``.
        (Increment 17: the previous spectral test was met by ordinary 1/f
        background, and per-channel runs from different minutes were combined.)"""
        from ..layer3_features.rhythmicity import HOP_S, WIN_S, delta_rhythmicity_map

        fs = sig.sampling_rate_hz
        idx = [i for i, n in enumerate(sig.channel_names) if is_eeg_channel(n)]
        if len(idx) < 4:
            return []
        names = [sig.channel_names[i] for i in idx]
        t, freq, rho, _ = delta_rhythmicity_map(sig.signal[idx], fs)
        if t.size == 0:
            return []
        active = rho >= thr.rda_autocorr_min                         # (C, W)
        side = np.array([_side(n) for n in names])
        left, right = active[side == "left"], active[side == "right"]
        frac = active.mean(axis=0)
        gen = (frac >= thr.generalized_channel_fraction) & left.any(axis=0) & right.any(axis=0)
        nl, nr = left.sum(axis=0), right.sum(axis=0)
        lat = ~gen & (((nl >= 2) & (nl >= 2 * nr)) | ((nr >= 2) & (nr >= 2 * nl)))

        def longest(mask, min_w):
            best = None
            for a, b in _runs(mask):
                if b - a >= min_w and (best is None or b - a > best[1] - best[0]):
                    best = (a, b)
            return best

        for code, mask, min_w in (("grda", gen, thr.grda_min_windows), ("lrda", lat, thr.lrda_min_windows)):
            run = longest(mask, min_w)
            if run is None:
                continue
            a, b = run
            sub = active[:, a:b]
            involved = [n for n, k in zip(names, sub.mean(axis=1)) if k >= 0.5]
            if code == "lrda":
                left_n = int(nl[a:b].sum())
                lateral = "left" if left_n >= int(nr[a:b].sum()) else "right"
                involved = [n for n in involved if _side(n) == lateral] or involved
            else:
                lateral = "generalized"
            f = freq[:, a:b][sub]
            frequency = float(np.median(f)) if f.size else 0.0
            t0 = float(t[a] - WIN_S / 2)
            t1 = float(t[b - 1] + WIN_S / 2)
            ch_frac = len(involved) / len(names)
            ru, uz = self.labels(code)
            return [Event(
                code=code, label_ru=ru, label_uz=uz, group=self.group,
                localization=Localization(channels=sorted(involved),
                                          region="generalized" if code == "grda" else "hemispheric",
                                          lateralization=lateral),
                t_start=round(max(0.0, t0), 1), t_end=round(t1, 1),
                confidence=float(np.clip(0.5 + 0.3 * ch_frac, 0.5, 0.85)),
                acns=AcnsModifiers(frequency_hz=round(frequency, 2)),
                evidence=[
                    EventEvidence(feature="rda_frequency_hz", value=round(frequency, 2), unit="Hz",
                                  reference="0.5-4 Hz (delta)"),
                    EventEvidence(feature="rda_channel_fraction", value=round(ch_frac, 3)),
                    EventEvidence(feature="rda_autocorr_median",
                                  value=round(float(np.median(rho[:, a:b][sub])), 3),
                                  reference=f">= {thr.rda_autocorr_min}"),
                    EventEvidence(feature="rda_duration_s", value=round(t1 - t0, 1), unit="s",
                                  reference=f">= {WIN_S + HOP_S * (min_w - 1):.0f} s"),
                ],
            )]
        return []


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


PD_LOCK_S = 0.05


def _consensus_discharges(times_by_ch: dict[str, np.ndarray], lock_s: float = PD_LOCK_S,
                          min_channels: int = 2) -> list[tuple[float, list[str]]]:
    """Group per-channel peak times into discharges seen on >= ``min_channels``
    channels within +-``lock_s`` -> [(time, [channels])]."""
    pts = sorted((float(t), c) for c, ts in times_by_ch.items() for t in ts)
    out: list[tuple[float, list[str]]] = []
    i = 0
    while i < len(pts):
        j = i
        while j + 1 < len(pts) and pts[j + 1][0] - pts[i][0] <= 2 * lock_s:
            j += 1
        chans = sorted({c for _, c in pts[i:j + 1]})
        if len(chans) >= min_channels:
            out.append((float(np.median([t for t, _ in pts[i:j + 1]])), chans))
        i = j + 1
    return out


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
