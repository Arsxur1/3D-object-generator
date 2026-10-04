"""FeatureSet — the assembled Layer-3 output consumed by Layer-4 detectors.

Holds per-channel summary features, time-resolved (epoched) features used for
event detection, and specialized traces (aEEG, DSA, burst-suppression).
Computed with a single-FFT-per-epoch periodogram for speed and transparency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..contracts.config import FilterConfig
from ..contracts.signal import UnifiedSignal
from ..layer1_ingest.electrodes import is_eeg_channel
from .aeeg import AEEG, aeeg_envelope
from .asymmetry import alpha_delta_ratio, asymmetry_index
from .dsa import DSA, density_spectral_array
from .epochs import epoch_indices
from .spectral import BANDS, band_powers, peak_alpha_frequency, sef95, spectral_entropy
from .suppression import BurstSuppression, burst_suppression

# Homologous left/right channel pairs for asymmetry (classic 10-20).
HOMOLOGOUS_PAIRS = [
    ("Fp1", "Fp2"), ("F3", "F4"), ("C3", "C4"), ("P3", "P4"),
    ("O1", "O2"), ("F7", "F8"), ("T3", "T4"), ("T5", "T6"),
]


# Bump when epoch features change, so derived caches are recomputed.
FEATURE_VERSION = 2  # v2: + epoch line length and Teager energy (morphology)


@dataclass
class FeatureSet:
    fs: float
    duration_s: float
    eeg_channels: list[str]
    epoch_s: float
    epoch_times: np.ndarray                      # [n_epochs]
    epoch_relpow: dict[str, np.ndarray]          # band -> [n_epochs][n_ch]
    epoch_rms: np.ndarray                        # [n_epochs][n_ch]
    epoch_domfreq: np.ndarray                    # [n_epochs][n_ch]
    epoch_band_conc: np.ndarray                  # [n_epochs][n_ch] rhythmicity 0..1
    channel_summary: dict[str, dict[str, float]] # channel -> feature -> value
    global_relpow: dict[str, float]              # mean over EEG channels
    pdr_hz: float
    asymmetry: dict[str, float]                  # region/pair -> index
    global_asymmetry: float
    aeeg: Optional[AEEG]
    dsa: Optional[DSA]
    burst: BurstSuppression
    representative_channel: str
    extra: dict[str, Any] = field(default_factory=dict)

    def channel_index(self, name: str) -> int:
        return self.eeg_channels.index(name)


def _epoch_spectrum(seg: np.ndarray, fs: float):
    """Single-FFT periodogram of one epoch (Hann window)."""
    n = len(seg)
    w = np.hanning(n)
    xw = (seg - seg.mean()) * w
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = (np.abs(np.fft.rfft(xw)) ** 2) / (np.sum(w**2) + 1e-12)
    return freqs, psd


def _band_rel(freqs: np.ndarray, psd: np.ndarray) -> dict[str, float]:
    total = float(np.trapezoid(psd, freqs)) + 1e-12
    out = {}
    for name, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & (freqs < hi)
        out[name] = float(np.trapezoid(psd[m], freqs[m])) / total if m.any() else 0.0
    return out


def _dom_and_conc(freqs: np.ndarray, psd: np.ndarray, fmin=1.0, fmax=30.0):
    m = (freqs >= fmin) & (freqs <= fmax)
    if not m.any():
        return 0.0, 0.0
    f, p = freqs[m], psd[m]
    dom = float(f[int(np.argmax(p))])
    # concentration: power within +/-1 Hz of the peak divided by total in-band
    peak_band = (f >= dom - 1.0) & (f <= dom + 1.0)
    conc = float(np.sum(p[peak_band]) / (np.sum(p) + 1e-12))
    return dom, conc


def compute_features(
    sig: UnifiedSignal,
    cfg: FilterConfig | None = None,
    *,
    epoch_s: float = 2.0,
    overlap: float = 0.5,
) -> FeatureSet:
    fs = sig.sampling_rate_hz
    names = sig.channel_names
    eeg_idx = [i for i, n in enumerate(names) if is_eeg_channel(n)]
    eeg_names = [names[i] for i in eeg_idx]
    data = sig.signal.astype(np.float64)

    # --- epoched features ---
    spans = list(epoch_indices(sig.n_samples, fs, epoch_s, overlap))
    n_ep = len(spans)
    n_ch = len(eeg_idx)
    relpow = {b: np.zeros((n_ep, n_ch)) for b in BANDS}
    rms = np.zeros((n_ep, n_ch))
    domf = np.zeros((n_ep, n_ch))
    conc = np.zeros((n_ep, n_ch))
    linelen = np.zeros((n_ep, n_ch))   # morphology: mean |dx| per sample (µV)
    teager = np.zeros((n_ep, n_ch))    # morphology: mean Teager-Kaiser energy (µV²)
    times = np.zeros(n_ep)

    for ei, (s0, s1) in enumerate(spans):
        times[ei] = (s0 + s1) / 2.0 / fs
        for ci, chan in enumerate(eeg_idx):
            seg = data[chan, s0:s1]
            freqs, psd = _epoch_spectrum(seg, fs)
            br = _band_rel(freqs, psd)
            for b in BANDS:
                relpow[b][ei, ci] = br[b]
            rms[ei, ci] = float(np.sqrt(np.mean(seg**2)))
            d, c = _dom_and_conc(freqs, psd)
            domf[ei, ci] = d
            conc[ei, ci] = c
            if seg.size > 2:
                linelen[ei, ci] = float(np.mean(np.abs(np.diff(seg))))
                teager[ei, ci] = float(np.mean(seg[1:-1] ** 2 - seg[:-2] * seg[2:]))

    # --- per-channel summary ---
    summary: dict[str, dict[str, float]] = {}
    for ci, chan in enumerate(eeg_idx):
        x = data[chan]
        abs_bp = band_powers(x, fs, relative=False)
        rel_bp = band_powers(x, fs, relative=True)
        summary[eeg_names[ci]] = {
            **{f"abs_{k}": v for k, v in abs_bp.items()},
            **{f"rel_{k}": v for k, v in rel_bp.items()},
            "sef95": sef95(x, fs),
            "paf": peak_alpha_frequency(x, fs),
            "spectral_entropy": spectral_entropy(x, fs),
            "adr": alpha_delta_ratio(x, fs),
        }

    global_relpow = {
        b: float(np.mean([summary[n][f"rel_{b}"] for n in eeg_names])) for b in BANDS
    }

    # --- PDR from occipital channels ---
    from .background import posterior_dominant_rhythm

    def _chan(name: str) -> np.ndarray:
        return data[names.index(name)] if name in names else np.zeros(sig.n_samples)

    pdr = posterior_dominant_rhythm(_chan("O1"), _chan("O2"), fs)

    # --- asymmetry over homologous pairs (delta-band power based) ---
    asym: dict[str, float] = {}
    diffs = []
    for left, right in HOMOLOGOUS_PAIRS:
        if left in eeg_names and right in eeg_names:
            pl = summary[left]["abs_delta"] + summary[left]["abs_theta"]
            pr = summary[right]["abs_delta"] + summary[right]["abs_theta"]
            ai = asymmetry_index(pl, pr)
            asym[f"{left}/{right}"] = ai
            diffs.append(ai)
    global_asym = float(np.max(np.abs(diffs))) if diffs else 0.0

    # --- representative channel: central, else first EEG ---
    rep = next((c for c in ("C3", "C4", "Cz", "P3") if c in eeg_names), eeg_names[0] if eeg_names else "")
    rep_sig = _chan(rep) if rep else np.zeros(sig.n_samples)

    aeeg = aeeg_envelope(rep_sig, fs, channel=rep) if rep else None
    dsa = density_spectral_array(rep_sig, fs, channel=rep) if rep else None
    burst = burst_suppression(rep_sig, fs)

    return FeatureSet(
        fs=fs,
        duration_s=sig.duration_s or (sig.n_samples / fs),
        eeg_channels=eeg_names,
        epoch_s=epoch_s,
        epoch_times=times,
        epoch_relpow=relpow,
        epoch_rms=rms,
        epoch_domfreq=domf,
        epoch_band_conc=conc,
        channel_summary=summary,
        global_relpow=global_relpow,
        pdr_hz=pdr,
        asymmetry=asym,
        global_asymmetry=global_asym,
        aeeg=aeeg,
        dsa=dsa,
        burst=burst,
        representative_channel=rep,
        extra={"epoch_linelen": linelen, "epoch_teager": teager,
               "feature_version": FEATURE_VERSION},
    )
