"""Spectral features (TZ §6): band powers, SEF95, peak alpha, spectral entropy.

Bands (Hz): delta 0.5-4, theta 4-8, alpha 8-13, beta 13-30, gamma >30.
All functions take a 1-D signal and sampling rate; power via Welch PSD.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _psd(x: np.ndarray, fs: float):
    nper = int(min(len(x), max(64, fs * 2)))
    f, p = welch(x, fs=fs, nperseg=nper)
    return f, p


def band_powers(x: np.ndarray, fs: float, relative: bool = False) -> dict[str, float]:
    """Absolute (uV^2) or relative band powers for one channel."""
    f, p = _psd(x, fs)
    total = float(np.trapezoid(p, f)) + 1e-12
    out: dict[str, float] = {}
    for name, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        bp = float(np.trapezoid(p[m], f[m])) if m.any() else 0.0
        out[name] = bp / total if relative else bp
    return out


def sef95(x: np.ndarray, fs: float, edge: float = 0.95) -> float:
    """Spectral edge frequency: frequency below which ``edge`` of power lies."""
    f, p = _psd(x, fs)
    csum = np.cumsum(p)
    if csum[-1] <= 0:
        return 0.0
    csum = csum / csum[-1]
    idx = int(np.searchsorted(csum, edge))
    idx = min(idx, len(f) - 1)
    return float(f[idx])


def peak_alpha_frequency(x: np.ndarray, fs: float) -> float:
    """Frequency of maximum PSD within the alpha band (0 if none)."""
    f, p = _psd(x, fs)
    lo, hi = BANDS["alpha"]
    m = (f >= lo) & (f <= hi)
    if not m.any():
        return 0.0
    band_f, band_p = f[m], p[m]
    return float(band_f[int(np.argmax(band_p))])


def spectral_entropy(x: np.ndarray, fs: float) -> float:
    """Normalized Shannon entropy of the PSD (0..1)."""
    f, p = _psd(x, fs)
    p = p[f <= 45.0]
    p = p / (np.sum(p) + 1e-12)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    ent = -np.sum(p * np.log2(p))
    return float(ent / np.log2(p.size)) if p.size > 1 else 0.0
