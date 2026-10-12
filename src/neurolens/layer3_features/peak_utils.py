"""Small spectral-peak helpers shared across Layer-3 modules."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch


def dominant_frequency(x: np.ndarray, fs: float, fmin: float = 0.5, fmax: float = 45.0) -> float:
    """Frequency of maximum PSD within [fmin, fmax] (0 if flat/empty)."""
    x = np.asarray(x, dtype=np.float64)
    nper = int(min(len(x), max(64, fs * 2)))
    if nper < 16:
        return 0.0
    f, p = welch(x, fs=fs, nperseg=nper)
    m = (f >= fmin) & (f <= fmax)
    if not m.any() or np.all(p[m] <= 0):
        return 0.0
    fm, pm = f[m], p[m]
    return float(fm[int(np.argmax(pm))])


def spectral_peak(x: np.ndarray, fs: float, fmin: float = 4.0, fmax: float = 13.0,
                  min_ratio: float = 2.0) -> float:
    """Frequency of a genuine spectral peak in [fmin, fmax]: the spectrum must exceed a
    power-law (1/f) fit of the flanking background (1-``fmin`` and 15-30 Hz) by at least
    ``min_ratio``. Returns 0.0 when there is no such peak (increment 17: the plain
    maximum of a 1/f spectrum is its lowest frequency, which reported a 4 Hz "posterior
    dominant rhythm" - and hence slowing - in normal recordings without visible alpha)."""
    x = np.asarray(x, dtype=np.float64)
    nper = int(min(len(x), max(64, fs * 4)))
    if nper < 16:
        return 0.0
    f, p = welch(x, fs=fs, nperseg=nper)
    fit = ((f >= 1.0) & (f < fmin)) | ((f >= 15.0) & (f <= 30.0))
    band = (f >= fmin) & (f <= fmax)
    if fit.sum() < 3 or not band.any() or np.all(p[band] <= 0):
        return 0.0
    lf, lp = np.log10(f[fit]), np.log10(p[fit] + 1e-20)
    slope, icpt = np.polyfit(lf, lp, 1)
    resid = np.log10(p[band] + 1e-20) - (slope * np.log10(f[band]) + icpt)
    k = int(np.argmax(resid))
    if resid[k] < np.log10(min_ratio):
        return 0.0
    return float(f[band][k])
