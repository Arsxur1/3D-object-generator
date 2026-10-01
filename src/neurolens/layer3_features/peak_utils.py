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
