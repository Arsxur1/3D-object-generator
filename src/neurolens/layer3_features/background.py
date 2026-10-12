"""Background characterization: posterior dominant rhythm, continuity (TZ §6)."""

from __future__ import annotations

import numpy as np

from .peak_utils import dominant_frequency, spectral_peak
from .suppression import suppression_ratio


def posterior_dominant_rhythm(
    o1: np.ndarray, o2: np.ndarray, fs: float
) -> float:
    """PDR = frequency of a genuine 4-13 Hz spectral peak over the occipital channels
    (>= 2x the 1/f background); 0.0 = not identified (see ``spectral_peak``)."""
    occ = (np.asarray(o1, dtype=np.float64) + np.asarray(o2, dtype=np.float64)) / 2.0
    return spectral_peak(occ, fs, fmin=4.0, fmax=13.0)


def continuity_index(x: np.ndarray, fs: float, amp_thresh_uv: float = 10.0) -> float:
    """1 - suppression_ratio: fraction of time the trace is continuous (0..1)."""
    return float(1.0 - suppression_ratio(x, fs, amp_thresh_uv=amp_thresh_uv))
