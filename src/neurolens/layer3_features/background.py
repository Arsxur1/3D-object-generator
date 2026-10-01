"""Background characterization: posterior dominant rhythm, continuity (TZ §6)."""

from __future__ import annotations

import numpy as np

from .peak_utils import dominant_frequency
from .suppression import suppression_ratio


def posterior_dominant_rhythm(
    o1: np.ndarray, o2: np.ndarray, fs: float
) -> float:
    """Estimate PDR as the dominant 4-13 Hz frequency over occipital channels."""
    occ = (np.asarray(o1, dtype=np.float64) + np.asarray(o2, dtype=np.float64)) / 2.0
    return dominant_frequency(occ, fs, fmin=4.0, fmax=13.0)


def continuity_index(x: np.ndarray, fs: float, amp_thresh_uv: float = 10.0) -> float:
    """1 - suppression_ratio: fraction of time the trace is continuous (0..1)."""
    return float(1.0 - suppression_ratio(x, fs, amp_thresh_uv=amp_thresh_uv))
