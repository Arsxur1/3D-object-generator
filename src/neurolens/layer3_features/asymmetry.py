"""Interhemispheric asymmetry and alpha-delta ratio (TZ §6)."""

from __future__ import annotations

import numpy as np

from .spectral import band_powers


def asymmetry_index(power_left: float, power_right: float) -> float:
    """Normalized asymmetry: (L - R) / (L + R), range -1..1.

    Positive -> left > right. Magnitude near 1 means strong lateralization.
    """
    denom = power_left + power_right
    if denom <= 0:
        return 0.0
    return float((power_left - power_right) / denom)


def alpha_delta_ratio(x: np.ndarray, fs: float) -> float:
    """Alpha/Delta ratio (ADR) — a background/ischemia trend marker."""
    bp = band_powers(x, fs, relative=False)
    return float(bp["alpha"] / (bp["delta"] + 1e-12))
