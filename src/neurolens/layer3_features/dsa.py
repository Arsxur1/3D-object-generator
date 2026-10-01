"""Density Spectral Array (DSA) / spectrogram trend for cEEG (TZ §6, §11.4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import spectrogram


@dataclass
class DSA:
    times_s: np.ndarray       # [n_windows]
    freqs_hz: np.ndarray      # [n_freqs]
    power_db: np.ndarray      # [n_freqs][n_windows], 10*log10 power
    channel: str


def density_spectral_array(
    x: np.ndarray, fs: float, channel: str = "", window_s: float = 4.0, fmax: float = 30.0
) -> DSA:
    nper = int(window_s * fs)
    nover = nper // 2
    f, t, Sxx = spectrogram(x, fs=fs, nperseg=nper, noverlap=nover)
    m = f <= fmax
    power_db = 10.0 * np.log10(Sxx[m] + 1e-12)
    return DSA(times_s=t, freqs_hz=f[m], power_db=power_db, channel=channel)
