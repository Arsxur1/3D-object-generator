"""Layer 3 feature tests."""

from __future__ import annotations

import numpy as np

from neurolens.layer3_features.spectral import (
    band_powers,
    peak_alpha_frequency,
    sef95,
)
from neurolens.layer3_features.suppression import burst_suppression, suppression_ratio
from neurolens.layer3_features.asymmetry import asymmetry_index


def _sine(freq, fs=256.0, seconds=8.0, amp=30.0):
    t = np.arange(int(fs * seconds)) / fs
    return amp * np.sin(2 * np.pi * freq * t)


def test_band_power_peaks_in_correct_band():
    fs = 256.0
    x = _sine(10.0, fs)  # alpha
    bp = band_powers(x, fs, relative=True)
    assert bp["alpha"] == max(bp.values())
    assert bp["alpha"] > 0.6


def test_peak_alpha_frequency():
    fs = 256.0
    x = _sine(11.0, fs)
    assert abs(peak_alpha_frequency(x, fs) - 11.0) < 1.0


def test_sef95_monotone():
    fs = 256.0
    low = _sine(5.0, fs)
    high = _sine(25.0, fs)
    assert sef95(high, fs) > sef95(low, fs)


def test_suppression_on_flat_signal():
    fs = 256.0
    flat = np.zeros(int(fs * 10))
    assert suppression_ratio(flat, fs, amp_thresh_uv=10.0) > 0.9


def test_burst_suppression_detects_bursts():
    fs = 256.0
    n = int(fs * 12)
    t = np.arange(n) / fs
    # 1s burst every 3s
    burst = ((t % 3.0) < 1.0).astype(float)
    x = burst * 60 * np.sin(2 * np.pi * 8 * t)
    bs = burst_suppression(x, fs, amp_thresh_uv=10.0)
    assert bs.n_bursts >= 3
    assert 0.3 < bs.suppression_ratio < 0.9


def test_asymmetry_index_sign():
    assert asymmetry_index(10, 2) > 0  # left greater -> positive
    assert asymmetry_index(2, 10) < 0
    assert asymmetry_index(5, 5) == 0
