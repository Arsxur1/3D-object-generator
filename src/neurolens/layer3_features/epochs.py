"""Epoching helper shared by Layer-3 features and Layer-4 detectors."""

from __future__ import annotations

import numpy as np


def epoch_indices(n_samples: int, fs: float, epoch_s: float, overlap: float = 0.0):
    """Yield (start, stop) sample indices for consecutive (optionally overlapping) epochs."""
    step = int(round(epoch_s * fs * (1.0 - overlap)))
    win = int(round(epoch_s * fs))
    if step <= 0 or win <= 0:
        return
    start = 0
    while start + win <= n_samples:
        yield start, start + win
        start += step


def epoch_times(n_samples: int, fs: float, epoch_s: float, overlap: float = 0.0) -> np.ndarray:
    """Center time (seconds) of each epoch."""
    centers = []
    for a, b in epoch_indices(n_samples, fs, epoch_s, overlap):
        centers.append((a + b) / 2.0 / fs)
    return np.asarray(centers)
