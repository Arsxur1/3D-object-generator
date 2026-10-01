"""Bad-channel detection (TZ §5, PREP-style ideas, lightweight).

Flags flat channels, extreme-amplitude channels, and channels poorly correlated
with their neighbours. Interpolation is a hook (full spherical-spline
interpolation is provided by the optional MNE backend in v2).
"""

from __future__ import annotations

import numpy as np

from ..contracts.signal import UnifiedSignal
from ..layer1_ingest.electrodes import is_eeg_channel


def detect_bad_channels(
    sig: UnifiedSignal,
    *,
    flat_uv: float = 0.5,
    extreme_uv: float = 500.0,
    corr_threshold: float = 0.3,
) -> list[str]:
    """Return the names of channels judged bad.

    - flat: standard deviation below ``flat_uv`` microvolts
    - extreme: robust amplitude above ``extreme_uv`` microvolts
    - uncorrelated: max abs correlation with any other EEG channel below
      ``corr_threshold`` (isolated noise)
    """
    data = sig.signal.astype(np.float64)
    names = sig.channel_names
    eeg_idx = [i for i, n in enumerate(names) if is_eeg_channel(n)]
    bad: set[str] = set()

    for i in eeg_idx:
        sd = float(np.std(data[i]))
        if sd < flat_uv:
            bad.add(names[i])
        p99 = float(np.percentile(np.abs(data[i]), 99))
        if p99 > extreme_uv:
            bad.add(names[i])

    if len(eeg_idx) >= 3:
        sub = data[eeg_idx]
        # guard against zero-variance rows in corrcoef
        std = sub.std(axis=1)
        valid = std > 1e-9
        if valid.sum() >= 3:
            corr = np.corrcoef(sub[valid])
            np.fill_diagonal(corr, 0.0)
            max_corr = np.max(np.abs(corr), axis=1)
            valid_names = [names[eeg_idx[k]] for k in range(len(eeg_idx)) if valid[k]]
            for name, mc in zip(valid_names, max_corr):
                if mc < corr_threshold:
                    bad.add(name)

    return sorted(bad)
