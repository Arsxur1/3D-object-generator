"""Signal Quality Index per channel (TZ §5).

A pragmatic 0..1 SQI combining: amplitude plausibility (EEG typically within a
few hundred microvolts), line-noise contamination, and flatness. Feeds mode-B
gating (TZ §2) and the artifact-fraction estimate.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from ..contracts.signal import UnifiedSignal
from ..layer1_ingest.electrodes import is_eeg_channel


def _line_noise_ratio(x: np.ndarray, fs: float, mains: float) -> float:
    if fs <= 2 * mains:
        return 0.0
    nper = int(min(len(x), fs * 2))
    if nper < 16:
        return 0.0
    f, p = welch(x, fs=fs, nperseg=nper)
    total = np.trapezoid(p, f) + 1e-12
    band = (f >= mains - 1) & (f <= mains + 1)
    return float(np.trapezoid(p[band], f[band]) / total)


def compute_sqi(sig: UnifiedSignal, mains_hz: float = 50.0) -> list[float]:
    """Return an SQI in [0, 1] for every channel (non-EEG channels get 1.0)."""
    data = sig.signal.astype(np.float64)
    fs = sig.sampling_rate_hz
    out: list[float] = []
    for i, name in enumerate(sig.channel_names):
        if not is_eeg_channel(name):
            out.append(1.0)
            continue
        x = data[i]
        sd = float(np.std(x))
        # amplitude score: penalize flat (<1uV) and huge (>300uV) channels
        if sd < 1.0:
            amp = 0.0
        elif sd > 300.0:
            amp = 0.2
        else:
            amp = 1.0
        p99 = float(np.percentile(np.abs(x), 99))
        clip = 1.0 if p99 < 500 else 0.3
        line = _line_noise_ratio(x, fs, mains_hz)
        line_score = max(0.0, 1.0 - 5.0 * line)  # heavy penalty when line noise dominates
        sqi = float(np.clip(0.5 * amp + 0.25 * clip + 0.25 * line_score, 0.0, 1.0))
        out.append(sqi)
    return out


def overall_quality(sqi: list[float], channel_names: list[str]) -> float:
    """Mean SQI over EEG channels (used by mode-B gating)."""
    vals = [s for s, n in zip(sqi, channel_names) if is_eeg_channel(n)]
    return float(np.mean(vals)) if vals else 0.0
