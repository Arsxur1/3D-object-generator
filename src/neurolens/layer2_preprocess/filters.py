"""Filtering: highpass, lowpass, and mains notch (TZ §5).

Zero-phase (filtfilt) IIR filters on numpy arrays. Mains notch defaults to
50 Hz (Uzbekistan grid), switchable to 60 Hz via config.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from ..contracts.config import FilterConfig
from ..contracts.signal import UnifiedSignal


def _sos_filtfilt_bandpass(
    data: np.ndarray, fs: float, hp: float, lp: float, order: int
) -> np.ndarray:
    nyq = fs / 2.0
    lp_eff = min(lp, nyq * 0.99)
    b, a = butter(order, [hp / nyq, lp_eff / nyq], btype="bandpass")
    return filtfilt(b, a, data, axis=-1)


def apply_filters(sig: UnifiedSignal, cfg: FilterConfig) -> UnifiedSignal:
    """Return a copy of ``sig`` with band-pass + optional notch applied.

    Non-finite samples are zeroed before filtering to keep filtfilt stable.
    """
    fs = sig.sampling_rate_hz
    data = np.nan_to_num(sig.signal.astype(np.float64), copy=True)

    data = _sos_filtfilt_bandpass(
        data, fs, cfg.highpass_hz, cfg.lowpass_hz, cfg.filter_order
    )

    if cfg.notch_enabled and cfg.notch_hz < fs / 2.0:
        w0 = cfg.notch_hz / (fs / 2.0)
        b, a = iirnotch(w0, Q=30.0)
        data = filtfilt(b, a, data, axis=-1)

    flags = list(sig.quality.flags)
    flags.append(
        f"filtered:hp={cfg.highpass_hz}Hz,lp={cfg.lowpass_hz}Hz,"
        f"notch={cfg.notch_hz if cfg.notch_enabled else 'off'}Hz"
    )
    new = sig.with_signal(data.astype(np.float32))
    new.quality.flags = flags
    return new
