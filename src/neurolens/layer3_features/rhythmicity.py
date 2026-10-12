"""Rhythmicity of delta activity (increment 17).

ACNS 2021: *rhythmic* = repetition of a waveform with relatively uniform
morphology and duration and no interval between consecutive waveforms, for at
least 6 cycles. The legacy measure (dominant 1-30 Hz frequency in delta and >= 40%
of the power within +-1 Hz of it) is met by ordinary 1/f background: with no alpha
peak the spectrum's maximum is simply its lowest frequency and 1-2 Hz holds about
half the power. On routine EEG it reported GRDA in 87% of normal recordings.

Here rhythmicity is measured in the time domain, which is what the definition
describes: the signal is band-passed to 0.5-4 Hz, and in each 6 s window (>= 6
cycles at 1 Hz) the normalized autocorrelation is evaluated at its first peak after
lag 0 (lags 0.25-2 s, i.e. 0.5-4 Hz). A sustained waveform repeating at a fixed
period gives a value near 1; broadband (aperiodic) delta decays to ~0.2-0.4.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

WIN_S = 6.0
HOP_S = 2.0
DELTA = (0.5, 4.0)


def delta_rhythmicity_map(data: np.ndarray, fs: float, win_s: float = WIN_S, hop_s: float = HOP_S):
    """(C, N) data -> (window centres (W,), rhythm frequency Hz (C, W),
    autocorrelation peak 0..1 (C, W), delta amplitude µV peak-to-trough (C, W))."""
    data = np.asarray(data, dtype=np.float64)
    C, N = data.shape
    win, hop = int(round(win_s * fs)), int(round(hop_s * fs))
    if N < win:
        z = np.zeros((C, 0))
        return np.zeros(0), z, z, z
    sos = butter(4, [DELTA[0] / (fs / 2), DELTA[1] / (fs / 2)], btype="bandpass", output="sos")
    xd = sosfiltfilt(sos, data, axis=1)
    starts = np.arange(0, N - win + 1, hop)
    lo, hi = int(round(fs / DELTA[1])), int(round(fs / DELTA[0]))
    freq = np.zeros((C, starts.size))
    rho = np.zeros((C, starts.size))
    amp = np.zeros((C, starts.size))
    nfft = 1 << int(np.ceil(np.log2(2 * win)))
    for wi, s in enumerate(starts):
        seg = xd[:, s:s + win]
        seg = seg - seg.mean(axis=1, keepdims=True)
        F = np.fft.rfft(seg, nfft, axis=1)
        ac = np.fft.irfft(np.abs(F) ** 2, nfft, axis=1)[:, :win]
        ac = ac / (ac[:, :1] + 1e-12)
        for c in range(C):
            # first peak after the first zero crossing (a decaying autocorrelation
            # is otherwise largest at the shortest lag)
            neg = np.nonzero(ac[c, 1:hi + 1] < 0)[0]
            if neg.size == 0:
                continue  # no oscillation within 0.5-4 Hz
            z0 = max(lo, int(neg[0]) + 1)
            seg_ac = ac[c, z0:hi + 1]
            if seg_ac.size < 3:
                continue
            # first local maximum = one period (later maxima are its multiples)
            d = np.diff(seg_ac)
            turn = np.nonzero((d[:-1] > 0) & (d[1:] <= 0))[0]
            k = int(turn[0] + 1) if turn.size else int(np.argmax(seg_ac))
            rho[c, wi] = float(np.clip(seg_ac[k], 0.0, 1.0))
            freq[c, wi] = fs / (z0 + k)
        amp[:, wi] = np.percentile(seg, 97.5, axis=1) - np.percentile(seg, 2.5, axis=1)
    return (starts + win / 2) / fs, freq, rho, amp
