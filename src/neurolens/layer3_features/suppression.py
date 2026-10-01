"""Suppression and burst-suppression quantification (TZ §6, §7.5)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BurstSuppression:
    suppression_ratio: float           # fraction of time suppressed (0..1)
    n_bursts: int
    mean_ibi_s: float                  # mean inter-burst interval
    burst_intervals_s: list[tuple[float, float]] = field(default_factory=list)


def suppression_ratio(
    x: np.ndarray, fs: float, amp_thresh_uv: float = 10.0, min_dur_s: float = 0.5
) -> float:
    """Fraction of time the (rectified, smoothed) amplitude stays below threshold."""
    env = _envelope(x, fs)
    suppressed = env < amp_thresh_uv
    min_len = int(min_dur_s * fs)
    mask = _min_run(suppressed, min_len)
    return float(np.mean(mask)) if mask.size else 0.0


def burst_suppression(
    x: np.ndarray, fs: float, amp_thresh_uv: float = 10.0, min_dur_s: float = 0.5
) -> BurstSuppression:
    env = _envelope(x, fs)
    suppressed = _min_run(env < amp_thresh_uv, int(min_dur_s * fs))
    sr = float(np.mean(suppressed)) if suppressed.size else 0.0

    # bursts are runs of non-suppressed samples
    bursts = _runs(~suppressed)
    burst_intervals = [(a / fs, b / fs) for a, b in bursts if (b - a) >= int(min_dur_s * fs)]
    # inter-burst intervals between consecutive bursts
    ibis = [
        burst_intervals[i + 1][0] - burst_intervals[i][1]
        for i in range(len(burst_intervals) - 1)
    ]
    mean_ibi = float(np.mean(ibis)) if ibis else 0.0
    return BurstSuppression(
        suppression_ratio=sr,
        n_bursts=len(burst_intervals),
        mean_ibi_s=mean_ibi,
        burst_intervals_s=burst_intervals,
    )


def suppression_regions(
    x: np.ndarray,
    fs: float,
    amp_thresh_uv: float = 10.0,
    win_s: float = 8.0,
    frac_thresh: float = 0.4,
    min_region_s: float = 12.0,
) -> list[tuple[float, float]]:
    """Locate contiguous burst-suppression episodes (localized, not global).

    Computes a sample-level suppressed mask, smooths it into a local suppressed
    *fraction* over ``win_s`` windows, and returns regions where that fraction
    stays above ``frac_thresh`` for at least ``min_region_s`` seconds.
    """
    env = _envelope(x, fs)
    suppressed = _min_run(env < amp_thresh_uv, int(0.5 * fs))
    win = max(1, int(win_s * fs))
    kernel = np.ones(win) / win
    local = np.convolve(suppressed.astype(float), kernel, mode="same")
    region_mask = local >= frac_thresh
    return [
        (a / fs, b / fs)
        for a, b in _runs(region_mask)
        if (b - a) >= int(min_region_s * fs)
    ]


def suppression_regions_multichannel(
    channels: np.ndarray,
    fs: float,
    amp_thresh_uv: float = 10.0,
    win_s: float = 8.0,
    frac_thresh: float = 0.4,
    min_region_s: float = 12.0,
) -> list[tuple[float, float]]:
    """Locate GENERALIZED burst-suppression: suppression across channels together.

    Uses the mean per-channel amplitude envelope, so a single quiet channel does
    not trigger a (false) burst-suppression — real burst-suppression is
    generalized. ``channels`` is [n_ch][samples].
    """
    if channels.ndim != 2 or channels.shape[0] == 0:
        return []
    envs = np.stack([_envelope(channels[i], fs) for i in range(channels.shape[0])], axis=0)
    mean_env = envs.mean(axis=0)
    suppressed = _min_run(mean_env < amp_thresh_uv, int(0.5 * fs))
    win = max(1, int(win_s * fs))
    kernel = np.ones(win) / win
    local = np.convolve(suppressed.astype(float), kernel, mode="same")
    region_mask = local >= frac_thresh
    return [
        (a / fs, b / fs)
        for a, b in _runs(region_mask)
        if (b - a) >= int(min_region_s * fs)
    ]


def _envelope(x: np.ndarray, fs: float, smooth_s: float = 0.2) -> np.ndarray:
    rect = np.abs(x - np.mean(x))
    win = max(1, int(smooth_s * fs))
    kernel = np.ones(win) / win
    return np.convolve(rect, kernel, mode="same")


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, stop) index pairs for each True run."""
    if mask.size == 0:
        return []
    diff = np.diff(mask.astype(np.int8))
    starts = list(np.where(diff == 1)[0] + 1)
    stops = list(np.where(diff == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        stops = stops + [mask.size]
    return list(zip(starts, stops))


def _min_run(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Keep only True runs at least ``min_len`` samples long."""
    out = np.zeros_like(mask)
    for a, b in _runs(mask):
        if (b - a) >= max(1, min_len):
            out[a:b] = True
    return out
