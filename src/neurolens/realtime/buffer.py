"""Rolling window buffer for streaming analysis (TZ §4: buffered windows w/ overlap)."""

from __future__ import annotations

import numpy as np


class WindowBuffer:
    """Fixed-size rolling buffer that emits overlapping analysis windows.

    Appends incoming chunks; once ``window_s`` of data is held, emits the latest
    window every time ``step_s`` of new samples has accumulated (overlap =
    window - step). Memory is bounded to one window.
    """

    def __init__(self, n_channels: int, fs: float, window_s: float, step_s: float):
        self.fs = fs
        self.window = max(1, int(round(window_s * fs)))
        self.step = max(1, int(round(step_s * fs)))
        self.buf = np.zeros((n_channels, self.window), dtype=np.float64)
        self.filled = 0
        self.total = 0          # total samples ingested
        self._since_emit = 0

    def push(self, chunk: np.ndarray) -> list[tuple[np.ndarray, int]]:
        """Append a chunk; return list of (window_copy, start_sample) ready to analyze."""
        n = chunk.shape[1]
        if n == 0:
            return []
        if n >= self.window:
            self.buf = chunk[:, -self.window :].astype(np.float64).copy()
            self.filled = self.window
        else:
            self.buf = np.roll(self.buf, -n, axis=1)
            self.buf[:, -n:] = chunk
            self.filled = min(self.window, self.filled + n)
        self.total += n
        self._since_emit += n

        outs: list[tuple[np.ndarray, int]] = []
        if self.filled >= self.window and self._since_emit >= self.step:
            start_sample = self.total - self.window
            outs.append((self.buf.copy(), start_sample))
            # keep overlap semantics: one emit per accumulated step
            self._since_emit = self._since_emit % self.step
        return outs
