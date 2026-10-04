"""Stream sources for real-time cEEG (TZ §4).

A :class:`StreamSource` yields :class:`Chunk` s of samples in stream order.
``EdfReplaySource`` replays an EDF file chunk-by-chunk (optionally paced to
wall-clock) — the concrete offline/deterministic source. ``LslStreamSource`` is
an optional interface over Lab Streaming Layer (requires the native ``liblsl``;
a no-op stub otherwise, wired in v2).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ..layer1_ingest.registry import ingest


@dataclass
class Chunk:
    """A block of consecutive samples starting at ``t_start`` seconds."""

    data: np.ndarray          # [channels][samples]
    t_start: float            # stream time of the first sample (seconds)


class StreamSource(ABC):
    channel_names: list[str]
    sampling_rate_hz: float
    reference: str = "unknown"

    @abstractmethod
    def chunks(self) -> Iterator[Chunk]:
        """Yield chunks of samples in stream order."""


class EdfReplaySource(StreamSource):
    """Replays an EDF as a stream of fixed-size chunks."""

    def __init__(self, path: str | Path, chunk_s: float = 5.0, realtime: bool = False):
        sig = ingest(path)
        # keep float32 (as ingested); the window buffer is float64 and the monitor
        # re-casts windows to float32, so values are identical at half the memory
        self._data = sig.signal
        self.channel_names = list(sig.channel_names)
        self.sampling_rate_hz = float(sig.sampling_rate_hz)
        self.reference = sig.reference
        self.chunk_samples = max(1, int(round(chunk_s * self.sampling_rate_hz)))
        self.realtime = realtime
        self.source_name = Path(path).name

    def chunks(self) -> Iterator[Chunk]:
        fs = self.sampling_rate_hz
        n = self._data.shape[1]
        for start in range(0, n, self.chunk_samples):
            stop = min(start + self.chunk_samples, n)
            if self.realtime:
                time.sleep((stop - start) / fs)  # pace to wall-clock
            yield Chunk(data=self._data[:, start:stop], t_start=start / fs)


class LslStreamSource(StreamSource):  # pragma: no cover - requires liblsl
    """Optional LSL source (TZ §4). Requires ``pylsl`` + native ``liblsl``.

    Kept as an interface so the monitor is transport-agnostic; not exercised in
    the offline test suite.
    """

    def __init__(self, stream_name: str | None = None, chunk_s: float = 5.0):
        try:
            import pylsl  # noqa: F401
        except Exception as exc:  # liblsl not installed
            raise RuntimeError(
                "LSL source requires pylsl + native liblsl (v2). "
                "Use EdfReplaySource for offline streaming."
            ) from exc
        self._pylsl = __import__("pylsl")
        streams = self._pylsl.resolve_streams()
        if not streams:
            raise RuntimeError("no LSL streams found")
        info = next((s for s in streams if not stream_name or s.name() == stream_name), streams[0])
        self._inlet = self._pylsl.StreamInlet(info)
        self.sampling_rate_hz = float(info.nominal_srate())
        self.channel_names = _lsl_channel_names(info)
        self.reference = "unknown"
        self.chunk_s = chunk_s

    def chunks(self) -> Iterator[Chunk]:
        fs = self.sampling_rate_hz
        want = int(self.chunk_s * fs)
        t = 0.0
        while True:
            samples, _ = self._inlet.pull_chunk(timeout=self.chunk_s, max_samples=want)
            if not samples:
                break
            data = np.asarray(samples, dtype=np.float64).T  # [channels][samples]
            yield Chunk(data=data, t_start=t)
            t += data.shape[1] / fs


def _lsl_channel_names(info) -> list[str]:  # pragma: no cover
    names = []
    ch = info.desc().child("channels").child("channel")
    for _ in range(info.channel_count()):
        names.append(ch.child_value("label"))
        ch = ch.next_sibling()
    return names or [f"ch{i}" for i in range(info.channel_count())]
