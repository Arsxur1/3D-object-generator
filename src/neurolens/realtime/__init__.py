"""Real-time cEEG monitoring with alarms (TZ §2 sub-mode, §4, §10, §16).

Streaming pipeline: a :class:`StreamSource` yields buffered, overlapping windows;
the :class:`RealtimeMonitor` runs a lightweight subset of Layers 2-4 per window
and raises :class:`~neurolens.contracts.alarms.Alarm` s via the
:class:`AlarmManager`, which controls false-alarms/hour (persistence, refractory,
per-hour caps) and attaches an EEG fragment to each alarm.

The concrete source is a streaming EDF replay (offline, deterministic); an LSL
source is provided as an optional interface (v2, requires liblsl).
"""

from __future__ import annotations

from .stream import Chunk, StreamSource, EdfReplaySource, LslStreamSource
from .buffer import WindowBuffer
from .alarms import AlarmManager
from .monitor import RealtimeMonitor

__all__ = [
    "Chunk",
    "StreamSource",
    "EdfReplaySource",
    "LslStreamSource",
    "WindowBuffer",
    "AlarmManager",
    "RealtimeMonitor",
]
