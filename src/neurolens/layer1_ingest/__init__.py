"""Layer 1 — ingestion & normalization (TZ §4).

Reads clinical/instrument formats and the real-time stream into the single
internal representation (``UnifiedSignal``). EDF/EDF+ is the mandatory MVP
input; instrument formats register behind the same ``Ingestor`` interface (v2).
"""

from __future__ import annotations

from .base import Ingestor
from .edf import EdfIngestor
from .registry import get_ingestor, register_ingestor, ingest
from . import electrodes

__all__ = [
    "Ingestor",
    "EdfIngestor",
    "get_ingestor",
    "register_ingestor",
    "ingest",
    "electrodes",
]
