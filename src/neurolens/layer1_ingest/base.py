"""Ingestor interface — the Layer-1 contract (TZ §4)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..contracts.signal import ClinicalContext, PatientInfo, UnifiedSignal


class Ingestor(ABC):
    """Reads a source into a :class:`UnifiedSignal`.

    Implementations normalize channel names, detect the electrode system,
    preserve the reference, and convert to microvolts.
    """

    #: file extensions this ingestor handles (lowercase, with dot)
    extensions: tuple[str, ...] = ()
    #: human-readable name / TZ source tag
    source_name: str = "unknown"

    @abstractmethod
    def read(
        self,
        path: str | Path,
        *,
        patient: PatientInfo | None = None,
        context: ClinicalContext | None = None,
    ) -> UnifiedSignal:
        """Read ``path`` and return a normalized :class:`UnifiedSignal`."""

    def can_read(self, path: str | Path) -> bool:
        return Path(path).suffix.lower() in self.extensions
