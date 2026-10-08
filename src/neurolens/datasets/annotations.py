"""Dataset-agnostic expert annotations (ground truth for evaluation)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SeizureInterval:
    onset_s: float
    offset_s: float

    @property
    def duration_s(self) -> float:
        return self.offset_s - self.onset_s


@dataclass
class RecordAnnotation:
    """One EDF file and its expert-marked seizures (seconds from file start)."""

    database: str
    subject: str
    file: str  # path relative to the database root, e.g. "chb01/chb01_03.edf"
    seizures: list[SeizureInterval] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    sampling_rate_hz: float | None = None
    duration_s: float | None = None
    warnings: list[str] = field(default_factory=list)
    # regions marked by some but not all annotators (multi-expert datasets): neither
    # counted as seizures nor required to be detected; used by secondary analyses
    ambiguous: list[SeizureInterval] = field(default_factory=list)

    @property
    def has_seizure(self) -> bool:
        return bool(self.seizures)
