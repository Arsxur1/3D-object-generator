"""Unified signal representation — the contract passed between layers (TZ §3).

The signal array itself is carried out-of-band as a numpy array (float32,
shape ``[channels][samples]``, microvolts) for efficiency; this module models
all the *metadata* that travels with it and is serialized to JSON.

``UnifiedSignal`` is deliberately close to the TZ §3 JSON contract so the
on-disk/JSON form and the in-memory form stay aligned.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ElectrodeSystem(str, Enum):
    TEN_TWENTY = "10-20"
    TEN_TEN = "10-10"
    TEN_FIVE = "10-5"
    NEONATAL_REDUCED = "neonatal_reduced"
    UNKNOWN = "unknown"


class PatientInfo(BaseModel):
    """Patient descriptors. Age / postmenstrual age drive interpretation (TZ §1)."""

    model_config = ConfigDict(extra="forbid")

    age_years: Optional[float] = Field(default=None, ge=0, le=130)
    postmenstrual_age_weeks: Optional[float] = Field(
        default=None, ge=20, le=60, description="Key parameter for neonatal EEG maturation."
    )
    sex: Optional[str] = None


class ClinicalContext(BaseModel):
    """Clinical context — input to the causal/physiological analysis (TZ §3, §8).

    Everything is optional; presence sharpens the causal chains (e.g. sedation
    or hypothermia distinguishes burst-suppression from severe injury).
    """

    model_config = ConfigDict(extra="forbid")

    sedatives: list[str] = Field(default_factory=list)
    antiseizure_meds: list[str] = Field(default_factory=list)
    temperature_c: Optional[float] = Field(default=None, ge=20, le=45)
    clinical_question: Optional[str] = None
    symptoms: list[str] = Field(default_factory=list)
    prior_eeg_id: Optional[str] = None


class QualityInfo(BaseModel):
    """Per-channel signal-quality metadata produced by Layer 2."""

    model_config = ConfigDict(extra="forbid")

    per_channel_sqi: list[float] = Field(default_factory=list)
    impedances: list[float] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    digitized: bool = False
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    source_file: Optional[str] = None
    pipeline_version: Optional[str] = None


class UnifiedSignal(BaseModel):
    """Single internal representation of an EEG recording (TZ §3).

    The numeric samples live in :attr:`signal` (a numpy array, excluded from
    JSON). :meth:`metadata_dict` returns the JSON-safe metadata contract.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # Out-of-band numeric payload: float32[channels][samples], microvolts.
    signal: Any = Field(repr=False, exclude=True)

    sampling_rate_hz: float = Field(gt=0)
    channel_names: list[str]
    electrode_system: ElectrodeSystem = ElectrodeSystem.UNKNOWN
    reference: str = "unknown"
    duration_s: Optional[float] = Field(default=None, ge=0)
    units: str = "uV"
    source: str = "edf"

    patient: PatientInfo = Field(default_factory=PatientInfo)
    context: ClinicalContext = Field(default_factory=ClinicalContext)
    quality: QualityInfo = Field(default_factory=QualityInfo)
    provenance: Provenance = Field(default_factory=Provenance)

    @field_validator("signal")
    @classmethod
    def _coerce_signal(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("signal must be 2-D [channels][samples]")
        return arr

    @model_validator(mode="after")
    def _check_shapes(self) -> "UnifiedSignal":
        n_ch = self.signal.shape[0]
        if len(self.channel_names) != n_ch:
            raise ValueError(
                f"channel_names ({len(self.channel_names)}) must match "
                f"signal channels ({n_ch})"
            )
        if self.duration_s is None:
            self.duration_s = float(self.signal.shape[1]) / float(self.sampling_rate_hz)
        return self

    # -- convenience -----------------------------------------------------
    @property
    def n_channels(self) -> int:
        return int(self.signal.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.signal.shape[1])

    def channel_index(self, name: str) -> int:
        try:
            return self.channel_names.index(name)
        except ValueError as exc:  # pragma: no cover - defensive
            raise KeyError(f"unknown channel {name!r}") from exc

    def metadata_dict(self) -> dict[str, Any]:
        """JSON-safe metadata (numeric samples excluded)."""
        return self.model_dump(mode="json")

    def with_signal(self, new_signal: np.ndarray, **overrides: Any) -> "UnifiedSignal":
        """Return a copy carrying a new sample array (used across Layer 2 steps)."""
        data = self.model_dump(exclude={"signal"})
        data.update(overrides)
        return UnifiedSignal(signal=new_signal, **data)
