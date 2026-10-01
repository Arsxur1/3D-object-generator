"""Event contract — output of Layer 4 detection (TZ §7).

Every detected pattern/event is:
    {code, label_ru, label_uz, localization, t_start, t_end, evidence, confidence}

``evidence`` is the grounding anchor: Layer 6 (LLM) may only make statements
that trace back to an event's evidence or a causal-graph edge (TZ §9, §16).
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class Localization(BaseModel):
    """Where on the scalp an event is expressed."""

    model_config = ConfigDict(extra="forbid")

    channels: list[str] = Field(default_factory=list)
    region: Optional[str] = Field(
        default=None,
        description="e.g. frontal, temporal_left, generalized, hemispheric_right",
    )
    lateralization: Optional[str] = Field(
        default=None, description="left | right | bilateral | generalized | midline"
    )


class EventEvidence(BaseModel):
    """A quantitative fact supporting an event — the grounding unit.

    ``feature`` names a Layer-3 feature (or Layer-2 quality flag); ``value`` and
    ``reference`` let downstream layers and the grounding validator re-check it.
    """

    model_config = ConfigDict(extra="forbid")

    feature: str
    value: Optional[float] = None
    unit: Optional[str] = None
    reference: Optional[str] = Field(
        default=None, description="Normative/threshold reference the value is compared to."
    )
    channels: list[str] = Field(default_factory=list)
    note: Optional[str] = None


class AcnsModifiers(BaseModel):
    """ACNS 2021 modifiers for periodic/rhythmic patterns (TZ §7.4)."""

    model_config = ConfigDict(extra="forbid")

    frequency_hz: Optional[float] = Field(default=None, ge=0)
    prevalence: Optional[str] = Field(
        default=None, description="rare | occasional | frequent | abundant | continuous"
    )
    plus_features: list[str] = Field(
        default_factory=list, description="e.g. +F (fast), +R (rhythmic), +S (sharp)"
    )
    iic: bool = Field(
        default=False,
        description="On the ictal-interictal continuum (needs clinical correlation).",
    )


class Event(BaseModel):
    """A detected EEG event/pattern (TZ §7)."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(description="Stable machine code, e.g. 'diffuse_slowing'.")
    label_ru: str
    label_uz: str
    group: str = Field(
        description="Detector group: background | ied | ictal | periodic | "
        "suppression | special | artifact | variant | sleep"
    )
    localization: Localization = Field(default_factory=Localization)
    t_start: float = Field(ge=0, description="Seconds from recording start.")
    t_end: float = Field(ge=0)
    evidence: list[EventEvidence] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    is_artifact_hypothesis: bool = Field(
        default=False,
        description="True when this is a candidate artifact (e.g. ECG) rather than "
        "a genuine cerebral finding — resolved in Layer 5.",
    )
    acns: Optional[AcnsModifiers] = Field(
        default=None, description="ACNS modifiers for periodic/rhythmic patterns (§7.4)."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.t_end - self.t_start)


class SeizureBurden(BaseModel):
    """Seizure-burden accounting for cEEG (TZ §7.3)."""

    model_config = ConfigDict(extra="forbid")

    n_seizures: int = 0
    total_seizure_time_s: float = 0.0
    recording_duration_s: float = 0.0
    seizure_fraction: float = Field(default=0.0, ge=0, le=1)
    seizures_per_hour: float = 0.0
    longest_seizure_s: float = 0.0
    status_epilepticus_suspected: bool = False


class DetectionResult(BaseModel):
    """Aggregate output of Layer 4 for one recording."""

    model_config = ConfigDict(extra="forbid")

    events: list[Event] = Field(default_factory=list)
    detectors_run: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    seizure_burden: Optional[SeizureBurden] = None

    def by_group(self, group: str) -> list[Event]:
        return [e for e in self.events if e.group == group]

    def codes(self) -> set[str]:
        return {e.code for e in self.events}
