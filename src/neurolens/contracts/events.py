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


class Event(BaseModel):
    """A detected EEG event/pattern (TZ §7)."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(description="Stable machine code, e.g. 'diffuse_slowing'.")
    label_ru: str
    label_uz: str
    group: str = Field(
        description="Detector group: background | ied | ictal | periodic | "
        "suppression | special | artifact | sleep"
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
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.t_end - self.t_start)


class DetectionResult(BaseModel):
    """Aggregate output of Layer 4 for one recording."""

    model_config = ConfigDict(extra="forbid")

    events: list[Event] = Field(default_factory=list)
    detectors_run: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def by_group(self, group: str) -> list[Event]:
        return [e for e in self.events if e.group == group]

    def codes(self) -> set[str]:
        return {e.code for e in self.events}
