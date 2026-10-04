"""Config contracts — montages, filters, thresholds, and the neuro rule base.

These models validate the YAML files under ``configs/`` (TZ §0: nothing
hardcoded where a config belongs — mains frequency, montages, thresholds,
norms, the physiological rule base, ACNS terminology).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class MontageType(str, Enum):
    REFERENTIAL = "referential"
    BIPOLAR = "bipolar"
    LAPLACIAN = "laplacian"  # interface reserved; not computed in the skeleton


class MontagePair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str  # derivation label, e.g. "Fp1-F7"
    anode: str  # channel measured
    cathode: Optional[str] = None  # subtracted channel (None for referential)


class MontageConfig(BaseModel):
    """A montage definition (TZ §5: montage engine is first-class)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: MontageType
    description_ru: str = ""
    reference: Optional[str] = Field(
        default=None, description="For referential montages: the reference label."
    )
    pairs: list[MontagePair] = Field(default_factory=list)


class FilterConfig(BaseModel):
    """Preprocessing filter settings (TZ §5)."""

    model_config = ConfigDict(extra="forbid")

    highpass_hz: float = Field(default=0.5, ge=0)
    lowpass_hz: float = Field(default=70.0, gt=0)
    notch_hz: float = Field(default=50.0, description="Mains: 50 (UZ) switchable to 60.")
    notch_enabled: bool = True
    filter_order: int = Field(default=4, ge=1, le=10)


class ModeBGates(BaseModel):
    """Gates for autonomous mode B (TZ §2)."""

    model_config = ConfigDict(extra="forbid")

    min_confidence: float = Field(default=0.85, ge=0, le=1)
    min_signal_quality: float = Field(default=0.6, ge=0, le=1)
    max_artifact_fraction: float = Field(default=0.3, ge=0, le=1)
    require_plausible: bool = True


class Thresholds(BaseModel):
    """Confidence thresholds and alarm budgets (TZ §2, §10, §16)."""

    model_config = ConfigDict(extra="forbid")

    detection_min_confidence: float = Field(default=0.5, ge=0, le=1)
    mode_b: ModeBGates = Field(default_factory=ModeBGates)
    max_false_alarms_per_hour: float = Field(
        default=2.0, ge=0, description="cEEG alarm budget (placeholder for real-time, v2)."
    )
    # Detector-specific thresholds (kept in config, not hardcoded).
    diffuse_slowing_delta_theta_ratio: float = Field(default=0.55, ge=0, le=1)
    asymmetry_index_abnormal: float = Field(default=0.5, ge=0, le=1)
    suppression_amplitude_uv: float = Field(default=10.0, gt=0)
    burst_suppression_ratio: float = Field(default=0.5, ge=0, le=1)
    ictal_min_duration_s: float = Field(default=8.0, gt=0)
    ictal_rhythmicity: float = Field(default=0.45, ge=0, le=1)
    ictal_amplitude_factor: float = Field(default=1.5, ge=1)
    # Spatial extent: minimum channels in one ictal cluster. Real-data finding
    # (CHB-MIT, docs/validation_physionet.md): single-channel rhythmic runs are
    # mostly artifact/benign; seizures recruit several channels.
    ictal_min_channels: int = Field(default=1, ge=1)
    # Periodic discharges (GPD/LPD/BIPD) — ACNS §7.4
    generalized_channel_fraction: float = Field(default=0.6, ge=0, le=1)
    pd_min_discharges: int = Field(default=6, ge=3)
    pd_isi_cv_max: float = Field(default=0.35, ge=0, description="Max ISI coeff. of variation.")
    pd_freq_hz_min: float = Field(default=0.5, gt=0)
    pd_freq_hz_max: float = Field(default=3.0, gt=0)
    pd_plus_fast_ratio: float = Field(default=0.15, ge=0, le=1, description="Beta+gamma frac -> +F.")
    # Rhythmic delta (GRDA/LRDA)
    rda_min_duration_s: float = Field(default=6.0, gt=0)
    rda_rhythmicity: float = Field(default=0.4, ge=0, le=1)
    # Ictal-interictal continuum
    iic_freq_hz_min: float = Field(default=1.5, gt=0)
    iic_freq_hz_max: float = Field(default=2.5, gt=0)
    # FIRDA (frontal intermittent rhythmic delta)
    firda_max_continuous_fraction: float = Field(default=0.7, ge=0, le=1)
    firda_min_runs: int = Field(default=2, ge=1)
    # Extreme delta brush
    edb_beta_ratio: float = Field(default=0.12, ge=0, le=1)


class RuleCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    any_event: list[str] = Field(
        default_factory=list, description="Fires if any of these event codes is present."
    )
    all_events: list[str] = Field(
        default_factory=list, description="Requires all of these event codes."
    )
    context_present: list[str] = Field(
        default_factory=list,
        description="Requires clinical-context keys, e.g. 'sedatives', 'antiseizure_meds'.",
    )
    context_absent: list[str] = Field(default_factory=list)


class RuleChain(BaseModel):
    """A causal chain / rule (TZ §8.1, §18.3)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    description_ru: str = ""
    when: RuleCondition
    cause_ru: str
    cause_uz: str
    effect_ru: str
    effect_uz: str
    physiology: str = Field(default="pathologic")  # physiologic|pathologic|uncertain|artifact
    confidence: float = Field(default=0.6, ge=0, le=1)
    critical: bool = False
    # For artifact-vs-real discrimination rules (e.g. ECG artifact vs IED).
    resolves_artifact_for: list[str] = Field(default_factory=list)


class RuleBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = "0.1.0"
    chains: list[RuleChain] = Field(default_factory=list)


class AlarmRule(BaseModel):
    """Per-alarm-type gating for real-time monitoring (TZ §2, §16)."""

    model_config = ConfigDict(extra="forbid")

    min_confidence: float = Field(default=0.6, ge=0, le=1)
    persistence_windows: int = Field(
        default=2, ge=1, description="Consecutive windows the finding must persist before alarming."
    )
    refractory_s: float = Field(
        default=30.0, ge=0, description="Do not re-fire the same type within this interval."
    )
    max_per_hour: float = Field(
        default=6.0, ge=0, description="Cap on alarms of this type per hour (FA/h control)."
    )
    critical: bool = Field(
        default=True, description="Hard-safety type — never suppressed by FA/h control (TZ §10)."
    )


class RealtimeConfig(BaseModel):
    """Streaming cEEG monitor configuration (TZ §2 sub-mode, §4, §16)."""

    model_config = ConfigDict(extra="forbid")

    window_s: float = Field(default=30.0, gt=0, description="Rolling analysis window length.")
    step_s: float = Field(default=5.0, gt=0, description="Advance between analyses.")
    latency_budget_ms: float = Field(
        default=5000.0, gt=0, description="Per-window processing budget (must be << step)."
    )
    max_false_alarms_per_hour: float = Field(default=10.0, ge=0)
    alarms: dict[str, AlarmRule] = Field(
        default_factory=dict, description="Per-alarm-type rules keyed by AlarmType value."
    )
    threshold_overrides: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Detector thresholds used only by the streaming monitor (keys of Thresholds). "
            "Short windows + a rolling baseline need different operating points than "
            "whole-record analysis (validated separately on PhysioNet)."
        ),
    )

    def rule_for(self, alarm_type: str) -> AlarmRule:
        return self.alarms.get(alarm_type, AlarmRule())


class LearnedDetectorMode(BaseModel):
    """One operating point of the learned ictal detector (increment 10)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    model: str = Field(description="Model JSON path, relative to the repository root.")
    threshold: float = Field(ge=0, le=1)
    min_epochs: int = Field(ge=1)
    alarm_persistence_windows: int = Field(
        default=1, ge=1,
        description="Seizure-alarm persistence when this detector drives the monitor "
                    "(the run of min_epochs already provides persistence).")
    calibration: Optional[str] = Field(
        default=None,
        description="Calibration JSON (Platt/temperature for code ictal_ml), repo-relative. "
                    "Calibrated values are what reports and alarms show; detection itself is "
                    "gated only by threshold/min_epochs.")


class LearnedDetectorConfig(BaseModel):
    """configs/ml.yaml — learned ictal detector per path (offline / real-time)."""

    model_config = ConfigDict(extra="forbid")

    offline: Optional[LearnedDetectorMode] = None
    realtime: Optional[LearnedDetectorMode] = None
    provenance: dict = Field(default_factory=dict)

