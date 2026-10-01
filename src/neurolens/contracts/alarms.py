"""Real-time alarm contracts (TZ §2 cEEG sub-mode, §10, §16).

An alarm is a near-real-time, sensitivity-prioritized notification raised by the
streaming monitor. Every alarm carries an EEG fragment reference for physician
verification, and the monitor tracks false-alarms/hour to avoid alarm fatigue.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .events import SeizureBurden
from .report import Bilingual


class AlarmType(str, Enum):
    SEIZURE = "seizure"
    STATUS_EPILEPTICUS = "status_epilepticus"
    IIC = "iic"  # ictal-interictal continuum pattern needing attention
    BURST_SUPPRESSION = "burst_suppression"
    SUPPRESSION = "suppression"


class AlarmSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Alarm(BaseModel):
    """A single near-real-time alarm (TZ §2, §10)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: AlarmType
    severity: AlarmSeverity
    message: Bilingual
    t_start: float = Field(ge=0, description="Stream time (s) of the alarm window start.")
    t_end: float = Field(ge=0)
    channels: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list, description="Feature/event grounding refs.")
    window_index: int = Field(ge=0)
    latency_ms: float = Field(ge=0, description="Processing latency for the triggering window.")
    fragment_path: Optional[str] = Field(
        default=None, description="Path to the EEG fragment (PNG) for verification (TZ §10)."
    )
    is_critical_finding: bool = Field(
        default=False, description="Hard-safety finding — never suppressed by FA/h control."
    )


class MonitorSummary(BaseModel):
    """Session summary of a streaming cEEG run (TZ §16 budgets)."""

    model_config = ConfigDict(extra="forbid")

    source: str
    montage: str
    duration_s: float
    n_windows: int
    window_s: float
    step_s: float
    alarms: list[Alarm] = Field(default_factory=list)
    n_alarms_raised: int = 0
    n_alarms_suppressed: int = 0
    alarms_per_hour: float = 0.0
    false_alarm_budget_per_hour: float = 0.0
    within_budget: bool = True
    seizure_burden: Optional[SeizureBurden] = None
    latency_ms_mean: float = 0.0
    latency_ms_p95: float = 0.0
    realtime_feasible: bool = True  # per-window latency << step
