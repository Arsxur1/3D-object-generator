"""API request models (responses reuse the core contracts)."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class AnalyzeParams(BaseModel):
    """Analysis parameters (shared by JSON and multipart endpoints)."""

    mode: str = Field(default="A", pattern="^[ABab]$")
    montage: str = "double_banana"
    provider: str = Field(default="deterministic", pattern="^(auto|anthropic|deterministic)$")
    age_years: Optional[float] = None
    postmenstrual_age_weeks: Optional[float] = None
    sedatives: list[str] = Field(default_factory=list)
    antiseizure_meds: list[str] = Field(default_factory=list)
    temperature_c: Optional[float] = None
    clinical_question: Optional[str] = None


class AnalyzeRequest(AnalyzeParams):
    path: str = Field(description="Server-side path to an EDF (edge/bedside deployment).")


class MonitorRequest(BaseModel):
    path: str
    montage: str = "double_banana"
    window_s: Optional[float] = None
    step_s: Optional[float] = None
    # patient descriptors route the seizure detector (neonates: configs/ml.yaml neonatal)
    age_years: Optional[float] = None
    postmenstrual_age_weeks: Optional[float] = None


class FeedbackRequest(BaseModel):
    """Neurophysiologist correction (TZ §13)."""

    recording_id: str
    target_kind: str = Field(pattern="^(event|localization|causal_edge|impression)$")
    target_ref: str
    action: str = Field(pattern="^(confirm|reject|edit)$")
    corrected_value: Optional[dict[str, Any]] = None
    reviewer: Optional[str] = None
    model_confidence: Optional[float] = None
    notes: Optional[str] = None
