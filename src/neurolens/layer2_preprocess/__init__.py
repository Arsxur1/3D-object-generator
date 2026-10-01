"""Layer 2 — preprocessing, quality, montage, artifacts (TZ §5)."""

from __future__ import annotations

from .filters import apply_filters
from .reref import rereference
from .montage_engine import MontageEngine, MontagedSignal
from .bad_channels import detect_bad_channels
from .sqi import compute_sqi
from .artifacts import flag_artifacts, ArtifactStep, ICAStep

__all__ = [
    "apply_filters",
    "rereference",
    "MontageEngine",
    "MontagedSignal",
    "detect_bad_channels",
    "compute_sqi",
    "flag_artifacts",
    "ArtifactStep",
    "ICAStep",
]
