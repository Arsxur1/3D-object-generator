"""Detector registry — assemble and run the MVP detector set (TZ §7)."""

from __future__ import annotations

from ..contracts.config import Thresholds
from ..contracts.events import DetectionResult, Event
from ..contracts.signal import UnifiedSignal
from ..layer2_preprocess.artifacts import ArtifactReport
from ..layer3_features.feature_set import FeatureSet
from .artifact_events import EcgArtifactDetector
from .background_abn import DiffuseSlowingDetector, FocalSlowingDetector
from .base import Detector
from .ictal import IctalRhythmDetector
from .ied import SpikeDetector
from .special import TriphasicWaveDetector
from .suppression_events import BurstSuppressionDetector


def default_detectors() -> list[Detector]:
    """The classic/qEEG detector set active in the MVP skeleton."""
    return [
        DiffuseSlowingDetector(),
        FocalSlowingDetector(),
        IctalRhythmDetector(),
        BurstSuppressionDetector(),
        TriphasicWaveDetector(),
        EcgArtifactDetector(),
        SpikeDetector(),
    ]


def run_detectors(
    sig: UnifiedSignal,
    features: FeatureSet,
    thresholds: Thresholds,
    artifacts: ArtifactReport | None = None,
    detectors: list[Detector] | None = None,
) -> DetectionResult:
    detectors = detectors or default_detectors()
    events: list[Event] = []
    ran: list[str] = []
    notes: list[str] = []
    for det in detectors:
        ran.append(det.code)
        try:
            found = det.detect(sig, features, thresholds, artifacts)
        except Exception as exc:  # a broken detector must not sink the pipeline
            found = []
            notes.append(f"detector {det.code} failed: {exc}")
        # keep only events above the detection confidence floor
        events.extend(e for e in found if e.confidence >= thresholds.detection_min_confidence)
    return DetectionResult(events=events, detectors_run=ran, notes=notes)
