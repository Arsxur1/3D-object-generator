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
from .periodic import PeriodicPatternDetector
from .seizure_burden import compute_seizure_burden
from .special import ExtremeDeltaBrushDetector, FirdaDetector, TriphasicWaveDetector
from .suppression_events import BurstSuppressionDetector
from .variants import BenignVariantDetector


def default_detectors() -> list[Detector]:
    """The classic/qEEG detector set active in the MVP."""
    return [
        DiffuseSlowingDetector(),
        FocalSlowingDetector(),
        IctalRhythmDetector(),
        PeriodicPatternDetector(),
        BurstSuppressionDetector(),
        TriphasicWaveDetector(),
        FirdaDetector(),
        ExtremeDeltaBrushDetector(),
        BenignVariantDetector(),
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
    # ACNS 2021: "triphasic waves" are GPDs with triphasic morphology. The spectral
    # triphasic detector cannot see morphology, so its event is kept only alongside
    # GPDs overlapping in time (increment 17: 35% of normal routine EEGs otherwise).
    gpds = [(e.t_start, e.t_end) for e in events if e.code == "gpds"]
    dropped = [e for e in events if e.code == "triphasic_waves"
               and not any(a < e.t_end and b > e.t_start for a, b in gpds)]
    if dropped:
        events = [e for e in events if not any(e is d for d in dropped)]
        notes.append("triphasic_waves dropped: no overlapping GPDs (ACNS 2021)")
    result = DetectionResult(events=events, detectors_run=ran, notes=notes)
    result.seizure_burden = compute_seizure_burden(
        result, sig.duration_s or 0.0, neonate=sig.patient.postmenstrual_age_weeks is not None)
    return result
