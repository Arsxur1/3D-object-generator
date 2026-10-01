"""Layer 4 — pattern & event detection (TZ §7).

Layer 4 answers *what* is on the EEG (events with localization + confidence).
*Why* / *physiologic?* / *coherent?* is Layer 5. Detectors are classic/qEEG
and template based in the MVP; ML detectors (braindecode) are v2.
"""

from __future__ import annotations

from .base import Detector, LABELS
from .background_abn import DiffuseSlowingDetector, FocalSlowingDetector
from .ictal import IctalRhythmDetector
from .ied import SpikeDetector
from .artifact_events import EcgArtifactDetector
from .suppression_events import BurstSuppressionDetector
from .special import TriphasicWaveDetector
from .registry import default_detectors, run_detectors

__all__ = [
    "Detector",
    "LABELS",
    "DiffuseSlowingDetector",
    "FocalSlowingDetector",
    "IctalRhythmDetector",
    "SpikeDetector",
    "EcgArtifactDetector",
    "BurstSuppressionDetector",
    "TriphasicWaveDetector",
    "default_detectors",
    "run_detectors",
]
