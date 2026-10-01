"""Detector interface and bilingual label catalog (TZ §7)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..contracts.config import Thresholds
from ..contracts.events import Event
from ..contracts.signal import UnifiedSignal
from ..layer3_features.feature_set import FeatureSet
from ..layer2_preprocess.artifacts import ArtifactReport

# Canonical RU/UZ labels per event code (single source; used by all detectors).
LABELS: dict[str, tuple[str, str]] = {
    "diffuse_slowing": ("Диффузное замедление фоновой активности", "Fon faolligining diffuz sekinlashuvi"),
    "focal_slowing": ("Очаговое замедление / асимметрия", "O‘choqli sekinlashuv / assimetriya"),
    "ictal_rhythm": ("Иктальный ритмический паттерн (судорожная активность)", "Iktal ritmik pattern (tutqanoq faolligi)"),
    "spike": ("Эпилептиформный разряд (спайк/острая волна)", "Epileptiform razryad (spike/o‘tkir to‘lqin)"),
    "ecg_artifact": ("ЭКГ-артефакт", "EKG artefakti"),
    "burst_suppression": ("Паттерн «вспышка-подавление»", "“Portlash-bostirish” patterni"),
    "triphasic_waves": ("Трифазные волны", "Uch fazali to‘lqinlar"),
    "drowsiness_slowing": ("Замедление при сонливости (физиологическое)", "Uyquchanlikdagi sekinlashuv (fiziologik)"),
}


class Detector(ABC):
    """Detects one family of events from signal + Layer-3 features."""

    code: str = "detector"
    group: str = "background"

    @abstractmethod
    def detect(
        self,
        sig: UnifiedSignal,
        features: FeatureSet,
        thresholds: Thresholds,
        artifacts: ArtifactReport | None = None,
    ) -> list[Event]:
        ...

    def labels(self, code: str) -> tuple[str, str]:
        return LABELS.get(code, (code, code))
