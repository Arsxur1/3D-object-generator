"""Physiology assessment & normal-variant catalog (TZ §8.2).

Decides whether an event is physiologic (normal for age / an activation
response / a benign variant) or pathologic. The catalog lists benign variants
that are frequently mistaken for epileptiform activity.
"""

from __future__ import annotations

from ..contracts.causal import PhysiologyLabel
from ..contracts.events import Event
from ..contracts.signal import PatientInfo
from ..layer3_features.norms import NormsEngine

# Benign / physiologic variants often misread as epileptiform (TZ §8.2).
NORMAL_VARIANTS = {
    "wicket": ("Викет-спайки (доброкачественный вариант)", "Wicket-spike (xavfsiz variant)"),
    "bets_sreda": ("BETS/SREDA (доброкачественные)", "BETS/SREDA (xavfsiz)"),
    "posterior_slow_waves_of_youth": ("Задние медленные волны юности", "Yoshlikning orqa sekin to‘lqinlari"),
    "hypnagogic_hypersynchrony": ("Гипнагогическая гиперсинхрония", "Gipnagogik gipersinxroniya"),
    "mu_rhythm": ("Мю-ритм", "Mu-ritm"),
    "lambda": ("Лямбда-волны / POSTS", "Lambda to‘lqinlari / POSTS"),
    "photic_driving": ("Фотическое усвоение ритма", "Fotik ritm o‘zlashtirish"),
}

# Default physiology per event code (before context/norm adjustment).
_DEFAULT_PHYSIOLOGY = {
    "diffuse_slowing": PhysiologyLabel.PATHOLOGIC,
    "focal_slowing": PhysiologyLabel.PATHOLOGIC,
    "ictal_rhythm": PhysiologyLabel.PATHOLOGIC,
    "spike": PhysiologyLabel.PATHOLOGIC,
    "burst_suppression": PhysiologyLabel.PATHOLOGIC,
    "triphasic_waves": PhysiologyLabel.PATHOLOGIC,
    "ecg_artifact": PhysiologyLabel.ARTIFACT,
    "drowsiness_slowing": PhysiologyLabel.PHYSIOLOGIC,
    "lpds": PhysiologyLabel.PATHOLOGIC,
}


class PhysiologyEngine:
    def __init__(self, norms: NormsEngine):
        self.norms = norms

    def label_event(self, event: Event, patient: PatientInfo) -> PhysiologyLabel:
        """Assign a physiology label, adjusting for age norms where relevant."""
        base = _DEFAULT_PHYSIOLOGY.get(event.code, PhysiologyLabel.UNCERTAIN)

        # Age-norm refinement: a PDR within the age-expected range argues that
        # apparent 'slowing' may be physiologic for the patient.
        if event.code == "diffuse_slowing":
            pdr = next(
                (e.value for e in event.evidence if e.feature == "pdr_hz" and e.value is not None),
                None,
            )
            if pdr is not None:
                normal = self.norms.pdr_is_normal(pdr, patient)
                if normal is True:
                    return PhysiologyLabel.UNCERTAIN  # PDR ok -> re-examine slowing claim
        return base

    def variant_hint(self, event: Event) -> str | None:
        """If an epileptiform candidate matches a benign-variant profile, name it."""
        # Skeleton: only a placeholder hook; real variant templates land in v2.
        return None
