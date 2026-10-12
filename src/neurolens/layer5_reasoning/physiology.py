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
    "ictal_ml": PhysiologyLabel.PATHOLOGIC,
    "spike": PhysiologyLabel.PATHOLOGIC,
    "burst_suppression": PhysiologyLabel.PATHOLOGIC,
    "triphasic_waves": PhysiologyLabel.PATHOLOGIC,
    "ecg_artifact": PhysiologyLabel.ARTIFACT,
    "drowsiness_slowing": PhysiologyLabel.PHYSIOLOGIC,
    "wicket": PhysiologyLabel.PHYSIOLOGIC,
    "lpds": PhysiologyLabel.PATHOLOGIC,
    "gpds": PhysiologyLabel.PATHOLOGIC,
    "bipds": PhysiologyLabel.PATHOLOGIC,
    "lrda": PhysiologyLabel.PATHOLOGIC,
    "grda": PhysiologyLabel.PATHOLOGIC,
    "firda": PhysiologyLabel.PATHOLOGIC,
    "extreme_delta_brush": PhysiologyLabel.PATHOLOGIC,
    "neonatal_burst_suppression": PhysiologyLabel.PATHOLOGIC,
    "neonatal_inactive": PhysiologyLabel.PATHOLOGIC,
    "neonatal_background": PhysiologyLabel.UNCERTAIN,
}


LEARNED_IED_PATHOLOGIC = 0.95
ICTAL_ML_PATHOLOGIC = 0.5


class PhysiologyEngine:
    def __init__(self, norms: NormsEngine):
        self.norms = norms

    def label_event(self, event: Event, patient: PatientInfo) -> PhysiologyLabel:
        """Assign a physiology label, adjusting for age norms where relevant."""
        base = _DEFAULT_PHYSIOLOGY.get(event.code, PhysiologyLabel.UNCERTAIN)

        # Unvalidated evidence is a candidate for review, not a finding (increment 17):
        # heuristic spike candidates (increments 15-16: ~1,800 false labels/h on normal
        # EEG) unless the learned model rates them >= 0.95, and learned seizure events
        # whose calibrated probability is < 0.5.
        if event.code == "spike" and not event.metadata.get("detector"):
            p = event.metadata.get("ied_probability_learned")
            if p is None or p < LEARNED_IED_PATHOLOGIC:
                return PhysiologyLabel.UNCERTAIN
        if event.code == "ictal_ml" and event.confidence < ICTAL_ML_PATHOLOGIC:
            return PhysiologyLabel.UNCERTAIN

        # Neonatal background is physiologic iff appropriate for the PMA
        # (discontinuity is normal in a preterm, abnormal at term).
        if event.code == "neonatal_background":
            abnormal = event.metadata.get("abnormal_for_age")
            if abnormal is True:
                return PhysiologyLabel.PATHOLOGIC
            if abnormal is False:
                return PhysiologyLabel.PHYSIOLOGIC

        # Age-gated benign variant (e.g. posterior slow waves of youth).
        if self.is_age_physiologic_variant(event, patient):
            return PhysiologyLabel.PHYSIOLOGIC

        # Age-norm refinement: a PDR within the age-expected range argues that
        # apparent 'slowing' may be physiologic for the patient.
        if event.code == "diffuse_slowing":
            pdr = next(
                (e.value for e in event.evidence if e.feature == "pdr_hz" and e.value is not None),
                None,
            )
            if (pdr is None or pdr <= 0) and patient.age_years is not None and patient.age_years < 6:
                # slow background is normal in young children: without an identifiable
                # posterior rhythm, relative slow power alone cannot establish slowing
                return PhysiologyLabel.UNCERTAIN
            if pdr is not None and pdr > 0:  # 0 = no posterior rhythm identified
                normal = self.norms.pdr_is_normal(pdr, patient)
                if normal is True:
                    return PhysiologyLabel.UNCERTAIN  # PDR ok -> re-examine slowing claim
        return base

    def variant_hint(self, event: Event) -> str | None:
        """If an epileptiform candidate matches a benign-variant profile, name it.

        Temporal sharp transients are the classic location for wicket/BETS —
        variants routinely misread as epileptiform (TZ §8.2). The definitive
        wicket resolution happens in Layer 5 plausibility against detected
        wicket rhythms; this is a lightweight hint by location.
        """
        temporal = {"T3", "T4", "T5", "T6", "F7", "F8"}
        if event.group == "ied" and set(event.localization.channels) & temporal:
            return "wicket"
        return None

    def is_age_physiologic_variant(self, event: Event, patient: PatientInfo) -> bool:
        """Age-gated benign variants (posterior slow waves of youth, etc.)."""
        age = patient.age_years
        if age is None or age >= 21:
            return False
        _, band = self.norms.band_for_patient(patient)
        variants = band.get("physiologic_variants", [])
        region = (event.localization.region or "").lower()
        if "posterior_slow_waves_of_youth" in variants and region in ("occipital", "parietal"):
            return event.code in ("focal_slowing",)
        return False
