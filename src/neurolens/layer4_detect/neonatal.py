"""Neonatal background detector (TZ §1, §6 — newborns/aEEG first-class).

Runs only when a postmenstrual age (PMA) is provided. Classifies the aEEG
background (Hellström-Westas), detects sleep-wake cycling, quantifies
discontinuity (IBI / continuity), and judges whether the pattern is appropriate
for the PMA (discontinuity is normal in a preterm, abnormal at term). Emits a
``neonatal_background`` event plus a severe-pattern code when relevant, which the
Layer-5 rules turn into age-aware interpretation.
"""

from __future__ import annotations

import numpy as np

from ..contracts.events import Event, EventEvidence, Localization
from ..layer3_features.aeeg_classify import AeegCategory, classify_aeeg
from ..layer3_features.norms import NormsEngine
from .base import Detector


class NeonatalBackgroundDetector(Detector):
    code = "neonatal_background"
    group = "background"

    def __init__(self, norms: NormsEngine):
        self.norms = norms

    def detect(self, sig, features, thresholds, artifacts=None) -> list[Event]:
        patient = sig.patient
        pma = patient.postmenstrual_age_weeks
        if pma is None or features.aeeg is None:
            return []

        assess = classify_aeeg(features.aeeg, burst_suppression_ratio=features.burst.suppression_ratio)
        continuity = float(1.0 - features.burst.suppression_ratio)
        ibi = float(features.burst.mean_ibi_s)

        exp = self.norms.neonatal_expectations(patient) or {}
        expected_cat = exp.get("expected_aeeg")
        swc_expected = bool(exp.get("swc_expected", False))
        max_ibi = float(exp.get("max_normal_ibi_s", 30.0))
        continuity_ok = self.norms.is_continuity_normal_for_pma(continuity, patient)

        abnormal_for_age = (
            (continuity_ok is False)
            or assess.category in (AeegCategory.BS, AeegCategory.FT, AeegCategory.CLV)
            or (swc_expected and not assess.swc_present)
            or (ibi > max_ibi)
        )

        meta = {
            "aeeg_category": assess.category.value,
            "aeeg_upper_uv": assess.upper_uv,
            "aeeg_lower_uv": assess.lower_uv,
            "swc_present": assess.swc_present,
            "swc_cycles": assess.swc_cycles,
            "ibi_s": round(ibi, 1),
            "continuity": round(continuity, 3),
            "pma_weeks": pma,
            "expected_aeeg_category": expected_cat,
            "swc_expected": swc_expected,
            "abnormal_for_age": bool(abnormal_for_age),
        }
        events: list[Event] = []
        events.append(Event(
            code=self.code,
            label_ru=f"Неонатальный фон: {assess.label_ru}"
                     + (", SWC+" if assess.swc_present else ", SWC−")
                     + (f" (не по возрасту ПМВ {pma:.0f} нед)" if abnormal_for_age else f" (по возрасту ПМВ {pma:.0f} нед)"),
            label_uz=f"Neonatal fon: {assess.label_uz}",
            group=self.group,
            localization=Localization(region="generalized", lateralization="generalized"),
            t_start=0.0,
            t_end=sig.duration_s or 0.0,
            confidence=0.7,
            evidence=[
                EventEvidence(feature="aeeg_category", note=assess.category.value,
                              reference=f"ожидаемо: {expected_cat}"),
                EventEvidence(feature="continuity", value=round(continuity, 3),
                              reference=f">= {exp.get('continuity_normal_min')}"),
                EventEvidence(feature="mean_ibi_s", value=round(ibi, 1), unit="s",
                              reference=f"<= {max_ibi}"),
                EventEvidence(feature="swc_present", value=float(assess.swc_present),
                              reference=f"ожидаемо: {swc_expected}"),
            ],
            metadata=meta,
        ))

        # severe patterns -> dedicated codes for the neonatal causal rules / critical scan
        if assess.category == AeegCategory.BS:
            ru, uz = self.labels("neonatal_burst_suppression")
            events.append(Event(
                code="neonatal_burst_suppression", label_ru=ru, label_uz=uz,
                group="suppression", localization=Localization(region="generalized"),
                t_start=0.0, t_end=sig.duration_s or 0.0, confidence=0.75,
                evidence=[EventEvidence(feature="aeeg_category", note="BS"),
                          EventEvidence(feature="pma_weeks", value=float(pma))],
                metadata=meta,
            ))
        elif assess.category == AeegCategory.FT:
            ru, uz = self.labels("neonatal_inactive")
            events.append(Event(
                code="neonatal_inactive", label_ru=ru, label_uz=uz,
                group="suppression", localization=Localization(region="generalized"),
                t_start=0.0, t_end=sig.duration_s or 0.0, confidence=0.7,
                evidence=[EventEvidence(feature="aeeg_category", note="FT")],
                metadata=meta,
            ))
        return events
