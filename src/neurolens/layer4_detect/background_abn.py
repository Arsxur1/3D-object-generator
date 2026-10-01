"""Background abnormalities: diffuse slowing and focal slowing/asymmetry (TZ §7.1)."""

from __future__ import annotations

import numpy as np

from ..contracts.config import Thresholds
from ..contracts.events import Event, EventEvidence, Localization
from ..contracts.signal import UnifiedSignal
from ..layer2_preprocess.artifacts import ArtifactReport
from ..layer3_features.feature_set import FeatureSet
from .base import Detector


class DiffuseSlowingDetector(Detector):
    """Diffuse slowing: sustained excess of delta+theta across the scalp.

    Grades encephalopathy severity by the delta+theta relative-power fraction.
    """

    code = "diffuse_slowing"
    group = "background"

    def detect(self, sig, features, thresholds, artifacts=None) -> list[Event]:
        dt = features.epoch_relpow["delta"] + features.epoch_relpow["theta"]
        epoch_frac = dt.mean(axis=1)  # mean over channels per epoch
        thr = thresholds.diffuse_slowing_delta_theta_ratio
        slow = epoch_frac >= thr
        if slow.mean() < 0.4:  # must dominate the record
            return []

        times = features.epoch_times
        t_start = float(times[slow][0]) if slow.any() else 0.0
        t_end = float(times[slow][-1]) if slow.any() else features.duration_s
        frac = float(epoch_frac.mean())
        grade, grade_ru, grade_uz = _grade(frac)

        conf = float(np.clip((frac - thr) / (1.0 - thr) + 0.5, 0.4, 0.95))
        ru, uz = self.labels(self.code)
        return [
            Event(
                code=self.code,
                label_ru=f"{ru} ({grade_ru})",
                label_uz=f"{uz} ({grade_uz})",
                group=self.group,
                localization=Localization(region="generalized", lateralization="generalized"),
                t_start=t_start,
                t_end=t_end,
                confidence=conf,
                evidence=[
                    EventEvidence(
                        feature="rel_delta_theta_fraction",
                        value=round(frac, 3),
                        reference=f">= {thr}",
                        note=f"encephalopathy grade: {grade}",
                    ),
                    EventEvidence(
                        feature="global_rel_delta",
                        value=round(features.global_relpow["delta"], 3),
                    ),
                    EventEvidence(feature="pdr_hz", value=round(features.pdr_hz, 2), unit="Hz"),
                ],
                metadata={"grade": grade, "slow_epoch_fraction": round(float(slow.mean()), 3)},
            )
        ]


def _grade(frac: float) -> tuple[str, str, str]:
    if frac >= 0.8:
        return "severe", "тяжёлая", "og‘ir"
    if frac >= 0.65:
        return "moderate", "умеренная", "o‘rtacha"
    return "mild", "лёгкая", "yengil"


class FocalSlowingDetector(Detector):
    """Focal slowing / persistent interhemispheric asymmetry (TZ §7.1)."""

    code = "focal_slowing"
    group = "background"

    def detect(self, sig, features, thresholds, artifacts=None) -> list[Event]:
        thr = thresholds.asymmetry_index_abnormal
        # strongest homologous asymmetry
        if not features.asymmetry:
            return []
        pair, ai = max(features.asymmetry.items(), key=lambda kv: abs(kv[1]))
        if abs(ai) < thr:
            return []

        left, right = pair.split("/")
        higher = left if ai > 0 else right
        lateralization = "left" if ai > 0 else "right"
        region = _region_of(higher)
        conf = float(np.clip(abs(ai), 0.4, 0.9))
        ru, uz = self.labels(self.code)
        return [
            Event(
                code=self.code,
                label_ru=ru,
                label_uz=uz,
                group=self.group,
                localization=Localization(
                    channels=[higher], region=region, lateralization=lateralization
                ),
                t_start=0.0,
                t_end=features.duration_s,
                confidence=conf,
                evidence=[
                    EventEvidence(
                        feature="asymmetry_index",
                        value=round(ai, 3),
                        reference=f"|AI| >= {thr}",
                        channels=[left, right],
                        note=f"slow-power higher over {higher}",
                    )
                ],
                metadata={"pair": pair, "lateralization": lateralization},
            )
        ]


def _region_of(channel: str) -> str:
    c = channel.upper()
    if c.startswith(("FP", "F")):
        return "frontal"
    if c.startswith("C"):
        return "central"
    if c.startswith("P"):
        return "parietal"
    if c.startswith("O"):
        return "occipital"
    if c.startswith("T"):
        return "temporal"
    return "unknown"
