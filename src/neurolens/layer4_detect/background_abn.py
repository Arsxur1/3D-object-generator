"""Background abnormalities: diffuse slowing and focal slowing/asymmetry (TZ §7.1)."""

from __future__ import annotations

import numpy as np

from ..contracts.config import Thresholds
from ..contracts.events import Event, EventEvidence, Localization
from ..contracts.signal import UnifiedSignal
from ..layer2_preprocess.artifacts import ArtifactReport
from ..layer3_features.feature_set import FeatureSet
from .base import Detector


_EXCLUDE_SLOWING = {"FP1", "FP2", "F7", "F8", "A1", "A2", "M1", "M2"}


class DiffuseSlowingDetector(Detector):
    """Diffuse slowing: sustained excess of delta+theta across the scalp.

    Grades encephalopathy severity by the delta+theta relative-power fraction.
    """

    code = "diffuse_slowing"
    group = "background"

    def detect(self, sig, features, thresholds, artifacts=None) -> list[Event]:
        dt = features.epoch_relpow["delta"] + features.epoch_relpow["theta"]
        # increment 17: frontopolar/anterior-temporal channels carry eye-movement delta
        # and ear electrodes are references - neither reflects cerebral slowing
        keep = [i for i, c in enumerate(features.eeg_channels) if c.upper() not in _EXCLUDE_SLOWING]
        if len(keep) >= 4:
            dt = dt[:, keep]
        epoch_frac = np.median(dt, axis=1)  # median over channels per epoch
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
    """Focal slowing: *persistent* excess of slow activity on one side (TZ §7.1).

    Increment 17: per epoch, the difference of relative delta+theta power between
    homologous channels; focal when its median over the record is >= the threshold and
    it has the same sign in >= 70% of epochs. Relative power is insensitive to electrode
    impedance, and the median to artefact bursts; Fp1/Fp2 and F7/F8 (eye movements) are
    excluded. The previous whole-record absolute-power index flagged 39% of normal
    routine EEGs (one noisy electrode suffices)."""

    code = "focal_slowing"
    group = "background"
    PAIRS = (("F3", "F4"), ("C3", "C4"), ("P3", "P4"), ("O1", "O2"), ("T3", "T4"), ("T5", "T6"))

    def detect(self, sig, features, thresholds, artifacts=None) -> list[Event]:
        ch = features.eeg_channels
        if features.epoch_times.size == 0:
            return []
        dt = features.epoch_relpow["delta"] + features.epoch_relpow["theta"]
        thr = thresholds.focal_slowing_min_diff
        best = None
        for left, right in self.PAIRS:
            if left not in ch or right not in ch:
                continue
            d = dt[:, ch.index(left)] - dt[:, ch.index(right)]
            med = float(np.median(d))
            consistency = float(max((d > 0).mean(), (d < 0).mean()))
            if consistency >= 0.7 and abs(med) >= thr and (best is None or abs(med) > abs(best[2])):
                best = (left, right, med, consistency)
        if best is None:
            return []
        left, right, med, consistency = best
        higher = left if med > 0 else right
        lateralization = "left" if med > 0 else "right"
        ru, uz = self.labels(self.code)
        return [
            Event(
                code=self.code, label_ru=ru, label_uz=uz, group=self.group,
                localization=Localization(channels=[higher], region=_region_of(higher),
                                          lateralization=lateralization),
                t_start=0.0, t_end=features.duration_s,
                confidence=float(np.clip(0.4 + abs(med), 0.4, 0.9)),
                evidence=[
                    EventEvidence(feature="rel_slow_power_difference", value=round(med, 3),
                                  reference=f"|median| >= {thr}", channels=[left, right],
                                  note=f"more slow activity over {higher}"),
                    EventEvidence(feature="same_sign_epoch_fraction", value=round(consistency, 3),
                                  reference=">= 0.7"),
                ],
                metadata={"pair": f"{left}/{right}", "lateralization": lateralization},
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
