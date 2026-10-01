"""aEEG background classification & sleep-wake cycling (TZ §6, neonatology).

Classifies the amplitude-integrated EEG background into the Hellström-Westas
categories from the upper/lower amplitude margins (in raw µV), and detects
sleep-wake cycling (SWC) as cyclic modulation of the lower margin. These are the
core neonatal aEEG readouts (§1: aEEG first-class).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .aeeg import AEEG


class AeegCategory(str, Enum):
    CNV = "CNV"   # continuous normal voltage
    DNV = "DNV"   # discontinuous normal voltage
    BS = "BS"     # burst-suppression
    CLV = "CLV"   # continuous low voltage
    FT = "FT"     # flat / inactive trace


_LABELS = {
    AeegCategory.CNV: ("Непрерывный нормальный вольтаж (CNV)", "Uzluksiz normal kuchlanish (CNV)"),
    AeegCategory.DNV: ("Прерывистый нормальный вольтаж (DNV)", "Uzlukli normal kuchlanish (DNV)"),
    AeegCategory.BS: ("Вспышка-подавление (BS)", "Portlash-bostirish (BS)"),
    AeegCategory.CLV: ("Непрерывный низкий вольтаж (CLV)", "Uzluksiz past kuchlanish (CLV)"),
    AeegCategory.FT: ("Плоская/неактивная кривая (FT)", "Tekis/faolsiz egri (FT)"),
}

# Voltage thresholds (µV), Hellström-Westas aEEG classification (standard).
FLAT_UPPER = 5.0     # upper below this -> inactive/flat
CLV_UPPER = 10.0     # continuous low voltage upper margin
CNV_LOWER = 5.0      # lower margin at/above this -> continuous
BS_UPPER = 25.0      # bursts reach a high upper margin
BS_LOWER = 3.0       # suppression between bursts is very low


@dataclass
class AeegAssessment:
    category: AeegCategory
    upper_uv: float
    lower_uv: float
    swc_present: bool
    swc_cycles: int
    label_ru: str
    label_uz: str


def classify_aeeg(aeeg: AEEG, burst_suppression_ratio: float | None = None) -> AeegAssessment:
    """Classify the aEEG background and detect sleep-wake cycling."""
    u = aeeg.median_upper_raw()
    l = aeeg.median_lower_raw()

    if u < FLAT_UPPER:
        cat = AeegCategory.FT
    elif u < CLV_UPPER and l < CNV_LOWER:
        cat = AeegCategory.CLV
    elif l >= CNV_LOWER:
        cat = AeegCategory.CNV
    else:  # lower < 5, upper >= 10 -> discontinuous; BS if bursts high & baseline very low
        bs_like = (u >= BS_UPPER and l < BS_LOWER) or (
            burst_suppression_ratio is not None and burst_suppression_ratio >= 0.5
        )
        cat = AeegCategory.BS if bs_like else AeegCategory.DNV

    present, cycles = detect_swc(aeeg)
    ru, uz = _LABELS[cat]
    return AeegAssessment(
        category=cat, upper_uv=round(u, 1), lower_uv=round(l, 1),
        swc_present=present, swc_cycles=cycles, label_ru=ru, label_uz=uz,
    )


def detect_swc(aeeg: AEEG, modulation: float = 0.3, min_cycles: int = 1) -> tuple[bool, int]:
    """Sleep-wake cycling = cyclic modulation of the lower aEEG margin.

    Returns (present, n_cycles). SWC is a marker of an intact, maturing brain;
    its absence in a term neonate is concerning.
    """
    lo = aeeg.lower_raw if aeeg.lower_raw is not None else aeeg.lower_uv
    lo = np.asarray(lo, dtype=float)
    if lo.size < 6:
        return False, 0
    # smooth to suppress epoch noise (edge-aware, so a flat trace stays flat),
    # then measure modulation depth + oscillations
    from scipy.ndimage import uniform_filter1d

    k = max(1, lo.size // 12)
    s = uniform_filter1d(lo, size=k, mode="nearest")
    depth = (s.max() - s.min()) / (s.mean() + 1e-9)
    # count local maxima (cycles) on the smoothed lower margin
    d = np.diff(s)
    cycles = int(np.sum((d[:-1] > 0) & (d[1:] <= 0)))
    present = bool(depth >= modulation and cycles >= min_cycles)
    return present, cycles


def category_labels(cat: AeegCategory) -> tuple[str, str]:
    return _LABELS[cat]
