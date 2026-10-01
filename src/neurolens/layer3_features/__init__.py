"""Layer 3 — quantitative EEG and features (TZ §6).

All feature code operates on plain numpy arrays (auditable / regulator-ready)
and is unit-testable in isolation. The pipeline assembles a :class:`FeatureSet`
that Layer 4 detectors consume.
"""

from __future__ import annotations

from .feature_set import FeatureSet, BANDS, compute_features
from .spectral import band_powers, sef95, peak_alpha_frequency, spectral_entropy
from .dsa import density_spectral_array
from .aeeg import aeeg_envelope
from .suppression import suppression_ratio, burst_suppression
from .asymmetry import asymmetry_index, alpha_delta_ratio
from .background import posterior_dominant_rhythm, continuity_index
from .norms import NormsEngine

__all__ = [
    "FeatureSet",
    "BANDS",
    "compute_features",
    "band_powers",
    "sef95",
    "peak_alpha_frequency",
    "spectral_entropy",
    "density_spectral_array",
    "aeeg_envelope",
    "suppression_ratio",
    "burst_suppression",
    "asymmetry_index",
    "alpha_delta_ratio",
    "posterior_dominant_rhythm",
    "continuity_index",
    "NormsEngine",
]
