"""Age / maturational norms engine (TZ §6, first-class).

Selects the applicable normative band by age (or postmenstrual age for
neonates) and answers whether a measured PDR is within the expected range.
Norms come from ``configs/norms/age_norms.yaml`` — nothing hardcoded.
"""

from __future__ import annotations

from typing import Any, Optional

from ..contracts.signal import PatientInfo


class NormsEngine:
    def __init__(self, norms: dict[str, Any]):
        self.norms = norms

    def band_for_patient(self, patient: PatientInfo) -> tuple[str, dict[str, Any]]:
        """Return (band_name, band_dict) applicable to the patient.

        Neonatal selection uses postmenstrual_age_weeks; otherwise age_years.
        Falls back to 'adult' when age is unknown (with low confidence upstream).
        """
        if patient.postmenstrual_age_weeks is not None:
            neo = self.norms.get("neonate")
            if neo:
                return "neonate", neo

        age = patient.age_years
        if age is None:
            return "adult", self.norms.get("adult", {})

        for name in ("adult", "child_school", "child_early"):
            band = self.norms.get(name)
            if not band:
                continue
            lo = band.get("min_age_years", -1)
            hi = band.get("max_age_years", 200)
            if lo <= age <= hi:
                return name, band
        return "adult", self.norms.get("adult", {})

    def pdr_expected_range(self, patient: PatientInfo) -> Optional[tuple[float, float]]:
        _, band = self.band_for_patient(patient)
        rng = band.get("posterior_dominant_rhythm_hz")
        if rng and len(rng) == 2:
            return float(rng[0]), float(rng[1])
        return None

    def pdr_is_normal(self, pdr_hz: float, patient: PatientInfo) -> Optional[bool]:
        rng = self.pdr_expected_range(patient)
        if rng is None:
            return None
        return rng[0] <= pdr_hz <= rng[1]

    # -- neonatal maturational expectations (TZ §6) ----------------------
    def neonatal_expectations(self, patient: PatientInfo) -> Optional[dict[str, Any]]:
        """Return the maturational-norm band applicable to the patient's PMA."""
        pma = patient.postmenstrual_age_weeks
        if pma is None:
            return None
        bands = self.norms.get("neonate", {}).get("maturation", [])
        for band in bands:
            if pma <= band.get("pma_max", 100):
                return band
        return bands[-1] if bands else None

    def is_continuity_normal_for_pma(self, continuity: float, patient: PatientInfo) -> Optional[bool]:
        exp = self.neonatal_expectations(patient)
        if exp is None:
            return None
        return continuity >= float(exp.get("continuity_normal_min", 0.0))

    def expected_aeeg_category(self, patient: PatientInfo) -> Optional[str]:
        exp = self.neonatal_expectations(patient)
        return exp.get("expected_aeeg") if exp else None

    def swc_expected(self, patient: PatientInfo) -> Optional[bool]:
        exp = self.neonatal_expectations(patient)
        return bool(exp.get("swc_expected")) if exp else None
