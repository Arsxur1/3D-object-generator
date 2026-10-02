"""Pre-registered analysis for increment 10 (docs/preregistration_increment10.md).

Committed before the test data were downloaded; applied unchanged to the
``metrics.json`` written by ``neurolens ml-eval``.
"""

from __future__ import annotations

from typing import Any

from .prereg import _population, _subject, summarize

SENS_MARGIN = 0.05  # non-inferiority margin for sensitivity
QUIET_FA = 1.0
QUIET_SENS = 0.6


def macro_sensitivity(records: list[dict]) -> float:
    """Mean over patients of per-patient sensitivity (each patient weighs equally)."""
    per = summarize(records, _subject)
    vals = [g["sensitivity"] for g in per.values() if g["seizures"]]
    return round(sum(vals) / len(vals), 4) if vals else float("nan")


def assess(metrics: dict) -> dict[str, Any]:
    res = {r["mode"]: r for r in metrics["results"]}
    tab = {m: summarize(r["records"])["all"] for m, r in res.items()}
    macro = {m: macro_sensitivity(r["records"]) for m, r in res.items()}

    def noninferior(ml: str, ref: str) -> bool:
        return (tab[ml]["sensitivity"] >= tab[ref]["sensitivity"] - SENS_MARGIN
                and tab[ml]["fa_per_hour"] <= tab[ref]["fa_per_hour"])

    h = {
        "H1_realtime_mlA_noninferior_sens_and_no_more_FA": noninferior("realtime-ml-A", "realtime-threshold-default"),
        "H2_offline_mlA_noninferior_sens_and_no_more_FA": noninferior("offline-ml-A", "offline-threshold-default"),
        "H3_realtime_mlB_quiet_FA_le_1_and_sens_ge_0.6": (
            tab["realtime-ml-B"]["fa_per_hour"] <= QUIET_FA and tab["realtime-ml-B"]["sensitivity"] >= QUIET_SENS),
        "H4_realtime_mlA_macro_sens_noninferior": (
            macro["realtime-ml-A"] >= macro["realtime-threshold-default"] - SENS_MARGIN),
    }
    return {
        "overall": tab,
        "macro_sensitivity": macro,
        "by_population": {m: summarize(r["records"], _population) for m, r in res.items()},
        "by_subject": {m: summarize(r["records"], _subject) for m, r in res.items()},
        "hypotheses": h,
        "decision": {
            "ml_becomes_default_realtime_seizure_detector": h["H1_realtime_mlA_noninferior_sens_and_no_more_FA"],
            "ml_becomes_default_offline_ictal_detector": h["H2_offline_mlA_noninferior_sens_and_no_more_FA"],
            "quiet_mode_B_offered": h["H3_realtime_mlB_quiet_FA_le_1_and_sens_ge_0.6"],
        },
    }
