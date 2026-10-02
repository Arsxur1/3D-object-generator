"""Pre-registered analysis for increment 9 (docs/preregistration_increment9.md).

Written and committed *before* the test-set results existed, so hypotheses,
breakdowns and the decision rule are applied mechanically to whatever the
evaluation produced. Input: the ``metrics.json`` written by ``neurolens tune``.
"""

from __future__ import annotations

from typing import Any

from scipy.stats import beta

FA_BUDGET = 1.0
REL_SENS_FLOOR = 0.95  # decision rule: margin keeps >= 95% of default sensitivity


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def _population(file: str) -> str:
    return "siena" if file.startswith("PN") else "chbmit"


def _subject(file: str) -> str:
    return file.split("/")[0]


def summarize(records: list[dict], key=lambda f: "all") -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for r in records:
        g = groups.setdefault(key(r["file"]), {"hours": 0.0, "n": 0, "tp": 0, "fp": 0, "lat": []})
        g["hours"] += r["hours"]
        g["n"] += r["n_seizures"]
        g["tp"] += r["tp"]
        g["fp"] += r["fp"]
        g["lat"] += r["latencies_s"]
    out = {}
    for name, g in groups.items():
        lo, hi = clopper_pearson(g["tp"], g["n"])
        lat = sorted(g["lat"])
        out[name] = {
            "hours": round(g["hours"], 2), "seizures": g["n"], "tp": g["tp"], "fp": g["fp"],
            "sensitivity": round(g["tp"] / g["n"], 4) if g["n"] else None,
            "sens_ci95": [round(lo, 3), round(hi, 3)],
            "fa_per_hour": round(g["fp"] / g["hours"], 3) if g["hours"] else None,
            "latency_median_s": lat[len(lat) // 2] if lat else None,
        }
    return out


def assess(metrics: dict) -> dict[str, Any]:
    """Apply H1-H4 and the monitor-default decision rule to a tune metrics.json."""
    res = {r["mode"]: r for r in metrics["results"]}
    tab = {mode: summarize(r["records"])["all"] for mode, r in res.items()}

    def get(mode: str) -> dict:
        if mode not in tab:
            raise KeyError(f"mode {mode!r} missing from metrics (have {sorted(tab)})")
        return tab[mode]

    off_def, off_strict, off_margin = get("offline-default"), get("offline-strict"), get("offline-margin")
    rt_def, rt_margin = get("realtime-default"), get("realtime-margin")

    h = {
        "H1_offline_margin_sens_ge_strict": off_margin["sensitivity"] >= off_strict["sensitivity"],
        "H2_offline_margin_fa_le_budget": off_margin["fa_per_hour"] <= FA_BUDGET,
        "H3_default_most_sensitive_but_over_budget": (
            off_def["sensitivity"] >= max(t["sensitivity"] for m, t in tab.items() if m.startswith("offline"))
            and rt_def["sensitivity"] >= max(t["sensitivity"] for m, t in tab.items() if m.startswith("realtime"))
            and off_def["fa_per_hour"] > FA_BUDGET and rt_def["fa_per_hour"] > FA_BUDGET
        ),
        "H4_realtime_margin_sens_ge_0.9_and_fa_le_budget": (
            rt_margin["sensitivity"] >= 0.9 and rt_margin["fa_per_hour"] <= FA_BUDGET
        ),
    }
    adopt = (rt_margin["sensitivity"] >= REL_SENS_FLOOR * rt_def["sensitivity"]
             and rt_margin["fa_per_hour"] <= FA_BUDGET)
    return {
        "overall": tab,
        "by_population": {m: summarize(r["records"], _population) for m, r in res.items()},
        "by_subject": {m: summarize(r["records"], _subject) for m, r in res.items()},
        "hypotheses": h,
        "decision": {
            "adopt_realtime_margin_as_monitor_default": adopt,
            "criterion": (f"sens(margin) >= {REL_SENS_FLOOR} x sens(default) "
                          f"[{rt_margin['sensitivity']} vs {rt_def['sensitivity']}] and "
                          f"FA/h(margin) <= {FA_BUDGET} [{rt_margin['fa_per_hour']}]"),
        },
    }
