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


# ---------------------------------------------------------------------------
# mechanical reporting and decision application (written before results)
# ---------------------------------------------------------------------------

_MODES = [
    ("offline-threshold-default", "оффлайн", "пороговый, дефолт"),
    ("offline-ml-A", "оффлайн", "обученный, точка A"),
    ("offline-ml-B", "оффлайн", "обученный, точка B"),
    ("realtime-threshold-default", "real-time", "пороговый, дефолт"),
    ("realtime-ml-A", "real-time", "обученный, точка A"),
    ("realtime-ml-B", "real-time", "обученный, точка B"),
]


def _ci(t: dict) -> str:
    lo, hi = t["sens_ci95"]
    return f"{t['sensitivity']:.2f} ({lo:.2f}–{hi:.2f})"


def markdown_report(a: dict) -> str:
    """Result tables (RU) from an ``assess`` output — no hand-copied numbers."""
    o, mac = a["overall"], a["macro_sensitivity"]
    L = ["| Режим | Детектор | Найдено | Чувствительность (95% ДИ) | Макро-чувств. | FA/ч | Латентность, медиана, с |",
         "|---|---|---|---|---|---|---|"]
    for key, mode, name in _MODES:
        if key in o:
            t = o[key]
            L.append(f"| {mode} | {name} | {t['tp']}/{t['seizures']} | {_ci(t)} | {mac[key]:.2f} "
                     f"| {t['fa_per_hour']:.2f} | {t['latency_median_s']} |")
    pops = sorted({p for m in a["by_population"].values() for p in m})
    L += ["", "По популяциям (найдено / приступов, FA/ч):", "",
          "| Режим | Детектор | " + " | ".join(pops) + " |", "|---|---|" + "---|" * len(pops)]
    for key, mode, name in _MODES:
        if key in a["by_population"]:
            g = a["by_population"][key]
            cells = [f"{g[p]['tp']}/{g[p]['seizures']}, {g[p]['fa_per_hour']:.1f}" if p in g else "—" for p in pops]
            L.append(f"| {mode} | {name} | " + " | ".join(cells) + " |")
    subs = sorted({s for m in a["by_subject"].values() for s in m})
    keys = [k for k, _, _ in _MODES if k in a["by_subject"]]
    L += ["", "По пациентам (найдено / приступов, FA/ч):", "",
          "| Пациент | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for s in subs:
        cells = []
        for k in keys:
            g = a["by_subject"][k].get(s)
            cells.append(f"{g['tp']}/{g['seizures']}, {g['fa_per_hour']:.1f}" if g else "—")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    L += ["", "Гипотезы:", ""]
    for name, ok in a["hypotheses"].items():
        L.append(f"* **{name}** — {'подтверждена' if ok else 'отвергнута'}")
    L += ["", "Решение (по заранее заданным правилам):", ""]
    for name, ok in a["decision"].items():
        L.append(f"* {name}: **{'да' if ok else 'нет'}**")
    return "\n".join(L) + "\n"


def apply_decision(a: dict, ml_yaml: str = "configs/ml.yaml") -> dict:
    """Set ``enabled`` in configs/ml.yaml exactly as the pre-registered rules say
    (H1 -> real-time default, H2 -> offline default). Comments are preserved:
    only the ``enabled:`` line inside each top-level section is rewritten.
    Returns the flags written."""
    import re
    from pathlib import Path

    flags = {"offline": bool(a["decision"]["ml_becomes_default_offline_ictal_detector"]),
             "realtime": bool(a["decision"]["ml_becomes_default_realtime_seizure_detector"])}
    p = Path(ml_yaml)
    out, section = [], None
    for line in p.read_text(encoding="utf-8").splitlines(keepends=True):
        top = re.match(r"^([A-Za-z_]\w*):", line)
        if top:
            section = top.group(1)
        elif section in flags:
            m = re.match(r"^(\s+enabled:\s*)(true|false)\b(.*)$", line.rstrip("\n"))
            if m:
                line = f"{m.group(1)}{str(flags[section]).lower()}{m.group(3)}\n"
        out.append(line)
    p.write_text("".join(out), encoding="utf-8")
    return flags
