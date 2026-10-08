"""Pre-registered analysis for increment 12 (docs/preregistration_increment12.md):
seizure detection in neonates (Helsinki dataset, held-out half).

Written and committed before any neonatal EEG was downloaded. Modes in the
``metrics.json`` of ``neurolens ml-eval --db helsinki --compare ... --analysis prereg12``:
*-threshold-default (rule-based detector), *-ml-A (neonatal model, frozen in
increment 12), *-cmp-A (general v2 model, increment 11, zero-shot on neonates).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .prereg import _subject, summarize
from .prereg10 import macro_sensitivity

SENS_MARGIN = 0.05
PATHS = ("offline", "realtime")


def _noninferior(tab: dict, a: str, b: str) -> bool:
    return (tab[a]["sensitivity"] >= tab[b]["sensitivity"] - SENS_MARGIN
            and tab[a]["fa_per_hour"] <= tab[b]["fa_per_hour"])


def _overlaps(t0: float, t1: float, ivs) -> bool:
    return any(t0 < b and a < t1 for a, b in ivs)


def fa_excluding_ambiguous(records: list[dict], ambiguous: dict[str, list[tuple[float, float]]]) -> float:
    """Secondary: FA/h when detections overlapping a region marked by one or two
    (not all three) experts are not counted as false alarms."""
    fp = hours = 0.0
    for r in records:
        amb = ambiguous.get(r["file"], [])
        fp += sum(1 for d in r.get("false_positives", []) if not _overlaps(d[0], d[1], amb))
        hours += r["hours"]
    return round(fp / hours, 3) if hours else float("nan")


def choose(tab: dict, path: str) -> str:
    """Pre-registered rule per path: neonatal model if non-inferior to the threshold
    detector (and, when the general model also is, not inferior to it); else the
    general model if non-inferior to the threshold detector; else threshold."""
    thr, neo, gen = f"{path}-threshold-default", f"{path}-ml-A", f"{path}-cmp-A"
    neo_ok, gen_ok = _noninferior(tab, neo, thr), _noninferior(tab, gen, thr)
    if neo_ok and (not gen_ok or _noninferior(tab, neo, gen)):
        return "neonatal"
    if gen_ok:
        return "general"
    return "threshold"


def assess(metrics: dict, ambiguous: dict[str, list[tuple[float, float]]] | None = None) -> dict[str, Any]:
    res = {r["mode"]: r for r in metrics["results"]}
    tab = {m: summarize(r["records"])["all"] for m, r in res.items()}
    macro = {m: macro_sensitivity(r["records"]) for m, r in res.items()}
    h = {
        "H1_offline_neonatal_noninferior_to_threshold": _noninferior(tab, "offline-ml-A", "offline-threshold-default"),
        "H2_realtime_neonatal_noninferior_to_threshold": _noninferior(tab, "realtime-ml-A", "realtime-threshold-default"),
        "H3_offline_general_noninferior_to_threshold": _noninferior(tab, "offline-cmp-A", "offline-threshold-default"),
        "H4_realtime_general_noninferior_to_threshold": _noninferior(tab, "realtime-cmp-A", "realtime-threshold-default"),
    }
    out = {
        "overall": tab,
        "macro_sensitivity": macro,
        "by_subject": {m: summarize(r["records"], _subject) for m, r in res.items()},
        "hypotheses": h,
        "secondary": {
            "neonatal_vs_general_noninferior": {p: _noninferior(tab, f"{p}-ml-A", f"{p}-cmp-A") for p in PATHS},
        },
        "decision": {f"{p}_neonatal": choose(tab, p) for p in PATHS},
    }
    if ambiguous is not None:
        out["secondary"]["fa_per_hour_excluding_ambiguous"] = {
            m: fa_excluding_ambiguous(r["records"], ambiguous) for m, r in res.items()}
    return out


_MODES = [
    ("offline-threshold-default", "оффлайн", "пороговый, дефолт"),
    ("offline-cmp-A", "оффлайн", "общая модель v2 (без обучения на новорождённых)"),
    ("offline-ml-A", "оффлайн", "неонатальная модель, A"),
    ("realtime-threshold-default", "real-time", "пороговый, дефолт"),
    ("realtime-cmp-A", "real-time", "общая модель v2 (без обучения на новорождённых)"),
    ("realtime-ml-A", "real-time", "неонатальная модель, A"),
]


def markdown_report(a: dict) -> str:
    lines = ["| Режим | Детектор | Найдено | Чувствительность (95% ДИ) | Макро-чувств. | FA/ч | Латентность, медиана, с |",
             "|---|---|---|---|---|---|---|"]
    for mode, path, name in _MODES:
        t = a["overall"].get(mode)
        if t is None:
            continue
        lo, hi = t["sens_ci95"]
        lines.append(f"| {path} | {name} | {t['tp']}/{t['seizures']} | {t['sensitivity']:.2f} ({lo:.2f}–{hi:.2f}) "
                     f"| {a['macro_sensitivity'][mode]:.2f} | {t['fa_per_hour']:.2f} | {t['latency_median_s']} |")
    amb = a["secondary"].get("fa_per_hour_excluding_ambiguous")
    if amb:
        lines += ["", "Вторичное: FA/ч без детекций в зонах, отмеченных не всеми экспертами:", ""]
        lines += [f"* {m}: {v:.2f}" for m, v in amb.items()]
    lines += ["", "Гипотезы:", ""]
    lines += [f"* **{k}** — {'подтверждена' if v else 'отвергнута'}" for k, v in a["hypotheses"].items()]
    lines += ["", "Решение (по заранее заданным правилам):", ""]
    lines += [f"* {k}: **{v}**" for k, v in a["decision"].items()]
    return "\n".join(lines) + "\n"


def apply_decision(a: dict, frozen12: dict, ml_yaml: str = "configs/ml.yaml") -> dict:
    """Write the ``neonatal:`` section of configs/ml.yaml (replacing any previous
    one); the general offline/realtime sections are untouched."""
    pol: dict[str, Any] = {}
    for p in PATHS:
        choice = a["decision"][f"{p}_neonatal"]
        pol[p] = choice
        if choice == "neonatal":
            src = frozen12[p]
            prm = src["A_replacement"]["params"]
            pol[f"{p}_model"] = {"enabled": True, "model": src["model"]["path"],
                                 "threshold": prm["threshold"], "min_epochs": prm["min_epochs"],
                                 "calibration": src.get("calibration", {}).get("path")}
    block = ["neonatal:  # seizure detector for neonates (increment 12, evaluation/prereg12.apply_decision)\n"]
    for p in PATHS:
        block.append(f"  {p}: {pol[p]}\n")
    for p in PATHS:
        if f"{p}_model" in pol:
            block.append(f"  {p}_model:\n")
            for k, v in pol[f"{p}_model"].items():
                if v is None:
                    continue
                v = str(v).lower() if isinstance(v, bool) else v
                block.append(f"    {k}: {v}\n")
    path = Path(ml_yaml)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    out, skip = [], False
    for line in lines:
        if re.match(r"^neonatal:", line):
            skip = True
            continue
        if skip and re.match(r"^\S", line):
            skip = False
        if not skip:
            out.append(line)
    idx = next((i for i, l in enumerate(out) if l.startswith("provenance:")), len(out))
    out[idx:idx] = block
    path.write_text("".join(out), encoding="utf-8")
    return pol
