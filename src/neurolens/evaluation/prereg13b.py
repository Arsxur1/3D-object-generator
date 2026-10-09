"""Pre-registered analysis for increment 13b (docs/preregistration_increment13b.md):
zero-shot transfer of the frozen Sleep-EDF stager to clinical PSG (CAP Sleep
Database) with derivations approximated from the parasagittal bipolar chain.

Written and committed before any CAP recording (signal data) was downloaded."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

KAPPA_MIN = 0.60
MARGIN_OVER_RULE = 0.10


def assess(result: dict, by_group: dict[str, dict]) -> dict[str, Any]:
    L, R = result["learned"], result["rule_based"]
    h = {
        "H1_kappa_at_least_0.60": L["kappa"] >= KAPPA_MIN,
        "H2_beats_rule_based_by_0.10_kappa": L["kappa"] >= R["kappa"] + MARGIN_OVER_RULE,
    }
    return {
        "learned": {k: L[k] for k in ("accuracy", "kappa", "macro_f1", "f1", "per_night_kappa_median",
                                      "n_epochs", "n_nights")},
        "rule_based": {k: R[k] for k in ("accuracy", "kappa", "macro_f1", "f1", "per_night_kappa_median")},
        "by_group_kappa": {g: v["kappa"] for g, v in by_group.items()},
        "hypotheses": h,
        "decision": {"parasagittal_fallback": all(h.values())},
    }


def markdown_report(a: dict) -> str:
    L, R = a["learned"], a["rule_based"]
    rows = ["| Стадировщик | Точность | Каппа | Макро-F1 | F1 W / N1 / N2 / N3 / REM | Медиана каппы по ночам |",
            "|---|---|---|---|---|---|"]
    for name, t in (("обученный (Sleep-EDF, без дообучения)", L), ("правиловый", R)):
        f = " / ".join(f"{t['f1'][s]:.2f}" for s in ("W", "N1", "N2", "N3", "REM"))
        rows.append(f"| {name} | {t['accuracy']:.3f} | {t['kappa']:.3f} | {t['macro_f1']:.3f} | {f} | {t['per_night_kappa_median']:.3f} |")
    rows += ["", f"{L['n_nights']} записей, {L['n_epochs']} эпох.", "", "Каппа по группам (обученный):", ""]
    rows += [f"* {g}: {k:.3f}" for g, k in a["by_group_kappa"].items()]
    rows += ["", "Гипотезы:", ""]
    rows += [f"* **{k}** — {'подтверждена' if v else 'отвергнута'}" for k, v in a["hypotheses"].items()]
    rows += ["", "Решение (по заранее заданному правилу):", ""]
    rows += [f"* {k}: **{'да' if v else 'нет'}**" for k, v in a["decision"].items()]
    return "\n".join(rows) + "\n"


def apply_decision(a: dict, sleep_yaml: str = "configs/sleep.yaml") -> bool:
    value = a["decision"]["parasagittal_fallback"]
    p = Path(sleep_yaml)
    text = re.sub(r"(?m)^parasagittal_fallback:\s*\S+", f"parasagittal_fallback: {str(value).lower()}",
                  p.read_text(encoding="utf-8"))
    p.write_text(text, encoding="utf-8")
    return value
