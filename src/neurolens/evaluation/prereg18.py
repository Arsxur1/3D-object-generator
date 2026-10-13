"""Pre-registered analysis for increment 18 (docs/preregistration_increment18.md):
ECG/ocular mimic hypotheses for periodic discharges and the slow-PDR-only slowing rule,
tested on a fresh seeded NMT train-split subset disjoint from the development subset,
against the increment-17 pipeline (commit 286bb2a) on the same recordings.

Written and committed before either pipeline was run on the fresh subset.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .prereg17 import _counts

SPECIFICITY_GAIN_MIN = 0.05
SENSITIVITY_MARGIN = 0.05
SPECIFICITY_MIN = 0.55


def assess(new: dict, old: dict) -> dict[str, Any]:
    N, O = _counts(new["records"]), _counts(old["records"])
    ids_n = sorted(str(r.get("record_id")) for r in new["records"])
    ids_o = sorted(str(r.get("record_id")) for r in old["records"])
    if ids_n != ids_o:
        raise ValueError("the two runs did not score the same recordings")
    h = {
        "H1_specificity_gain_at_least_0.05": N["specificity"] - O["specificity"] >= SPECIFICITY_GAIN_MIN,
        "H2_sensitivity_not_lower_by_more_than_0.05": N["sensitivity"] >= O["sensitivity"] - SENSITIVITY_MARGIN,
        "H3_specificity_at_least_0.55": N["specificity"] >= SPECIFICITY_MIN,
    }
    return {"new": N, "old": O, "hypotheses": h,
            "decision": {"keep_increment18_rules": all(h.values())}}


def markdown_report(a: dict) -> str:
    def row(name, t):
        ci = lambda v: f"{v[0]:.2f}–{v[1]:.2f}"  # noqa: E731
        return (f"| {name} | {t['specificity']:.3f} ({ci(t['specificity_ci95'])}) | "
                f"{t['sensitivity']:.3f} ({ci(t['sensitivity_ci95'])}) | {t['balanced_accuracy']:.3f} |")

    rows = ["| Пайплайн | Специфичность (95% ДИ) | Чувствительность (95% ДИ) | Сбаланс. точность |",
            "|---|---|---|---|",
            row("инкремент 18", a["new"]), row("инкремент 17 (286bb2a)", a["old"]), "",
            f"{a['new']['n_normal']} нормальных и {a['new']['n_abnormal']} аномальных записей.",
            "", "Гипотезы:", ""]
    rows += [f"* **{k}** — {'подтверждена' if v else 'отвергнута'}" for k, v in a["hypotheses"].items()]
    rows += ["", "Решение (по заранее заданному правилу H1 ∧ H2 ∧ H3):", ""]
    rows += [f"* {k}: **{'да' if v else 'нет'}**" for k, v in a["decision"].items()]
    return "\n".join(rows) + "\n"


def apply_decision(a: dict, impression_yaml: str = "configs/impression.yaml",
                   thresholds_yaml: str = "configs/thresholds.yaml") -> dict:
    keep = a["decision"]["keep_increment18_rules"]
    if keep:
        N = a["new"]
        lines = [
            "# What the report may claim about its own normal/abnormal conclusion on routine EEG.",
            "# Written by evaluation/prereg18.apply_decision (docs/preregistration_increment18.md).",
            f"# Fresh NMT subset ({N['n_normal']} normal, {N['n_abnormal']} abnormal): specificity "
            f"{N['specificity']:.3f}, sensitivity {N['sensitivity']:.3f} (docs/increment18_report.md);",
            "# increment 17 (NMT evaluation split): 0.589 / 0.844.",
            "routine_validated: true",
            f"specificity: {N['specificity']:.3f}",
            f"sensitivity: {N['sensitivity']:.3f}",
            "provenance:",
            "  preregistration: docs/preregistration_increment18.md",
            "  report: docs/increment18_report.md",
            "  previous: docs/increment17_report.md",
        ]
        Path(impression_yaml).write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        p = Path(thresholds_yaml)
        text = re.sub(r"(?m)^pd_mimic_uncertain:.*$", "pd_mimic_uncertain: false", p.read_text(encoding="utf-8"))
        p.write_text(text, encoding="utf-8")
    return {"keep_increment18_rules": keep}
