"""Pre-registered analysis for increment 13 (docs/preregistration_increment13.md):
learned AASM sleep staging on held-out Sleep-EDF subjects.

Written and committed before any test-half recording was downloaded.
Input: the ``sleep_metrics.json`` of ``neurolens sleep-eval --split test``
({"learned": scores, "rule_based": scores}) and the frozen JSON (development
cross-validation kappa)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

KAPPA_MIN = 0.70
MARGIN_OVER_RULE = 0.10
TRANSFER_TOL = 0.05
MACRO_F1_MIN = 0.65


def assess(test: dict, frozen: dict) -> dict[str, Any]:
    L, R = test["learned"], test["rule_based"]
    cv = frozen["dev_cv"]["kappa"]
    h = {
        "H1_kappa_at_least_0.70": L["kappa"] >= KAPPA_MIN,
        "H2_beats_rule_based_by_0.10_kappa": L["kappa"] >= R["kappa"] + MARGIN_OVER_RULE,
        "H3_transfer_within_0.05_of_dev_cv": abs(L["kappa"] - cv) <= TRANSFER_TOL,
        "H4_macro_f1_at_least_0.65": L["macro_f1"] >= MACRO_F1_MIN,
    }
    enable = h["H1_kappa_at_least_0.70"] and h["H2_beats_rule_based_by_0.10_kappa"]
    return {
        "learned": {k: L[k] for k in ("accuracy", "kappa", "macro_f1", "f1", "per_night_kappa_median",
                                      "n_epochs", "n_nights", "n_subjects")},
        "rule_based": {k: R[k] for k in ("accuracy", "kappa", "macro_f1", "f1", "per_night_kappa_median")},
        "dev_cv_kappa": cv,
        "hypotheses": h,
        "decision": {"sleep_staging_enabled_by_default": enable},
    }


def markdown_report(a: dict) -> str:
    L, R = a["learned"], a["rule_based"]
    rows = ["| Стадировщик | Точность | Каппа Коэна | Макро-F1 | F1 W / N1 / N2 / N3 / REM | Медиана каппы по ночам |",
            "|---|---|---|---|---|---|"]
    for name, t in (("обученный (заморожен)", L), ("правиловый (без обучения)", R)):
        f = " / ".join(f"{t['f1'][s]:.2f}" for s in ("W", "N1", "N2", "N3", "REM"))
        rows.append(f"| {name} | {t['accuracy']:.3f} | {t['kappa']:.3f} | {t['macro_f1']:.3f} | {f} | {t['per_night_kappa_median']:.3f} |")
    rows += ["", f"Тест: {L['n_nights']} ночей, {L['n_subjects']} испытуемых, {L['n_epochs']} эпох. "
             f"Каппа кросс-валидации на развивающей половине: {a['dev_cv_kappa']:.3f}.", "", "Гипотезы:", ""]
    rows += [f"* **{k}** — {'подтверждена' if v else 'отвергнута'}" for k, v in a["hypotheses"].items()]
    rows += ["", "Решение (по заранее заданному правилу):", ""]
    rows += [f"* {k}: **{'да' if v else 'нет'}**" for k, v in a["decision"].items()]
    return "\n".join(rows) + "\n"


def apply_decision(a: dict, frozen: dict, sleep_yaml: str = "configs/sleep.yaml") -> dict:
    """Rewrite ``enabled`` and ``model`` in configs/sleep.yaml (comments kept)."""
    plan = {"enabled": a["decision"]["sleep_staging_enabled_by_default"], "model": frozen["model"]["path"]}
    p = Path(sleep_yaml)
    out = []
    for line in p.read_text(encoding="utf-8").splitlines(keepends=True):
        m = re.match(r"^(enabled|model):(\s*)([^#\n]*?)(\s*#.*)?$", line.rstrip("\n"))
        if m:
            v = plan[m.group(1)]
            v = str(v).lower() if isinstance(v, bool) else v
            line = f"{m.group(1)}:{m.group(2) or ' '}{v}{m.group(4) or ''}\n"
        out.append(line)
    p.write_text("".join(out), encoding="utf-8")
    return plan
