"""Pre-registered analysis for increment 14 (docs/preregistration_increment14.md):
a sleep stager trained for clinical parasagittal derivations, tested on CAP Sleep
Database recordings never used before (``capsleep.select_test_records_14``).

Written and committed before any test recording (signal data) was downloaded.
Input: ``neurolens sleep cap14`` metrics: {"clinical": scores, "v1": scores,
"rule_based": scores} on the same test records, and the frozen JSON (CV kappa)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

KAPPA_MIN = 0.60
MARGIN_OVER_V1 = 0.05
TRANSFER_TOL = 0.05


def assess(result: dict, frozen: dict, by_group: dict[str, dict] | None = None) -> dict[str, Any]:
    C, V = result["clinical"], result["v1"]
    cv = frozen["cv"]["kappa"]
    h = {
        "H1_clinical_kappa_at_least_0.60": C["kappa"] >= KAPPA_MIN,
        "H2_clinical_beats_v1_by_0.05": C["kappa"] >= V["kappa"] + MARGIN_OVER_V1,
        "H3_transfer_within_0.05_of_cv": abs(C["kappa"] - cv) <= TRANSFER_TOL,
    }
    keep = ("accuracy", "kappa", "macro_f1", "f1", "per_night_kappa_median")
    return {
        "clinical": {k: C[k] for k in keep + ("n_epochs", "n_nights")},
        "v1": {k: V[k] for k in keep},
        "rule_based": {k: result["rule_based"][k] for k in keep},
        "cv_kappa": cv,
        "by_group_kappa": {g: v["kappa"] for g, v in (by_group or {}).items()},
        "hypotheses": h,
        "decision": {"parasagittal_fallback_with_clinical_model":
                     h["H1_clinical_kappa_at_least_0.60"] and h["H2_clinical_beats_v1_by_0.05"]},
    }


def markdown_report(a: dict) -> str:
    rows = ["| Стадировщик | Точность | Каппа | Макро-F1 | F1 W / N1 / N2 / N3 / REM | Медиана каппы по ночам |",
            "|---|---|---|---|---|---|"]
    for name, key in (("клинический (инкремент 14)", "clinical"), ("v1 (Sleep-EDF, инкремент 13)", "v1"),
                      ("правиловый", "rule_based")):
        t = a[key]
        f = " / ".join(f"{t['f1'][s]:.2f}" for s in ("W", "N1", "N2", "N3", "REM"))
        rows.append(f"| {name} | {t['accuracy']:.3f} | {t['kappa']:.3f} | {t['macro_f1']:.3f} | {f} | {t['per_night_kappa_median']:.3f} |")
    c = a["clinical"]
    rows += ["", f"{c['n_nights']} записей, {c['n_epochs']} эпох; каппа кросс-валидации при обучении {a['cv_kappa']:.3f}."]
    if a["by_group_kappa"]:
        rows += ["", "Каппа клинического по группам:", ""] + [f"* {g}: {k:.3f}" for g, k in a["by_group_kappa"].items()]
    rows += ["", "Гипотезы:", ""]
    rows += [f"* **{k}** — {'подтверждена' if v else 'отвергнута'}" for k, v in a["hypotheses"].items()]
    rows += ["", "Решение (по заранее заданному правилу):", ""]
    rows += [f"* {k}: **{'да' if v else 'нет'}**" for k, v in a["decision"].items()]
    return "\n".join(rows) + "\n"


def apply_decision(a: dict, frozen: dict, sleep_yaml: str = "configs/sleep.yaml") -> dict:
    on = a["decision"]["parasagittal_fallback_with_clinical_model"]
    p = Path(sleep_yaml)
    text = p.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^parasagittal_fallback:\s*\S+", f"parasagittal_fallback: {str(on).lower()}", text)
    model = frozen["model"]["path"] if on else "null"
    if re.search(r"(?m)^parasagittal_model:", text):
        text = re.sub(r"(?m)^parasagittal_model:.*$", f"parasagittal_model: {model}", text)
    else:
        text = re.sub(r"(?m)^(parasagittal_fallback:.*)$", rf"\1\nparasagittal_model: {model}", text)
    p.write_text(text, encoding="utf-8")
    return {"parasagittal_fallback": on, "parasagittal_model": model}
