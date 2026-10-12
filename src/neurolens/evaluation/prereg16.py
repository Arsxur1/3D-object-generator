"""Pre-registered analysis for increment 16 (docs/preregistration_increment16.md):
external validation of the frozen IED model ``ied_gbm_v1`` (increment 15) on the IFCN
criteria dataset (Kural et al. 2020) - epileptiform vs benign sharp transients.

Written and committed before any detector was run on these recordings (only file
headers, annotation markers and the gold-standard table were read).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

AUC_MIN = 0.70
AUC_MARGIN_OVER_HEURISTIC = 0.05
SPECIFICITY_MIN = 0.80
# Reference points from the original study (not hypotheses): expert scoring accuracy
# 92%; five IFCN criteria: specificity 95.65%, sensitivity 81.48%.
EXPERT_ACCURACY = 0.92


def assess(summary: dict) -> dict[str, Any]:
    L, H = summary["learned"], summary["heuristic"]
    lf = L["at_frozen_threshold"]
    h = {
        "H1_learned_auc_at_least_0.70_and_ci_above_0.5": L["auc"] >= AUC_MIN and L["auc_ci95"][0] > 0.5,
        "H2_learned_auc_exceeds_heuristic_by_0.05": summary["auc_difference_learned_minus_heuristic"]
        >= AUC_MARGIN_OVER_HEURISTIC,
        "H3_learned_specificity_at_threshold_at_least_0.80": lf["specificity"] >= SPECIFICITY_MIN,
    }
    return {
        "summary": summary,
        "hypotheses": h,
        "decision": {"annotate_heuristic_candidates_with_learned_probability": all(h.values())},
    }


def markdown_report(a: dict) -> str:
    s = a["summary"]
    L, H = s["learned"], s["heuristic"]
    lf, hd = L["at_frozen_threshold"], H["at_default_z"]

    def ci(v):
        return f"{v[0]:.2f}–{v[1]:.2f}"

    rows = ["| Метод | AUC (95% ДИ) | Чувствительность (95% ДИ) | Специфичность (95% ДИ) | Точность |",
            "|---|---|---|---|---|",
            f"| обученный `ied_gbm_v1`, порог 0.95 | {L['auc']:.3f} ({ci(L['auc_ci95'])}) | "
            f"{lf['sensitivity']:.3f} ({ci(lf['sensitivity_ci95'])}) | {lf['specificity']:.3f} "
            f"({ci(lf['specificity_ci95'])}) | {lf['accuracy']:.2f} |",
            f"| эвристический, z = 6 | {H['auc']:.3f} ({ci(H['auc_ci95'])}) | {hd['sensitivity']:.3f} "
            f"({ci(hd['sensitivity_ci95'])}) | {hd['specificity']:.3f} ({ci(hd['specificity_ci95'])}) | "
            f"{hd['accuracy']:.2f} |",
            "| эксперты (исходное исследование) | — | — | — | 0.92 |",
            "| 5 критериев IFCN (исходное исследование) | — | 0.815 | 0.957 | 0.91 |",
            "",
            f"{s['n']} пациентов ({s['n_epileptic']} с эпилепсией, {s['n_non_epileptic']} без); разность AUC "
            f"(обученный − эвристический) {s['auc_difference_learned_minus_heuristic']:+.3f} "
            f"(95% ДИ {ci(s['auc_difference_ci95'])}).",
            "", "Гипотезы:", ""]
    rows += [f"* **{k}** — {'подтверждена' if v else 'отвергнута'}" for k, v in a["hypotheses"].items()]
    rows += ["", "Решение (по заранее заданному правилу H1 ∧ H2 ∧ H3):", ""]
    rows += [f"* {k}: **{'да' if v else 'нет'}**" for k, v in a["decision"].items()]
    return "\n".join(rows) + "\n"


def apply_decision(a: dict, ied_yaml: str = "configs/ied.yaml") -> dict:
    on = a["decision"]["annotate_heuristic_candidates_with_learned_probability"]
    p = Path(ied_yaml)
    text = p.read_text(encoding="utf-8")
    line = f"annotate_candidates: {str(on).lower()}"
    if re.search(r"(?m)^annotate_candidates:", text):
        text = re.sub(r"(?m)^annotate_candidates:.*$", line, text)
    else:
        text = re.sub(r"(?m)^(apply_to_neonates:.*)$",
                      rf"\1\n# increment 16 (docs/increment16_report.md): learned probability as evidence on "
                      rf"heuristic candidates\n{line}", text)
    text = re.sub(r"(?m)^(  report: docs/increment15_report.md)$",
                  r"\1\n  external_validation: docs/increment16_report.md", text) \
        if "external_validation:" not in text else text
    p.write_text(text, encoding="utf-8")
    return {"annotate_candidates": on}
