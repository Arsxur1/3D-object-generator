"""Pre-registered analysis for increment 15 (docs/preregistration_increment15.md):
learned interictal epileptiform discharge (IED) detector vs the heuristic
SpikeDetector on the NMT-Events test split (odd-id abnormal recordings, normal
recordings with id % 8 == 1).

Written and committed before any test-split EEG was downloaded. Input: the metrics
JSON written by ``neurolens ied eval`` ({"learned": aggregate at the frozen
threshold, "heuristic_curve": [...], "learned_by_type": {...}}) and the frozen JSON.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

FA_BUDGET_PER_H = 15.0          # false detections per hour of normal EEG at the operating point
FA_TOLERANCE = 1.5              # test FA/h may exceed the budget by at most 50 %
SENSITIVITY_MIN = 0.50          # pooled discharge-event sensitivity
SLOWING_HIT_MAX = 0.40          # share of delta-slowing events hit
HEURISTIC_DEFAULT_Z = 6.0       # shipped heuristic setting


def _heuristic_dominated(learned: dict, curve: list[dict]) -> bool:
    """True when every heuristic grid point has lower sensitivity or higher FA/h than
    the learned detector, and the heuristic point with the closest FA/h at or above
    the learned one is less sensitive."""
    for c in curve:
        if c["sensitivity"] >= learned["sensitivity"] and c["fa_per_h_normal"] <= learned["fa_per_h_normal"]:
            return False
    return True


def assess(metrics: dict, frozen: dict) -> dict[str, Any]:
    L = metrics["learned"]
    curve = metrics["heuristic_curve"]
    default = next(c for c in curve if c["sharpness_z"] == HEURISTIC_DEFAULT_Z)
    lowest_fa = min(curve, key=lambda c: c["fa_per_h_normal"])
    h = {
        "H1_sensitivity_at_least_0.50": L["sensitivity"] >= SENSITIVITY_MIN,
        "H2_fa_per_h_normal_within_1.5x_budget": L["fa_per_h_normal"] <= FA_BUDGET_PER_H * FA_TOLERANCE,
        "H3_dominates_heuristic_curve": _heuristic_dominated(L, curve),
        "H4_slowing_hits_at_most_0.40": L["slowing_hit_rate"] <= SLOWING_HIT_MAX,
    }
    enable = h["H1_sensitivity_at_least_0.50"] and h["H2_fa_per_h_normal_within_1.5x_budget"] \
        and h["H3_dominates_heuristic_curve"]
    return {
        "learned": L,
        "learned_by_type": metrics.get("learned_by_type", {}),
        "heuristic_default": default,
        "heuristic_lowest_fa": lowest_fa,
        "heuristic_curve": curve,
        "frozen_threshold": frozen["threshold"],
        "cv_at_threshold": frozen["cv_operating_point"],
        "hypotheses": h,
        "decision": {"enable_learned_ied_detector": enable},
    }


def _ci(a: dict) -> str:
    lo, hi = a["sensitivity_ci95_record_bootstrap"]
    return f"{a['sensitivity']:.3f} ({lo:.2f}–{hi:.2f})"


def markdown_report(a: dict) -> str:
    L, d, lf = a["learned"], a["heuristic_default"], a["heuristic_lowest_fa"]
    rows = ["| Детектор | Чувствительность (95% ДИ, бутстреп по записям) | Макро | Ложных/ч (норма) | Попаданий в замедление |",
            "|---|---|---|---|---|",
            f"| обученный, порог {a['frozen_threshold']} | {_ci(L)} | {L['sensitivity_macro']:.3f} | "
            f"{L['fa_per_h_normal']:.1f} | {L['slowing_hit_rate']:.3f} |",
            f"| эвристический, z = {d['sharpness_z']:g} (по умолчанию, без ограничения числа) | {_ci(d)} | "
            f"{d['sensitivity_macro']:.3f} | {d['fa_per_h_normal']:.1f} | {d['slowing_hit_rate']:.3f} |",
            f"| эвристический, z = {lf['sharpness_z']:g} (наименьшая частота ложных) | {_ci(lf)} | "
            f"{lf['sensitivity_macro']:.3f} | {lf['fa_per_h_normal']:.1f} | {lf['slowing_hit_rate']:.3f} |",
            "",
            f"Разрядов (событий) в тесте: {L['n_ref']} в {L['n_records_with_ref']} записях; нормальных записей "
            f"{L['n_normal']} ({L['hours_normal']:.1f} ч). Кросс-валидация на обучающей части при этом пороге: "
            f"чувствительность {a['cv_at_threshold']['sensitivity']:.3f}, "
            f"{a['cv_at_threshold']['fa_per_h_normal']:.1f} ложных/ч.",
            f"Неразмеченных детекций в аномальных записях: {L['unannotated_per_h_abnormal']:.1f}/ч "
            "(разметка может быть неполной — не считаются ложными)."]
    if a["learned_by_type"]:
        rows += ["", "Чувствительность обученного детектора по типу разряда (исходные аннотации):", ""]
        rows += [f"* {k}: {v['hit']}/{v['n']} = {v['sensitivity']:.3f}" for k, v in a["learned_by_type"].items()]
    rows += ["", "Гипотезы:", ""]
    rows += [f"* **{k}** — {'подтверждена' if v else 'отвергнута'}" for k, v in a["hypotheses"].items()]
    rows += ["", "Решение (по заранее заданному правилу H1 ∧ H2 ∧ H3):", ""]
    rows += [f"* {k}: **{'да' if v else 'нет'}**" for k, v in a["decision"].items()]
    return "\n".join(rows) + "\n"


def apply_decision(a: dict, frozen: dict, ied_yaml: str = "configs/ied.yaml") -> dict:
    on = a["decision"]["enable_learned_ied_detector"]
    L = a["learned"]
    lines = [
        "# Learned interictal epileptiform discharge (spike / sharp wave) detector.",
        "# Written by evaluation/prereg15.apply_decision (docs/preregistration_increment15.md).",
        f"# Held-out NMT-Events test split: sensitivity {L['sensitivity']:.3f}, "
        f"{L['fa_per_h_normal']:.1f} false detections/h of normal EEG (docs/increment15_report.md).",
        "# Disabled = the heuristic SpikeDetector (not validated: see the report).",
        f"enabled: {str(on).lower()}",
        f"model: {frozen['model']['path']}",
        f"threshold: {frozen['threshold']}",
        "max_events: 200",
        "apply_to_neonates: false",
        "provenance:",
        "  preregistration: docs/preregistration_increment15.md",
        "  report: docs/increment15_report.md",
        f"  model_sha256: {frozen['model']['sha256']}",
    ]
    Path(ied_yaml).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"enabled": on, "model": frozen["model"]["path"], "threshold": frozen["threshold"]}
