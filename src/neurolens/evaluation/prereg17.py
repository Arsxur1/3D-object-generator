"""Pre-registered analysis for increment 17 (docs/preregistration_increment17.md):
the report's normal/abnormal conclusion on routine EEG, NMT Scalp EEG Dataset,
authors' evaluation split (95 normal, 90 abnormal), after the routine-EEG fixes vs the
pipeline before them (commit 4519669) on the same recordings.

Written and committed before either pipeline was run on the evaluation split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SPECIFICITY_MIN = 0.45
SPECIFICITY_GAIN_MIN = 0.30
SENSITIVITY_MIN = 0.70
DETECTOR_LIMITS = {"burst_suppression": 0.05, "grda": 0.05, "lpds": 0.15, "triphasic_waves": 0.05,
                   "firda": 0.15}


def _counts(rows: list[dict]) -> dict:
    from .prereg import clopper_pearson

    nm = [r for r in rows if not r["abnormal"]]
    ab = [r for r in rows if r["abnormal"]]
    tn = sum(not r["impression_abnormal"] for r in nm)
    tp = sum(r["impression_abnormal"] for r in ab)
    return {"n_normal": len(nm), "n_abnormal": len(ab), "tn": tn, "tp": tp,
            "specificity": tn / len(nm), "specificity_ci95": list(clopper_pearson(tn, len(nm))),
            "sensitivity": tp / len(ab), "sensitivity_ci95": list(clopper_pearson(tp, len(ab))),
            "balanced_accuracy": (tn / len(nm) + tp / len(ab)) / 2}


def assess(new: dict, old: dict) -> dict[str, Any]:
    """``new`` / ``old``: JSON written by ``neurolens routine eval`` (records + summary)."""
    N, O = _counts(new["records"]), _counts(old["records"])
    rates = new["summary"]["detector_rate"]
    det = {c: rates.get(c, {"normal": 0.0})["normal"] for c in DETECTOR_LIMITS}
    h = {
        "H1_specificity_at_least_0.45": N["specificity"] >= SPECIFICITY_MIN,
        "H2_specificity_gain_at_least_0.30": N["specificity"] - O["specificity"] >= SPECIFICITY_GAIN_MIN,
        "H3_sensitivity_at_least_0.70": N["sensitivity"] >= SENSITIVITY_MIN,
        "H4_detector_false_firing_within_limits": all(det[c] <= lim for c, lim in DETECTOR_LIMITS.items()),
    }
    return {"new": N, "old": O, "normal_detector_rate": det, "detector_limits": DETECTOR_LIMITS,
            "hypotheses": h,
            "decision": {"routine_impression_validated": h["H1_specificity_at_least_0.45"]
                         and h["H2_specificity_gain_at_least_0.30"] and h["H3_sensitivity_at_least_0.70"]}}


def markdown_report(a: dict) -> str:
    def row(name, t):
        ci = lambda v: f"{v[0]:.2f}–{v[1]:.2f}"  # noqa: E731
        return (f"| {name} | {t['specificity']:.3f} ({ci(t['specificity_ci95'])}) | "
                f"{t['sensitivity']:.3f} ({ci(t['sensitivity_ci95'])}) | {t['balanced_accuracy']:.3f} |")

    rows = ["| Пайплайн | Специфичность (95% ДИ) | Чувствительность (95% ДИ) | Сбаланс. точность |",
            "|---|---|---|---|",
            row("после исправлений (инкремент 17)", a["new"]),
            row("до исправлений (коммит 4519669)", a["old"]),
            "",
            f"{a['new']['n_normal']} нормальных и {a['new']['n_abnormal']} аномальных записей (разметка неврологов).",
            "", "Частота срабатывания в нормальных записях (после исправлений):", ""]
    rows += [f"* {c}: {v:.3f} (предел {a['detector_limits'][c]})" for c, v in a["normal_detector_rate"].items()]
    rows += ["", "Гипотезы:", ""]
    rows += [f"* **{k}** — {'подтверждена' if v else 'отвергнута'}" for k, v in a["hypotheses"].items()]
    rows += ["", "Решение (по заранее заданному правилу H1 ∧ H2 ∧ H3):", ""]
    rows += [f"* {k}: **{'да' if v else 'нет'}**" for k, v in a["decision"].items()]
    return "\n".join(rows) + "\n"


def apply_decision(a: dict, path: str = "configs/impression.yaml") -> dict:
    on = a["decision"]["routine_impression_validated"]
    lines = [
        "# What the report may claim about its own normal/abnormal conclusion on routine EEG.",
        "# Written by evaluation/prereg17.apply_decision (docs/preregistration_increment17.md).",
        f"# NMT evaluation split: specificity {a['new']['specificity']:.3f}, sensitivity "
        f"{a['new']['sensitivity']:.3f} (docs/increment17_report.md).",
        f"routine_validated: {str(on).lower()}",
    ]
    if on:
        lines += [f"specificity: {a['new']['specificity']:.3f}", f"sensitivity: {a['new']['sensitivity']:.3f}"]
    lines += ["provenance:", "  preregistration: docs/preregistration_increment17.md",
              "  report: docs/increment17_report.md"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"routine_validated": on}
