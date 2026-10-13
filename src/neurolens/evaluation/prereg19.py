"""Pre-registered analysis for increment 19 (docs/preregistration_increment19.md):
heuristic spike candidates are pathologic only when the learned IED model finds >= 2
separate detections at probability >= 0.98 over the whole record (recurrence), instead
of one candidate rated >= 0.95. Fresh seeded NMT train-split subset (seed 19), disjoint
from the development subset and from the increment-18 test subset; comparator: the
increment-18 pipeline (commit c24186e) on the same recordings.

Written and committed before either pipeline was run on the fresh subset.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .prereg17 import _counts

SENSITIVITY_GAIN_MIN = 0.03
SPECIFICITY_MARGIN = 0.03
SPECIFICITY_MIN = 0.55


def assess(new: dict, old: dict) -> dict[str, Any]:
    N, O = _counts(new["records"]), _counts(old["records"])
    if sorted(str(r.get("record_id")) for r in new["records"]) != sorted(
            str(r.get("record_id")) for r in old["records"]):
        raise ValueError("the two runs did not score the same recordings")
    h = {
        "H1_sensitivity_gain_at_least_0.03": N["sensitivity"] - O["sensitivity"] >= SENSITIVITY_GAIN_MIN,
        "H2_specificity_not_lower_by_more_than_0.03": N["specificity"] >= O["specificity"] - SPECIFICITY_MARGIN,
        "H3_specificity_at_least_0.55": N["specificity"] >= SPECIFICITY_MIN,
    }
    return {"new": N, "old": O, "hypotheses": h, "decision": {"keep_recurrence_rule": all(h.values())}}


def markdown_report(a: dict) -> str:
    from .prereg18 import markdown_report as _md

    text = _md({**a, "decision": a["decision"]})
    return text.replace("инкремент 18 |", "инкремент 19 (повторяемость) |", 1).replace(
        "инкремент 17 (286bb2a)", "инкремент 18 (c24186e)")


def apply_decision(a: dict, ied_yaml: str = "configs/ied.yaml",
                   impression_yaml: str = "configs/impression.yaml") -> dict:
    keep = a["decision"]["keep_recurrence_rule"]
    if keep:
        N = a["new"]
        lines = [
            "# What the report may claim about its own normal/abnormal conclusion on routine EEG.",
            "# Written by evaluation/prereg19.apply_decision (docs/preregistration_increment19.md).",
            f"# Fresh NMT subset 19 ({N['n_normal']} normal, {N['n_abnormal']} abnormal): specificity "
            f"{N['specificity']:.3f}, sensitivity {N['sensitivity']:.3f} (docs/increment19_report.md);",
            "# earlier: increment 18 0.640 / 0.780, increment 17 0.589 / 0.844.",
            "routine_validated: true",
            f"specificity: {N['specificity']:.3f}",
            f"sensitivity: {N['sensitivity']:.3f}",
            "provenance:",
            "  preregistration: docs/preregistration_increment19.md",
            "  report: docs/increment19_report.md",
            "  previous: docs/increment18_report.md",
        ]
        Path(impression_yaml).write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        p = Path(ied_yaml)
        p.write_text(re.sub(r"(?m)^recurrent_threshold:.*$", "recurrent_threshold: null",
                            p.read_text(encoding="utf-8")), encoding="utf-8")
    return {"keep_recurrence_rule": keep}
