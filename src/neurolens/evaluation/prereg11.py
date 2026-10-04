"""Pre-registered analysis for increment 11 (docs/preregistration_increment11.md).

Written and committed before the test data were downloaded; applied unchanged
to the ``metrics.json`` of ``neurolens ml-eval --compare ... --analysis prereg11``.

Modes: *-threshold-default (threshold detector), *-ml-A (retrained v2 models,
31 training patients), *-cmp-A (increment-10 v1 models, 20 training patients).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .prereg import _population, _subject, summarize
from .prereg10 import macro_sensitivity

SENS_MARGIN = 0.05


def _noninferior(tab: dict, a: str, b: str) -> bool:
    """a is not worse than b: sensitivity >= b - margin and no more false alarms."""
    return (tab[a]["sensitivity"] >= tab[b]["sensitivity"] - SENS_MARGIN
            and tab[a]["fa_per_hour"] <= tab[b]["fa_per_hour"])


def assess(metrics: dict) -> dict[str, Any]:
    res = {r["mode"]: r for r in metrics["results"]}
    tab = {m: summarize(r["records"])["all"] for m, r in res.items()}
    macro = {m: macro_sensitivity(r["records"]) for m, r in res.items()}
    h = {
        "H1_realtime_v2_noninferior_to_threshold_monitor": _noninferior(tab, "realtime-ml-A", "realtime-threshold-default"),
        "H2_offline_v2_noninferior_to_threshold": _noninferior(tab, "offline-ml-A", "offline-threshold-default"),
        "H3_offline_v2_noninferior_to_v1": _noninferior(tab, "offline-ml-A", "offline-cmp-A"),
        "H4_realtime_v2_macro_noninferior": macro["realtime-ml-A"] >= macro["realtime-threshold-default"] - SENS_MARGIN,
    }
    v1_ok = _noninferior(tab, "offline-cmp-A", "offline-threshold-default")
    if h["H2_offline_v2_noninferior_to_threshold"] and h["H3_offline_v2_noninferior_to_v1"]:
        offline = "learned_v2"
    elif v1_ok:
        offline = "learned_v1"
    else:
        offline = "threshold"
    return {
        "overall": tab,
        "macro_sensitivity": macro,
        "by_population": {m: summarize(r["records"], _population) for m, r in res.items()},
        "by_subject": {m: summarize(r["records"], _subject) for m, r in res.items()},
        "hypotheses": h,
        "secondary": {"offline_v1_noninferior_to_threshold": v1_ok},
        "decision": {
            "realtime_default": "learned_v2" if h["H1_realtime_v2_noninferior_to_threshold_monitor"] else "threshold",
            "offline_default": offline,
        },
    }


_MODES = [
    ("offline-threshold-default", "оффлайн", "пороговый, дефолт"),
    ("offline-cmp-A", "оффлайн", "обученный v1 (20 пациентов), A"),
    ("offline-ml-A", "оффлайн", "обученный v2 (31 пациент), A"),
    ("realtime-threshold-default", "real-time", "пороговый, дефолт"),
    ("realtime-cmp-A", "real-time", "обученный v1 (20 пациентов), A"),
    ("realtime-ml-A", "real-time", "обученный v2 (31 пациент), A"),
]


def markdown_report(a: dict) -> str:
    from . import prereg10

    saved = prereg10._MODES
    try:
        prereg10._MODES = _MODES  # same table layout, increment-11 modes
        text = prereg10.markdown_report({**a, "decision": {}})
    finally:
        prereg10._MODES = saved
    text = text.rstrip() + "\n"
    for name, value in a["decision"].items():
        text += f"* {name}: **{value}**\n"
    text += f"* вторичное — v1 не хуже порогового оффлайн: **{'да' if a['secondary']['offline_v1_noninferior_to_threshold'] else 'нет'}**\n"
    return text


def apply_decision(a: dict, frozen11: dict, frozen10: dict, ml_yaml: str = "configs/ml.yaml") -> dict:
    """Rewrite configs/ml.yaml per the pre-registered rules (comments preserved):
    model path / threshold / min_epochs / enabled for each section."""
    d = a["decision"]
    plan: dict[str, dict] = {}
    for mode, choice in (("offline", d["offline_default"]), ("realtime", d["realtime_default"])):
        if choice == "learned_v2":
            src = frozen11[mode]
        elif choice == "learned_v1":
            src = frozen10[mode]
        else:
            src = None
        if src is None:
            plan[mode] = {"enabled": False}
        else:
            prm = src["A_replacement"]["params"]
            plan[mode] = {"enabled": True, "model": src["model"]["path"],
                          "threshold": prm["threshold"], "min_epochs": prm["min_epochs"]}
    p = Path(ml_yaml)
    out, section = [], None
    for line in p.read_text(encoding="utf-8").splitlines(keepends=True):
        top = re.match(r"^([A-Za-z_]\w*):", line)
        if top:
            section = top.group(1)
        elif section in plan:
            m = re.match(r"^(\s+)(enabled|model|threshold|min_epochs):(\s*)([^#\n]*?)(\s*#.*)?$", line.rstrip("\n"))
            if m and m.group(2) in plan[section]:
                val = plan[section][m.group(2)]
                val = str(val).lower() if isinstance(val, bool) else val
                line = f"{m.group(1)}{m.group(2)}:{m.group(3) or ' '}{val}{m.group(5) or ''}\n"
        out.append(line)
    p.write_text("".join(out), encoding="utf-8")
    return plan
