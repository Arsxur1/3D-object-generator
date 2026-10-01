"""Structured JSON output (TZ §11.1) — the machine-readable result.

Bundles features, events, causal graph, confidences, critical flags, versions,
mode, and montage/reference into one JSON document. A compact feature summary
is emitted (not the full epoched arrays) to keep the document practical.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import __version__
from ..contracts.causal import CausalGraph
from ..contracts.events import DetectionResult
from ..contracts.report import LLMReport
from ..contracts.signal import UnifiedSignal
from ..layer3_features.feature_set import FeatureSet
from ..layer6_interpret.llm_base import ModeGateResult


def _feature_summary(f: FeatureSet) -> dict[str, Any]:
    aeeg = None
    if f.aeeg is not None and f.aeeg.upper_uv.size:
        bw = f.aeeg.bandwidth()
        aeeg = {
            "channel": f.aeeg.channel,
            "upper_uv_median": round(float(sorted(f.aeeg.upper_uv)[len(f.aeeg.upper_uv) // 2]), 2),
            "bandwidth_uv_median": round(float(sorted(bw)[len(bw) // 2]), 2),
            "n_epochs": int(f.aeeg.upper_uv.size),
        }
    return {
        "pdr_hz": round(f.pdr_hz, 2),
        "global_relative_power": {k: round(v, 3) for k, v in f.global_relpow.items()},
        "global_asymmetry": round(f.global_asymmetry, 3),
        "asymmetry_by_pair": {k: round(v, 3) for k, v in f.asymmetry.items()},
        "burst_suppression": {
            "suppression_ratio": round(f.burst.suppression_ratio, 3),
            "n_bursts": f.burst.n_bursts,
            "mean_ibi_s": round(f.burst.mean_ibi_s, 2),
        },
        "aeeg": aeeg,
        "representative_channel": f.representative_channel,
        "per_channel": {
            ch: {k: round(v, 3) for k, v in feats.items()}
            for ch, feats in f.channel_summary.items()
        },
    }


def build_result_json(
    signal: UnifiedSignal,
    features: FeatureSet,
    detection: DetectionResult,
    graph: CausalGraph,
    report: LLMReport,
    gate: ModeGateResult,
    montage_name: str,
) -> dict[str, Any]:
    return {
        "neurolens_version": __version__,
        "rule_base_version": graph.rule_base_version,
        "mode": report.mode.value,
        "mode_gate_passed": report.mode_gate_passed,
        "montage": montage_name,
        "reference": signal.reference,
        "recording": signal.metadata_dict(),
        "features": _feature_summary(features),
        "events": [e.model_dump(mode="json") for e in detection.events],
        "detectors_run": detection.detectors_run,
        "seizure_burden": (
            detection.seizure_burden.model_dump(mode="json")
            if detection.seizure_burden else None
        ),
        "causal_graph": graph.model_dump(mode="json"),
        "critical_findings": [cf.model_dump(mode="json") for cf in report.critical_findings],
        "report": report.model_dump(mode="json"),
        "mode_b_gate": {
            "passed": gate.passed,
            "reasons_ru": gate.reasons_ru,
            "reasons_uz": gate.reasons_uz,
        },
    }


def save_json(result: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def validate_against_schema(result: dict[str, Any], schema_path: str | Path) -> None:
    """Validate a sub-document against a generated JSON Schema (optional check)."""
    import jsonschema

    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    jsonschema.validate(result, schema)
