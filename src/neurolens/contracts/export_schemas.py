"""Export JSON Schemas from the pydantic contracts (TZ §18.2).

Run:  python -m neurolens.contracts.export_schemas [out_dir]

Writes ``<out_dir>/<name>.schema.json`` for the signal, montage, event,
causal-graph, and LLM-report contracts. Keeps ``schemas/`` in sync with the
models (the single source of truth).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .alarms import Alarm, MonitorSummary
from .causal import CausalGraph
from .config import MontageConfig, RealtimeConfig, RuleBase, Thresholds
from .events import DetectionResult, Event
from .report import LLMReport
from .signal import UnifiedSignal

# name -> model
SCHEMAS = {
    "signal": UnifiedSignal,
    "montage": MontageConfig,
    "thresholds": Thresholds,
    "rule_base": RuleBase,
    "event": Event,
    "detection_result": DetectionResult,
    "causal_graph": CausalGraph,
    "llm_report": LLMReport,
    "realtime_config": RealtimeConfig,
    "alarm": Alarm,
    "monitor_summary": MonitorSummary,
}


def export(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, model in SCHEMAS.items():
        schema = model.model_json_schema()
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema.setdefault("title", name)
        path = out_dir / f"{name}.schema.json"
        path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written


def default_out_dir() -> Path:
    # repo_root/schemas  (this file is src/neurolens/contracts/export_schemas.py)
    return Path(__file__).resolve().parents[3] / "schemas"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out = Path(argv[0]) if argv else default_out_dir()
    written = export(out)
    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
