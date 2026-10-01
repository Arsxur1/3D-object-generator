"""Structured capture of neurophysiologist corrections (TZ §13).

This is the fuel for future fine-tuning and confidence calibration: a
neurophysiologist confirms/corrects an event, its localization, or a causal
chain, and we append a structured record. Confidence calibration (temperature
scaling, ECE) will consume this log in v2; the skeleton provides the schema and
an append-only JSONL sink.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class Correction:
    """One expert action against a NeuroLens output."""

    recording_id: str
    target_kind: str  # "event" | "localization" | "causal_edge" | "impression"
    target_ref: str  # event code / node id / edge "src->tgt"
    action: str  # "confirm" | "reject" | "edit"
    corrected_value: Optional[dict[str, Any]] = None
    reviewer: Optional[str] = None
    model_confidence: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: Optional[str] = None


class FeedbackLogger:
    """Append-only JSONL logger for expert corrections."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, correction: Correction) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(correction), ensure_ascii=False) + "\n")

    def read_all(self) -> list[Correction]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(Correction(**json.loads(line)))
        return out
