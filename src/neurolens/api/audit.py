"""Append-only audit log (TZ §12: audit trail, regulator-ready traceability).

Records every request without PHI: timestamp, endpoint, deidentified subject id,
software/rule-base versions, and a small outcome summary. One JSON object per
line (JSONL) for easy shipping to a SIEM / audit store.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import __version__

_LOCK = threading.Lock()


class AuditLogger:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.environ.get("NEUROLENS_AUDIT_LOG", "audit.log.jsonl"))
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        endpoint: str,
        subject_id: str | None,
        rule_base_version: str | None = None,
        summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "subject_id": subject_id,
            "neurolens_version": __version__,
            "rule_base_version": rule_base_version,
            "summary": summary or {},
        }
        line = json.dumps(entry, ensure_ascii=False)
        with _LOCK:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return entry

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]
