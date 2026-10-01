"""Per-code confidence calibrator (TZ §13).

Holds a temperature per event code (with a pooled default) and applies it to
detector confidences. Loaded from / saved to a small JSON file so calibration
is versioned and shippable. Unknown codes fall back to the default temperature
(identity when uncalibrated).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..contracts.events import DetectionResult
from .temperature import apply_temperature


class ConfidenceCalibrator:
    def __init__(
        self,
        temperatures: dict[str, float] | None = None,
        default: float = 1.0,
        metrics: dict[str, Any] | None = None,
        version: str = "0.1.0",
    ):
        self.temperatures = dict(temperatures or {})
        self.default = float(default)
        self.metrics = dict(metrics or {})
        self.version = version

    @classmethod
    def identity(cls) -> "ConfidenceCalibrator":
        return cls()

    @classmethod
    def load(cls, path: str | Path) -> "ConfidenceCalibrator":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            temperatures=data.get("temperatures", {}),
            default=data.get("default", 1.0),
            metrics=data.get("metrics", {}),
            version=data.get("version", "0.1.0"),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"version": self.version, "default": self.default,
                 "temperatures": self.temperatures, "metrics": self.metrics},
                ensure_ascii=False, indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        return path

    def temperature_for(self, code: str) -> float:
        return self.temperatures.get(code, self.default)

    def calibrate(self, code: str, confidence: float) -> float:
        return float(apply_temperature([confidence], self.temperature_for(code))[0])

    def apply_to_detection(self, detection: DetectionResult) -> None:
        """Replace each event's confidence with its calibrated value (in place).

        The raw value is preserved in ``event.metadata['confidence_raw']``.
        """
        for e in detection.events:
            raw = e.confidence
            e.metadata["confidence_raw"] = round(raw, 4)
            e.confidence = round(self.calibrate(e.code, raw), 4)

    @property
    def is_identity(self) -> bool:
        return self.default == 1.0 and all(t == 1.0 for t in self.temperatures.values())
