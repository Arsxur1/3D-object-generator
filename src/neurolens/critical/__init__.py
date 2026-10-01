"""Hard-safety critical findings (TZ §10) — raised independent of LLM and mode."""

from __future__ import annotations

from .safety import scan_critical_findings

__all__ = ["scan_critical_findings"]
