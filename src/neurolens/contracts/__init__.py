"""Data contracts (pydantic v2) — the single source of truth for layer I/O.

JSON Schemas under ``schemas/`` are generated from these models via
``python -m neurolens.contracts.export_schemas``.
"""

from __future__ import annotations

from .signal import (
    PatientInfo,
    ClinicalContext,
    QualityInfo,
    Provenance,
    UnifiedSignal,
    ElectrodeSystem,
)
from .events import (
    Event,
    EventEvidence,
    Localization,
    DetectionResult,
    AcnsModifiers,
    SeizureBurden,
)
from .causal import (
    CausalNode,
    CausalEdge,
    CausalGraph,
    NodeKind,
    PhysiologyLabel,
)
from .report import (
    LLMReport,
    ReportSection,
    Confidence,
    CriticalFinding,
    OperatingMode,
    Bilingual,
    DISCLAIMER_RU,
    DISCLAIMER_UZ,
)
from .config import (
    MontageConfig,
    MontageType,
    FilterConfig,
    Thresholds,
    RuleChain,
    RuleBase,
)

__all__ = [
    # signal
    "PatientInfo",
    "ClinicalContext",
    "QualityInfo",
    "Provenance",
    "UnifiedSignal",
    "ElectrodeSystem",
    # events
    "Event",
    "EventEvidence",
    "Localization",
    "DetectionResult",
    "AcnsModifiers",
    "SeizureBurden",
    # causal
    "CausalNode",
    "CausalEdge",
    "CausalGraph",
    "NodeKind",
    "PhysiologyLabel",
    # report
    "LLMReport",
    "ReportSection",
    "Confidence",
    "CriticalFinding",
    "OperatingMode",
    "Bilingual",
    "DISCLAIMER_RU",
    "DISCLAIMER_UZ",
    # config
    "MontageConfig",
    "MontageType",
    "FilterConfig",
    "Thresholds",
    "RuleChain",
    "RuleBase",
]
