"""Grounding validation (TZ §9, §16).

Enforces that 100% of the report's factual statements trace to an allowed
grounding key (a Layer-3 feature, a Layer-4 event code, or a Layer-5 causal
node/edge/rule). Ungrounded refs are stripped; factual sections left with no
valid grounding are dropped. This is what makes an LLM narrative safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..contracts.report import LLMReport

# Sections that make claims about the EEG and therefore MUST be grounded.
_FACTUAL_KEYS = {"background", "epileptiform", "ictal_acns", "causal"}


@dataclass
class GroundingResult:
    report: LLMReport
    dropped_sections: list[str] = field(default_factory=list)
    stripped_refs: list[str] = field(default_factory=list)
    fully_grounded: bool = True


def validate_grounding(report: LLMReport, allowed_keys: set[str]) -> GroundingResult:
    kept = []
    dropped: list[str] = []
    stripped: list[str] = []

    for sec in report.sections:
        valid_refs = [r for r in sec.grounding_refs if r and r in allowed_keys]
        invalid = [r for r in sec.grounding_refs if r and r not in allowed_keys]
        stripped.extend(invalid)

        if sec.key in _FACTUAL_KEYS and not valid_refs:
            dropped.append(sec.key)
            continue
        sec.grounding_refs = valid_refs
        kept.append(sec)

    report.sections = kept
    return GroundingResult(
        report=report,
        dropped_sections=dropped,
        stripped_refs=stripped,
        fully_grounded=not dropped and not stripped,
    )
