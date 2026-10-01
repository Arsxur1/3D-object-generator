"""Validation against expert-annotated EEG (TZ §16): scoring, runners, tuning."""

from __future__ import annotations

from .matching import AggregateScore, Detection, RecordScore, aggregate, score_record
from .runner import (
    EvalResult,
    PreparedRecord,
    ScoringRules,
    evaluate_offline,
    evaluate_realtime,
    load_items,
    prepare_record,
    prepare_records,
)

__all__ = [
    "AggregateScore",
    "Detection",
    "EvalResult",
    "PreparedRecord",
    "RecordScore",
    "ScoringRules",
    "aggregate",
    "evaluate_offline",
    "evaluate_realtime",
    "load_items",
    "prepare_record",
    "prepare_records",
    "score_record",
]
