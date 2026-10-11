"""Service layer — thin wrappers over the pipeline/monitor with deid + audit (§12)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts.report import OperatingMode
from ..contracts.signal import ClinicalContext, PatientInfo
from ..pipeline.config_loader import ConfigBundle, load_configs
from ..pipeline.pipeline import Pipeline
from ..realtime.monitor import RealtimeMonitor
from ..realtime.stream import EdfReplaySource
from .audit import AuditLogger
from .deid import deidentify_result, hash_identifier
from .models import AnalyzeParams


def _patient(p: AnalyzeParams) -> PatientInfo:
    return PatientInfo(age_years=p.age_years, postmenstrual_age_weeks=p.postmenstrual_age_weeks)


def _context(p: AnalyzeParams) -> ClinicalContext:
    return ClinicalContext(
        sedatives=p.sedatives, antiseizure_meds=p.antiseizure_meds,
        temperature_c=p.temperature_c, clinical_question=p.clinical_question,
    )


def run_analyze(
    path: str | Path,
    params: AnalyzeParams,
    config: ConfigBundle,
    audit: AuditLogger,
) -> dict[str, Any]:
    mode = OperatingMode.B_AUTONOMOUS if params.mode.upper() == "B" else OperatingMode.A_DECISION_SUPPORT
    pipe = Pipeline(config=config, provider_pref=params.provider)
    out = pipe.analyze_file(
        path, mode=mode, montage_name=params.montage,
        patient=_patient(params), context=_context(params),
    )
    result, subject_id = deidentify_result(out.result_json)
    audit.record(
        "/analyze", subject_id, out.graph.rule_base_version,
        {"n_events": len(out.detection.events),
         "n_critical": len(out.report.critical_findings),
         "mode": mode.value},
    )
    return result


def run_monitor(
    path: str | Path,
    montage: str,
    window_s: float | None,
    step_s: float | None,
    config: ConfigBundle,
    audit: AuditLogger,
    patient: PatientInfo | None = None,
) -> dict[str, Any]:
    import dataclasses

    cfg = config or load_configs()
    # per-request overrides on a copy: the server's shared config must not change
    # between requests (a window set by one caller leaked into all later ones)
    rt = cfg.realtime.model_copy(update={k: v for k, v in (("window_s", window_s), ("step_s", step_s))
                                         if v is not None})
    cfg = dataclasses.replace(cfg, realtime=rt)
    src = EdfReplaySource(path, chunk_s=cfg.realtime.step_s)
    # patient drives detector routing (neonates: configs/ml.yaml neonatal.realtime)
    summary = RealtimeMonitor(config=cfg, montage_name=montage, patient=patient).run(src)

    d = summary.model_dump(mode="json")
    subject_id = hash_identifier(str(d.get("source", "unknown")))
    d["source"] = subject_id  # deidentify: source filename -> subject hash
    d["subject_id"] = subject_id
    audit.record(
        "/monitor", subject_id, cfg.rule_base.version,
        {"n_alarms": summary.n_alarms_raised, "n_windows": summary.n_windows},
    )
    return d
