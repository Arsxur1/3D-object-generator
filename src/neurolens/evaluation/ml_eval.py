"""Held-out evaluation of the learned ictal detector vs the threshold detector.

Offline: whole-record features -> frozen offline model -> events at each frozen
operating point; the threshold detector with default settings on the same
records for comparison. Real-time: each record is replayed through the
monitor once in a worker process (trace not cached, to save disk); the worker
returns the monitor's epoch stream and the threshold monitor's scored alarms,
then the frozen real-time model runs on the stream with causal (streaming)
latency. Output: ``metrics.json`` in the same shape as ``neurolens tune``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional

from ..datasets.annotations import RecordAnnotation
from ..layer4_detect.ml_ictal import load_model
from ..pipeline.config_loader import ConfigBundle
from .matching import aggregate, score_record
from .ml import Dataset, detections_from_probs, realtime_records
from .report import save_results
from .runner import EvalResult, PreparedRecord, ScoringRules, evaluate_offline, prepare_records, run_memory_bounded


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stream_and_default(ann: RecordAnnotation, path: Path, cfg: ConfigBundle):
    """Worker: replay one record through the monitor; return its epoch stream
    record and the default threshold monitor's score (trace discarded)."""
    from .realtime_replay import build_trace, evaluate_replay

    tr = build_trace(ann, path, cfg)
    score = evaluate_replay([tr], cfg.thresholds, cfg.realtime).scores[0]
    return realtime_records([tr])[0], score


def _ml_result(mode: str, ds: Dataset, model, params: dict, causal: bool,
               rules: ScoringRules) -> EvalResult:
    t0 = perf_counter()
    scores = []
    for rec, X, t in zip(ds.records, ds.X, ds.times):
        p = model.predict_proba(X) if len(X) else X[:, 0]
        dets = detections_from_probs(t, p, params["threshold"], params["min_epochs"], causal)
        scores.append(score_record(rec.annotation.file, rec.annotation.seizures, dets,
                                   rec.duration_s, rules.pre_s, rules.post_s, rules.merge_gap_s))
    return EvalResult(mode, scores, aggregate(scores), perf_counter() - t0)


def evaluate_ml_test(
    items: list[tuple[RecordAnnotation, Path]],
    cfg: ConfigBundle,
    frozen: dict,
    feature_cache: str | Path | None = None,
    workers: int = 3,
    realtime: bool = True,
    progress: Optional[Callable[[str], None]] = None,
    rules: ScoringRules | None = None,
    compare: dict | None = None,
) -> list[EvalResult]:
    """``compare``: a second frozen set (e.g. the previous increment's models),
    scored as modes ``*-cmp-A`` on the same records."""
    rules = rules or ScoringRules()
    for fz in (frozen, compare) if compare else (frozen,):
        for mode in ("offline", "realtime") if realtime else ("offline",):
            m = fz[mode]["model"]
            if _sha(m["path"]) != m["sha256"]:
                raise RuntimeError(f"{m['path']} does not match the frozen sha256")
    results: list[EvalResult] = []

    recs = prepare_records(items, cfg, feature_cache, progress=progress, workers=workers)
    base = evaluate_offline(recs, cfg.thresholds, rules)
    base.mode = "offline-threshold-default"
    results.append(base)
    ds = Dataset.build(recs, rules)
    model = load_model(frozen["offline"]["model"]["path"])
    for op in ("A_replacement", "B_quiet"):
        if op in frozen["offline"]:
            results.append(_ml_result(f"offline-ml-{op[0]}", ds, model, frozen["offline"][op]["params"],
                                      False, rules))
    if compare:
        results.append(_ml_result("offline-cmp-A", ds, load_model(compare["offline"]["model"]["path"]),
                                  compare["offline"]["A_replacement"]["params"], False, rules))

    if realtime:
        got: dict[str, tuple[PreparedRecord, object]] = {}

        def done(res) -> None:
            rec, score = res
            got[rec.annotation.file] = (rec, score)
            if progress:
                progress(f"streamed {rec.annotation.file}")

        run_memory_bounded(stream_and_default, [(a, p, cfg) for a, p in items],
                           [int(p.stat().st_size * 8) for _, p in items], workers, done)
        order = [a.file for a, _ in items]
        srecs = [got[f][0] for f in order]
        rt_base_scores = [got[f][1] for f in order]
        rt_base = EvalResult("realtime-threshold-default", rt_base_scores, aggregate(rt_base_scores), 0.0)
        results.append(rt_base)
        rds = Dataset.build(srecs, rules)
        rmodel = load_model(frozen["realtime"]["model"]["path"])
        for op in ("A_replacement", "B_quiet"):
            if op in frozen["realtime"]:
                results.append(_ml_result(f"realtime-ml-{op[0]}", rds, rmodel,
                                          frozen["realtime"][op]["params"], True, rules))
        if compare:
            results.append(_ml_result("realtime-cmp-A", rds, load_model(compare["realtime"]["model"]["path"]),
                                      compare["realtime"]["A_replacement"]["params"], True, rules))
    return results


def run_and_save(items, cfg, frozen_path: str | Path, out_dir: str | Path,
                 compare_path: str | Path | None = None, **kw) -> dict:
    frozen = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
    compare = json.loads(Path(compare_path).read_text(encoding="utf-8")) if compare_path else None
    results = evaluate_ml_test(items, cfg, frozen, compare=compare, **kw)
    paths = save_results(results, out_dir, title=f"Held-out evaluation of the learned detector ({Path(frozen_path).name})",
                         extra={"frozen": str(frozen_path), "compare": str(compare_path) if compare_path else None,
                                "test_records": [a.file for a, _ in items]})
    return {"results": results, "paths": paths}
