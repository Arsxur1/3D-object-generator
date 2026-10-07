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
import os
import pickle
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


def stream_cache_path(cache_dir: str | Path, path: Path, cfg: ConfigBundle) -> Path:
    """Per-record cache file for :func:`stream_and_default` results. The key
    covers the EDF (name, size, mtime) and everything the result depends on:
    filters, monitor/alarm settings, detector thresholds, feature version."""
    from ..layer3_features.feature_set import FEATURE_VERSION

    st = path.stat()
    cfg_key = (cfg.filters.model_dump_json() + cfg.realtime.model_dump_json()
               + cfg.thresholds.model_dump_json() + f"|fv{FEATURE_VERSION}")
    key = hashlib.sha1(f"{path.name}:{st.st_size}:{st.st_mtime_ns}:{cfg_key}".encode()).hexdigest()[:16]
    return Path(cache_dir) / f"stream_{path.stem}_{key}.pkl"


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
    stream_cache: str | Path | None = None,
) -> list[EvalResult]:
    """``compare``: a second frozen set (e.g. the previous increment's models),
    scored as modes ``*-cmp-A`` on the same records.

    ``stream_cache``: directory where each record's monitor replay result is
    saved as soon as it finishes, so an interrupted run (e.g. a machine
    restart during hours of replay) resumes instead of starting over. Results
    are identical with or without it (the replay is deterministic)."""
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
        cpath = {a.file: stream_cache_path(stream_cache, p, cfg) for a, p in items} if stream_cache else {}
        if stream_cache:
            Path(stream_cache).mkdir(parents=True, exist_ok=True)
        for a, _ in items:
            if a.file in cpath and cpath[a.file].exists():
                got[a.file] = pickle.loads(cpath[a.file].read_bytes())
        if got and progress:
            progress(f"{len(got)} replayed records loaded from {stream_cache}")

        def done(res) -> None:
            rec, score = res
            got[rec.annotation.file] = (rec, score)
            if rec.annotation.file in cpath:
                tmp = cpath[rec.annotation.file].with_suffix(".tmp")
                tmp.write_bytes(pickle.dumps(res))
                os.replace(tmp, cpath[rec.annotation.file])
            if progress:
                progress(f"streamed {rec.annotation.file}")

        todo = [(a, p) for a, p in items if a.file not in got]
        run_memory_bounded(stream_and_default, [(a, p, cfg) for a, p in todo],
                           [int(p.stat().st_size * 8) for _, p in todo], workers, done)
        order = [a.file for a, _ in items]
        for a, _ in items:  # cached records carry the annotation they were built with
            got[a.file][0].annotation = a
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
