"""Fast real-time evaluation by replaying a recorded monitor trace.

Running :class:`RealtimeMonitor` costs ~3 min per recording hour (every 5 s it
re-filters and re-featurises a 30 s window). Window features and the rolling
amplitude baseline do not depend on detector thresholds or alarm gates, so we
record them once per record (``build_trace``) and then replay only the ictal
detector + :class:`AlarmManager` for any threshold/alarm-rule combination in
milliseconds. ``replay`` reproduces the monitor's SEIZURE alarms exactly for the
same configuration (tested), which makes real-time tuning tractable.
"""

from __future__ import annotations

import copy
import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ..contracts.alarms import AlarmSeverity, AlarmType
from ..contracts.config import RealtimeConfig, Thresholds
from ..contracts.report import Bilingual
from ..datasets.annotations import RecordAnnotation
from ..layer4_detect.ictal import IctalRhythmDetector
from ..pipeline.config_loader import ConfigBundle
from ..realtime.alarms import AlarmCandidate, AlarmManager
from .matching import Detection, RecordScore, aggregate, score_record
from .runner import EvalResult, ScoringRules

_MSG = Bilingual(ru="судорожная активность", uz="tutqanoq faolligi")


@dataclass
class MonitorTrace:
    annotation: RecordAnnotation
    duration_s: float
    windows: list[tuple]  # (t0, t1, FeatureSet, baseline ndarray)


def build_trace(ann: RecordAnnotation, path: Path, cfg: ConfigBundle) -> MonitorTrace:
    from ..realtime.monitor import RealtimeMonitor
    from ..realtime.stream import EdfReplaySource

    src = EdfReplaySource(path, chunk_s=cfg.realtime.step_s)
    mon = RealtimeMonitor(cfg, detectors=[IctalRhythmDetector()])
    mon.trace = []
    summary = mon.run(src)
    return MonitorTrace(ann, summary.duration_s, mon.trace)


def build_traces(
    items: list[tuple[RecordAnnotation, Path]],
    cfg: ConfigBundle,
    cache_dir: str | Path | None = None,
    progress: Optional[Callable[[str], None]] = None,
    workers: int = 4,
) -> list[MonitorTrace]:
    """Traces for many records, memoised on disk (key: file + filters + window/step)."""
    out = []
    cdir = Path(cache_dir) if cache_dir else None
    if cdir:
        cdir.mkdir(parents=True, exist_ok=True)
    rt = cfg.realtime
    ckey = hashlib.sha1(
        (cfg.filters.model_dump_json() + f"{rt.window_s}:{rt.step_s}").encode()
    ).hexdigest()[:10]
    def cache_path(path: Path) -> Optional[Path]:
        if not cdir:
            return None
        st = path.stat()
        key = hashlib.sha1(f"{path.name}:{st.st_size}:{st.st_mtime_ns}:{ckey}".encode()).hexdigest()[:16]
        return cdir / f"rt_{path.stem}_{key}.pkl"

    todo = [(ann, path) for ann, path in items
            if not ((cp := cache_path(path)) and cp.exists())]
    built: dict[Path, MonitorTrace] = {}
    if todo:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        n = max(1, min(workers, len(todo)))
        with ProcessPoolExecutor(max_workers=n) as pool:
            futs = {pool.submit(build_trace, ann, path, cfg): path for ann, path in todo}
            for fut in as_completed(futs):
                path = futs[fut]
                tr = fut.result()
                if (cp := cache_path(path)) is not None:
                    cp.write_bytes(pickle.dumps(tr))
                built[path] = tr
                if progress:
                    progress(f"traced {tr.annotation.file} ({len(tr.windows)} windows)")
    for ann, path in items:
        tr = built.get(path)
        if tr is None:
            tr = pickle.loads(cache_path(path).read_bytes())
        tr.annotation = ann
        out.append(tr)
    return out


def replay(trace: MonitorTrace, thresholds: Thresholds, rt: RealtimeConfig) -> list[Detection]:
    det = IctalRhythmDetector()
    th = thresholds.model_copy(update=dict(rt.threshold_overrides))
    mgr = AlarmManager(rt)
    dets: list[Detection] = []
    for i, (t0, t1, feats, baseline) in enumerate(trace.windows):
        det.external_baseline = baseline
        cands = [
            AlarmCandidate(type=AlarmType.SEIZURE, severity=AlarmSeverity.CRITICAL,
                           message=_MSG, confidence=e.confidence, is_critical=True)
            for e in det.detect(None, feats, th)
            if e.confidence >= th.detection_min_confidence
        ]
        for a in mgr.process(cands, i, t0, t1, 0.0):
            if a.type == AlarmType.SEIZURE:
                dets.append(Detection(a.t_start, a.t_end, a.confidence, t_known=a.t_end))
    return dets


def evaluate_replay(
    traces: list[MonitorTrace], thresholds: Thresholds, rt: RealtimeConfig,
    rules: ScoringRules | None = None,
) -> EvalResult:
    from time import perf_counter

    rules = rules or ScoringRules()
    t0 = perf_counter()
    scores: list[RecordScore] = []
    for tr in traces:
        scores.append(score_record(
            tr.annotation.file, tr.annotation.seizures, replay(tr, thresholds, rt),
            tr.duration_s, rules.pre_s, rules.post_s, rules.merge_gap_s,
        ))
    return EvalResult("realtime", scores, aggregate(scores), perf_counter() - t0)


def with_realtime_params(rt: RealtimeConfig, params: dict) -> RealtimeConfig:
    """Split tuning params: ``persistence_windows`` -> seizure alarm rule,
    everything else -> ``threshold_overrides``."""
    rt = copy.deepcopy(rt)
    p = dict(params)
    pers = p.pop("persistence_windows", None)
    if pers is not None:
        rule = rt.rule_for(AlarmType.SEIZURE.value).model_copy(update={"persistence_windows": int(pers)})
        rt.alarms = {**rt.alarms, AlarmType.SEIZURE.value: rule}
    rt.threshold_overrides = {**rt.threshold_overrides, **{k: float(v) for k, v in p.items()}}
    return rt

