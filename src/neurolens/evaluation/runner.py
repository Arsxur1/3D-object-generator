"""Run NeuroLens seizure detection over annotated recordings and score it.

Two operating paths are evaluated, matching how NeuroLens is used:

* ``offline``  — whole-record batch analysis (Layer 2 filters + average
  reference -> Layer 3 features -> Layer 4 ictal detector). Features do not
  depend on detector thresholds, so they are computed once per record and
  cached; threshold tuning then only re-runs the cheap detector.
* ``realtime`` — the streaming :class:`RealtimeMonitor` replaying the EDF in
  chunks, scoring raised SEIZURE alarms (persistence/refractory gates included),
  with latency measured to the alarm window end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional

from ..contracts.alarms import AlarmType
from ..contracts.config import Thresholds
from ..datasets.annotations import RecordAnnotation
from ..layer1_ingest.registry import ingest
from ..layer2_preprocess.filters import apply_filters
from ..layer2_preprocess.reref import rereference
from ..layer3_features.feature_set import FeatureSet, compute_features
from ..layer4_detect.ictal import IctalRhythmDetector
from ..pipeline.config_loader import ConfigBundle, load_configs
from .matching import AggregateScore, Detection, RecordScore, aggregate, score_record


@dataclass
class ScoringRules:
    pre_s: float = 30.0
    post_s: float = 60.0
    merge_gap_s: float = 10.0


@dataclass
class PreparedRecord:
    annotation: RecordAnnotation
    path: Path
    duration_s: float
    features: FeatureSet
    signal: object = None  # not retained (detector uses features only)
    flags: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    mode: str
    scores: list[RecordScore]
    total: AggregateScore
    seconds: float

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "aggregate": self.total.as_dict(),
            "runtime_s": round(self.seconds, 1),
            "records": [
                {
                    "file": s.file,
                    "hours": round(s.hours, 3),
                    "n_seizures": s.n_seizures,
                    "tp": s.tp, "fn": s.fn, "fp": s.fp,
                    "latencies_s": [round(x, 1) for x in s.latencies_s],
                    "missed": [[m.onset_s, m.offset_s] for m in s.missed],
                    "false_positives": [
                        [round(d.t_start, 1), round(d.t_end, 1), round(d.confidence, 3)]
                        for d in s.false_positives
                    ],
                }
                for s in self.scores
            ],
        }


def prepare_record(ann: RecordAnnotation, path: Path, cfg: ConfigBundle) -> PreparedRecord:
    sig = ingest(path)
    duration_s, flags = sig.duration_s, list(sig.quality.flags)
    filt = apply_filters(sig, cfg.filters)
    del sig  # long multi-channel records: free each stage as soon as possible
    analysis = rereference(filt, "average")
    del filt
    feats = compute_features(analysis, cfg.filters)
    # the ictal detector works on features only; dropping the samples keeps the
    # on-disk feature cache ~100x smaller
    return PreparedRecord(
        annotation=ann, path=path, duration_s=duration_s, features=feats,
        signal=None, flags=flags,
    )


# Peak resident memory per byte of EDF observed when processing a whole record
# (int16 on disk -> float64 working copies through filtering/re-referencing).
MEM_PER_EDF_BYTE = 16.0


def memory_budget_bytes(fraction: float = 0.6) -> int:
    """Usable memory for parallel workers (cgroup limit if set, else physical RAM)."""
    import os

    limit = None
    try:
        raw = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if raw.isdigit():
            limit = int(raw)
    except OSError:
        pass
    phys = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return int(fraction * min(limit or phys, phys))


def run_memory_bounded(fn, tasks: list[tuple], est_bytes: list[int], workers: int,
                       on_done: Callable, budget: int | None = None) -> None:
    """Run ``fn(*task)`` in worker processes, never starting a task while the
    estimated memory of running tasks plus its own would exceed ``budget``
    (a task larger than the budget runs alone). Results go to ``on_done``."""
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    budget = budget or memory_budget_bytes()
    order = sorted(range(len(tasks)), key=lambda i: -est_bytes[i])  # big first
    running: dict = {}
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        while order or running:
            used = sum(running.values())
            started = False
            for k, i in enumerate(order):
                if len(running) < workers and (not running or used + est_bytes[i] <= budget):
                    running[pool.submit(fn, *tasks[i])] = est_bytes[i]
                    order.pop(k)
                    started = True
                    break
            if started:
                continue
            finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for f in finished:
                running.pop(f)
                on_done(f.result())


def prepare_records(
    items: list[tuple[RecordAnnotation, Path]],
    cfg: ConfigBundle,
    cache_dir: str | Path | None = None,
    progress: Optional[Callable[[str], None]] = None,
    workers: int = 1,
) -> list[PreparedRecord]:
    """Prepare many records (in parallel processes); optionally memoise on disk.

    The cache key covers the EDF (name, size, mtime) and the filter config, so
    a filter change invalidates it. Cache files are local derived data
    (gitignored), never shipped.
    """
    import hashlib
    import pickle

    cdir = Path(cache_dir) if cache_dir else None
    if cdir:
        cdir.mkdir(parents=True, exist_ok=True)
    fkey = hashlib.sha1(cfg.filters.model_dump_json().encode()).hexdigest()[:10]

    def cache_path(path: Path) -> Optional[Path]:
        if not cdir:
            return None
        st = path.stat()
        key = hashlib.sha1(f"{path.name}:{st.st_size}:{st.st_mtime_ns}:{fkey}".encode()).hexdigest()[:16]
        return cdir / f"{path.stem}_{key}.pkl"

    todo = [(a, p) for a, p in items if not ((cp := cache_path(p)) and cp.exists())]
    built: dict[Path, PreparedRecord] = {}

    def done(rec: PreparedRecord, secs: float) -> None:
        if (cp := cache_path(rec.path)) is not None:
            cp.write_bytes(pickle.dumps(rec))
        built[rec.path] = rec
        if progress:
            progress(f"prepared {rec.annotation.file} in {secs:.0f}s")

    if workers > 1 and len(todo) > 1:
        t0 = perf_counter()
        run_memory_bounded(
            prepare_record, [(a, p, cfg) for a, p in todo],
            [int(p.stat().st_size * MEM_PER_EDF_BYTE) for _, p in todo],
            min(workers, len(todo)), lambda rec: done(rec, perf_counter() - t0),
        )
    else:
        for a, p in todo:
            t0 = perf_counter()
            done(prepare_record(a, p, cfg), perf_counter() - t0)

    out = []
    for ann, path in items:
        rec = built.get(path)
        if rec is None:
            rec = pickle.loads(cache_path(path).read_bytes())
        rec.annotation = ann  # annotations may be re-parsed/fixed
        out.append(rec)
    return out


def offline_detections(rec: PreparedRecord, thresholds: Thresholds) -> list[Detection]:
    events = IctalRhythmDetector().detect(rec.signal, rec.features, thresholds)
    return [Detection(e.t_start, e.t_end, e.confidence) for e in events
            if e.confidence >= thresholds.detection_min_confidence]


def evaluate_offline(
    records: list[PreparedRecord],
    thresholds: Thresholds,
    rules: ScoringRules | None = None,
    min_confidence: float = 0.0,
) -> EvalResult:
    rules = rules or ScoringRules()
    t0 = perf_counter()
    scores = []
    for rec in records:
        dets = [d for d in offline_detections(rec, thresholds) if d.confidence >= min_confidence]
        scores.append(score_record(
            rec.annotation.file, rec.annotation.seizures, dets, rec.duration_s,
            rules.pre_s, rules.post_s, rules.merge_gap_s,
        ))
    return EvalResult("offline", scores, aggregate(scores), perf_counter() - t0)


def realtime_detections(path: Path, cfg: ConfigBundle) -> tuple[list[Detection], float]:
    from ..realtime.monitor import RealtimeMonitor
    from ..realtime.stream import EdfReplaySource

    src = EdfReplaySource(path, chunk_s=cfg.realtime.step_s)
    mon = RealtimeMonitor(cfg, detectors=[IctalRhythmDetector()])
    summary = mon.run(src)
    dets = [
        Detection(a.t_start, a.t_end, a.confidence, t_known=a.t_end)
        for a in summary.alarms
        if a.type == AlarmType.SEIZURE
    ]
    return dets, summary.duration_s


def evaluate_realtime(
    items: list[tuple[RecordAnnotation, Path]],
    cfg: ConfigBundle,
    rules: ScoringRules | None = None,
    progress: Optional[Callable[[str], None]] = None,
) -> EvalResult:
    rules = rules or ScoringRules()
    t0 = perf_counter()
    scores = []
    for ann, path in items:
        dets, dur = realtime_detections(path, cfg)
        scores.append(score_record(
            ann.file, ann.seizures, dets, dur, rules.pre_s, rules.post_s, rules.merge_gap_s,
        ))
        if progress:
            progress(f"realtime {ann.file}: tp={scores[-1].tp} fp={scores[-1].fp}")
    return EvalResult("realtime", scores, aggregate(scores), perf_counter() - t0)


def load_items(
    database: str, subjects: list[str], cache_dir: str | Path | None = None,
    only_cached: bool = True,
) -> list[tuple[RecordAnnotation, Path]]:
    """Annotations + local paths for subjects (cached files only by default)."""
    from ..datasets.physionet import PhysioNetClient

    client = PhysioNetClient(database, cache_dir)
    items = []
    for sub in subjects:
        for ann in client.annotations(sub):
            if only_cached and not client.is_cached(ann.file):
                continue
            items.append((ann, client.local_path(ann.file)))
    return items


def default_config() -> ConfigBundle:
    return load_configs()
