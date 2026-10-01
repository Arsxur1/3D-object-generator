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
    filt = apply_filters(sig, cfg.filters)
    analysis = rereference(filt, "average")
    feats = compute_features(analysis, cfg.filters)
    # the ictal detector works on features only; dropping the samples keeps the
    # on-disk feature cache ~100x smaller
    return PreparedRecord(
        annotation=ann, path=path, duration_s=sig.duration_s, features=feats,
        signal=None, flags=list(sig.quality.flags),
    )


def prepare_records(
    items: list[tuple[RecordAnnotation, Path]],
    cfg: ConfigBundle,
    cache_dir: str | Path | None = None,
    progress: Optional[Callable[[str], None]] = None,
) -> list[PreparedRecord]:
    """Prepare many records; optionally memoise features on disk.

    The cache key covers the EDF (name, size, mtime) and the filter config, so
    a filter change invalidates it. Cache files are local derived data
    (gitignored), never shipped.
    """
    import hashlib
    import pickle

    out = []
    cdir = Path(cache_dir) if cache_dir else None
    if cdir:
        cdir.mkdir(parents=True, exist_ok=True)
    fkey = hashlib.sha1(cfg.filters.model_dump_json().encode()).hexdigest()[:10]
    for ann, path in items:
        pk = None
        if cdir:
            st = path.stat()
            key = hashlib.sha1(f"{path.name}:{st.st_size}:{st.st_mtime_ns}:{fkey}".encode()).hexdigest()[:16]
            pk = cdir / f"{path.stem}_{key}.pkl"
            if pk.exists():
                rec = pickle.loads(pk.read_bytes())
                rec.annotation = ann  # annotations may be re-parsed/fixed
                out.append(rec)
                continue
        t0 = perf_counter()
        rec = prepare_record(ann, path, cfg)
        if pk:
            pk.write_bytes(pickle.dumps(rec))
        out.append(rec)
        if progress:
            progress(f"prepared {ann.file} in {perf_counter() - t0:.0f}s")
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
