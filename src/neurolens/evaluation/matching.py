"""Event-level seizure scoring (TZ §16) — "any-overlap" (OVLP) convention.

A true seizure is detected (TP) if at least one detection overlaps the
tolerance-extended interval [onset - pre_s, offset + post_s]. Detections that
overlap no seizure are false positives; consecutive detections closer than
``merge_gap_s`` are merged first so one long false run counts once. This is the
scoring used by most CHB-MIT literature (Shoeb 2009; Ziyabari et al. 2017) and
makes sensitivity / FA-per-hour comparable to published numbers.

Latency is measured from expert onset to the *time the system could know*:
detection start for offline analysis, alarm window end for streaming alarms.
Negative latencies (detection inside the pre-onset tolerance) are kept — they
indicate the detector fired on pre-ictal changes or the label is late.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..datasets.annotations import SeizureInterval


@dataclass(frozen=True)
class Detection:
    t_start: float
    t_end: float
    confidence: float = 1.0
    t_known: float | None = None  # when the system emitted it (default: t_start)

    @property
    def known_at(self) -> float:
        return self.t_start if self.t_known is None else self.t_known


@dataclass
class RecordScore:
    file: str
    duration_s: float
    n_seizures: int = 0
    tp: int = 0
    fn: int = 0
    fp: int = 0
    latencies_s: list[float] = field(default_factory=list)
    missed: list[SeizureInterval] = field(default_factory=list)
    false_positives: list[Detection] = field(default_factory=list)

    @property
    def hours(self) -> float:
        return self.duration_s / 3600.0


def merge_detections(dets: list[Detection], merge_gap_s: float = 10.0) -> list[Detection]:
    if not dets:
        return []
    s = sorted(dets, key=lambda d: d.t_start)
    out = [s[0]]
    for d in s[1:]:
        last = out[-1]
        if d.t_start <= last.t_end + merge_gap_s:
            out[-1] = Detection(
                last.t_start, max(last.t_end, d.t_end),
                max(last.confidence, d.confidence),
                min(last.known_at, d.known_at),
            )
        else:
            out.append(d)
    return out


def score_record(
    file: str,
    truth: list[SeizureInterval],
    detections: list[Detection],
    duration_s: float,
    pre_s: float = 30.0,
    post_s: float = 60.0,
    merge_gap_s: float = 10.0,
) -> RecordScore:
    dets = merge_detections(detections, merge_gap_s)
    sc = RecordScore(file=file, duration_s=duration_s, n_seizures=len(truth))
    used = [False] * len(dets)
    for sz in truth:
        lo, hi = sz.onset_s - pre_s, sz.offset_s + post_s
        hits = [i for i, d in enumerate(dets) if d.t_end >= lo and d.t_start <= hi]
        if hits:
            sc.tp += 1
            for i in hits:
                used[i] = True
            sc.latencies_s.append(min(dets[i].known_at for i in hits) - sz.onset_s)
        else:
            sc.fn += 1
            sc.missed.append(sz)
    for i, d in enumerate(dets):
        if not used[i]:
            sc.fp += 1
            sc.false_positives.append(d)
    return sc


@dataclass
class AggregateScore:
    n_records: int
    hours: float
    n_seizures: int
    tp: int
    fn: int
    fp: int
    sensitivity: float
    fa_per_hour: float
    latency_median_s: float | None
    latency_p90_s: float | None
    f1_event: float

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def aggregate(scores: list[RecordScore]) -> AggregateScore:
    hours = sum(s.hours for s in scores)
    tp = sum(s.tp for s in scores)
    fn = sum(s.fn for s in scores)
    fp = sum(s.fp for s in scores)
    n_sz = tp + fn
    lat = np.asarray([x for s in scores for x in s.latencies_s], dtype=float)
    sens = tp / n_sz if n_sz else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 and n_sz else 0.0
    return AggregateScore(
        n_records=len(scores), hours=hours, n_seizures=n_sz, tp=tp, fn=fn, fp=fp,
        sensitivity=sens, fa_per_hour=fp / hours if hours else 0.0,
        latency_median_s=float(np.median(lat)) if lat.size else None,
        latency_p90_s=float(np.percentile(lat, 90)) if lat.size else None,
        f1_event=f1,
    )
