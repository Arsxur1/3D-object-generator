"""Routine-EEG specificity: does the offline pipeline call neurologist-normal
recordings abnormal? (increment 17, NMT Scalp EEG Dataset).

For each recording the full pipeline (template provider, no ICA) is run with the
patient's age; the harness records which detectors fired, which causal-graph nodes
were judged pathologic (they decide the report's impression), and a few background
features. Aggregates: per-detector firing rate in normal vs abnormal recordings and
the impression's sensitivity/specificity for the neurologists' label.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from ..datasets.nmt import AMPLITUDE_SCALE, NMTRecord


def diagnostic_subset(records: list[NMTRecord], n_normal: int, n_abnormal: int, seed: int = 17) -> list[NMTRecord]:
    """Seeded random subset of the authors' *train* split (the eval split is never touched)."""
    rng = random.Random(seed)
    ok = [r for r in records if r.loc == "train" and not r.label_conflict]
    tr_n = sorted((r for r in ok if not r.abnormal), key=lambda r: r.record_id)
    tr_a = sorted((r for r in ok if r.abnormal), key=lambda r: r.record_id)
    return rng.sample(tr_n, min(n_normal, len(tr_n))) + rng.sample(tr_a, min(n_abnormal, len(tr_a)))


def fresh_subset(records: list[NMTRecord], n_normal: int, n_abnormal: int, seed: int = 18,
                 dev_seed: int = 17, dev_sizes: tuple[int, int] = (120, 80)) -> list[NMTRecord]:
    """Seeded subset of the authors' train split that excludes the development subset
    (``diagnostic_subset`` with ``dev_seed``) - records never run before (increment 18)."""
    dev = {r.record_id for r in diagnostic_subset(records, *dev_sizes, seed=dev_seed)}
    rest = [r for r in records if r.record_id not in dev]
    return diagnostic_subset(rest, n_normal, n_abnormal, seed=seed)


def run_record(path: Path, rec: NMTRecord, scale: float = AMPLITUDE_SCALE, cfg=None) -> dict:
    from ..layer1_ingest.registry import ingest
    from ..layer2_preprocess.sqi import robust_amplitude_uv
    from ..pipeline.pipeline import Pipeline

    sig = ingest(path)
    sig = sig.with_signal((sig.signal * scale).astype(np.float32))
    if rec.age_years is not None:
        sig = sig.model_copy(update={"patient": sig.patient.model_copy(update={"age_years": rec.age_years})})
    out = Pipeline(cfg, provider_pref="template", run_ica=False).analyze_signal(sig)
    patho = sorted({n.event_code or n.id for n in out.graph.nodes
                    if getattr(n.physiology, "value", n.physiology) == "pathologic"})
    f = out.features
    return {
        "record_id": rec.record_id, "abnormal": rec.abnormal, "age_years": rec.age_years,
        "detectors_fired": sorted({e.code for e in out.detection.events}),
        "pathologic_nodes": patho,
        "impression_abnormal": bool(patho),
        "pdr_hz": float(f.pdr_hz),
        "global_rel_delta": float(f.global_relpow["delta"]),
        "global_rel_theta": float(f.global_relpow["theta"]),
        "global_rel_alpha": float(f.global_relpow["alpha"]),
        "robust_amplitude_uv": robust_amplitude_uv(out.signal),
        "flags": list(out.signal.quality.flags),
        "duration_s": float(f.duration_s),
        "events": [{"code": e.code, "t_start": e.t_start, "t_end": e.t_end,
                    "confidence": round(float(e.confidence), 3),
                    "channels": list(e.localization.channels)[:20],
                    "evidence": {ev.feature: ev.value for ev in e.evidence}} for e in out.detection.events],
    }


def summarize(rows: list[dict]) -> dict:
    nm = [r for r in rows if not r["abnormal"]]
    ab = [r for r in rows if r["abnormal"]]
    det = Counter(c for r in nm for c in r["detectors_fired"])
    deta = Counter(c for r in ab for c in r["detectors_fired"])
    pat = Counter(c for r in nm for c in r["pathologic_nodes"])
    pata = Counter(c for r in ab for c in r["pathologic_nodes"])
    codes = sorted(set(det) | set(deta))
    tp = sum(r["impression_abnormal"] for r in ab)
    tn = sum(not r["impression_abnormal"] for r in nm)
    return {
        "n_normal": len(nm), "n_abnormal": len(ab),
        "impression_sensitivity": tp / len(ab) if ab else float("nan"),
        "impression_specificity": tn / len(nm) if nm else float("nan"),
        "detector_rate": {c: {"normal": det[c] / max(1, len(nm)), "abnormal": deta[c] / max(1, len(ab))}
                          for c in codes},
        "pathologic_node_rate": {c: {"normal": pat[c] / max(1, len(nm)), "abnormal": pata[c] / max(1, len(ab))}
                                 for c in sorted(set(pat) | set(pata))},
    }


def save(rows: list[dict], out: Path, extra: Optional[dict] = None) -> dict:
    s = summarize(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": s, **(extra or {}), "records": rows}, indent=1, default=float) + "\n",
                   encoding="utf-8")
    return s
