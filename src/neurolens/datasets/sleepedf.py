"""Sleep-EDF Expanded, sleep-cassette part (PhysioNet sleep-edfx 1.0.0).

153 whole-night PSG recordings (``SC4ssN*-PSG.edf``; ss = subject 00-82, N =
night 1/2) of 78 healthy subjects aged 25-101, with expert hypnograms
(``SC4ssN*-Hypnogram.edf``, EDF+ annotations "Sleep stage W/1/2/3/4/R/?" and
"Movement time") scored in 30-s epochs by Rechtschaffen & Kales.

Conventions used by NeuroLens (fixed in the increment-13 pre-registration):

* R&K -> AASM: stages 3 and 4 merge into N3; "?" and movement time are
  excluded from scoring (``None``).
* Analysis window: from 30 min before the first sleep epoch to 30 min after the
  last one (recordings last ~20 h; scoring the long daytime wake would inflate
  accuracy).
* Split by subject number: even -> development, odd -> test.

Citation: Kemp B, Zwinderman AH, Tuk B, Kamphuisen HAC, Oberye JJL. Analysis of a
sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG.
IEEE-BME 47(9):1185-1194 (2000); Goldberger AL et al., PhysioNet, Circulation 2000.
License: Open Data Commons Attribution License v1.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EPOCH_S = 30.0
STAGES = ("W", "N1", "N2", "N3", "REM")
_RK = {"W": "W", "1": "N1", "2": "N2", "3": "N3", "4": "N3", "R": "REM"}
TRIM_S = 30 * 60
_NAME = re.compile(r"sleep-cassette/SC4(\d\d)(\d)([A-Z0-9]{2})-(PSG|Hypnogram)\.edf$")


@dataclass
class SleepRecord:
    subject: int
    night: int
    psg: str          # path relative to the database root
    hypnogram: str
    stages: list = field(default_factory=list)   # per 30-s epoch: index into STAGES or None
    start_epoch: int = 0                          # analysis window, epochs from file start
    end_epoch: int = 0

    @property
    def key(self) -> str:
        return f"SC4{self.subject:02d}{self.night}"


def pair_records(paths: list[str]) -> list[SleepRecord]:
    """PSG <-> hypnogram by subject + night (hypnogram suffixes differ: EC, EH, EJ...)."""
    psg, hyp = {}, {}
    for p in paths:
        m = _NAME.search(p)
        if not m:
            continue
        k = (int(m.group(1)), int(m.group(2)))
        (psg if m.group(4) == "PSG" else hyp)[k] = p
    return [SleepRecord(s, n, psg[(s, n)], hyp[(s, n)]) for s, n in sorted(psg) if (s, n) in hyp]


def split_by_rule(records: list[SleepRecord]) -> dict[str, list[SleepRecord]]:
    return {"dev": [r for r in records if r.subject % 2 == 0],
            "test": [r for r in records if r.subject % 2 == 1]}


def stage_label(text: str):
    t = text.strip()
    if t.startswith("Sleep stage "):
        code = t.removeprefix("Sleep stage ").strip()
        if code in _RK:
            return STAGES.index(_RK[code])
    return None  # "?", movement time, anything else: not scored


def epochs_from_annotations(onsets, durations, texts, n_epochs: int) -> list:
    """Expand (onset, duration, label) annotations into per-epoch stage indices."""
    out: list = [None] * n_epochs
    for on, du, tx in zip(onsets, durations, texts):
        lab = stage_label(tx)
        a = int(round(float(on) / EPOCH_S))
        b = int(round((float(on) + float(du)) / EPOCH_S))
        for i in range(max(a, 0), min(b, n_epochs)):
            out[i] = lab
    return out


def analysis_window(stages: list, trim_epochs: int = int(TRIM_S / EPOCH_S)) -> tuple[int, int]:
    sleep = [i for i, s in enumerate(stages) if s is not None and s != 0]
    if not sleep:
        return 0, len(stages)
    return max(0, sleep[0] - trim_epochs), min(len(stages), sleep[-1] + 1 + trim_epochs)


def read_hypnogram(path: str | Path, n_epochs: int) -> list:
    import pyedflib

    r = pyedflib.EdfReader(str(path))
    try:
        on, du, tx = r.readAnnotations()
    finally:
        r.close()
    return epochs_from_annotations(on, du, tx, n_epochs)


def read_psg(path: str | Path, channels=("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal")) -> tuple[dict, float]:
    """Selected PSG channels (µV, float32) and their common sampling rate."""
    import pyedflib

    r = pyedflib.EdfReader(str(path))
    try:
        labels = r.getSignalLabels()
        out, fs = {}, None
        for ch in channels:
            i = labels.index(ch)
            f = r.getSampleFrequency(i)
            fs = f if fs is None else fs
            if f != fs:
                raise ValueError(f"{ch}: {f} Hz differs from {fs} Hz")
            out[ch] = r.readSignal(i).astype(np.float32)
        return out, float(fs)
    finally:
        r.close()


class SleepEDFClient:
    """PhysioNet download/verification (SHA-256) + record pairing and split."""

    def __init__(self, cache_dir: str | Path | None = None):
        from .physionet import PhysioNetClient

        self.pn = PhysioNetClient("sleepedf", cache_dir)
        self.cache = self.pn.cache

    def records(self) -> list[SleepRecord]:
        return pair_records(list(self.pn.checksums()))

    def split(self) -> dict[str, list[SleepRecord]]:
        return split_by_rule(self.records())

    def local_path(self, rel: str) -> Path:
        return self.pn.local_path(rel)

    def fetch_records(self, recs: list[SleepRecord], workers: int = 4, on_done=None) -> dict:
        rels = [p for r in recs for p in (r.psg, r.hypnogram)]
        return self.pn.fetch_many(rels, workers=workers, on_done=on_done)

    def load(self, rec: SleepRecord) -> tuple[dict, float, SleepRecord]:
        sig, fs = read_psg(self.local_path(rec.psg))
        n = int(len(next(iter(sig.values()))) / fs // EPOCH_S)
        rec.stages = read_hypnogram(self.local_path(rec.hypnogram), n)
        rec.start_epoch, rec.end_epoch = analysis_window(rec.stages)
        return sig, fs, rec
