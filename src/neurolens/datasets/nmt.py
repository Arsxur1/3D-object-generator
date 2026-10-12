"""NMT Scalp EEG Dataset (NUST-MH-TUKL; Khan et al., Front Neurosci 2022;
Zenodo record 10909103, CC BY 4.0).

2,417 routine scalp EEG recordings (Rawalpindi, Pakistan; same acquisition and
record numbering as NMT-Events, of which NMT-Events is a subset: record 0000004
is "abnormal, 12, male" in both), each labelled **normal** or **abnormal** by a
team of neurologists; age and gender in ``Labels.csv``; the authors' own
train/eval split (1,877 + 95 normal, 355 + 90 abnormal). 200 Hz, 19 channels +
A1/A2; same unit error as NMT-Events (samples labelled "uV" are 10 µV units, D43).

Read with HTTP range requests like NMT-Events (13.4 GB single zip).
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .nmt_events import AMPLITUDE_SCALE, NMTEventsClient  # noqa: F401  (same unit correction)

ZENODO_RECORD = "10909103"
ZIP_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/NMT.zip/content"
ZIP_SIZE = 13415695628
ZIP_MD5 = "32c3ee30bfc5cc402880d93cad07c9fb"
ROOT = "nmt_scalp_eeg_dataset"
CITATION = ("Khan HA, Ul Ain R, Kamboh AM, Butt HT, Shafait S, Alamgir W, Stricker D, Shafait F. "
            "The NMT Scalp EEG Dataset: An Open-Source Annotated Dataset of Healthy and "
            "Pathological EEG Recordings for Predictive Modeling. Front Neurosci 15:755817, 2022. "
            "Data: Zenodo 10.5281/zenodo.10909103.")
LICENSE = "Creative Commons Attribution 4.0 International"


@dataclass
class NMTRecord:
    record_id: int
    abnormal: bool
    age_years: Optional[float]
    gender: str
    loc: str  # authors' split: "train" | "eval"

    @property
    def member(self) -> str:
        return f"{ROOT}/{'abnormal' if self.abnormal else 'normal'}/{self.loc}/{self.record_id:07d}.edf"


def parse_labels(text: str) -> list[NMTRecord]:
    out = []
    for row in csv.DictReader(io.StringIO(text)):
        name = row["recordname"].strip()
        if not name:
            continue
        lab = row["label"].strip().lower()
        if lab not in ("normal", "abnormal"):
            raise ValueError(f"unexpected label {lab!r}")
        try:
            age = float(row["age"])
        except (TypeError, ValueError):
            age = None
        out.append(NMTRecord(int(Path(name).stem), lab == "abnormal", age,
                             row.get("gender", "").strip(), row["loc"].strip()))
    return out


class NMTClient(NMTEventsClient):
    """Fetch members of the NMT zip (same mechanics as NMT-Events)."""

    def __init__(self, cache_dir: str | Path = "data/nmt", url: str = ZIP_URL, size: int = ZIP_SIZE):
        super().__init__(cache_dir, url, size)

    def local_path(self, member: str) -> Path:
        parts = Path(member).parts
        if member.lower().endswith(".edf"):
            return self.cache / parts[-3] / parts[-1]  # normal|abnormal / id.edf
        return self.cache / parts[-1]

    def labels(self) -> list[NMTRecord]:
        p = self.fetch(f"{ROOT}/Labels.csv")
        return parse_labels(p.read_text(encoding="utf-8-sig", errors="replace"))
