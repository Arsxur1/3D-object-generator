"""NMT-Events scalp EEG dataset (Alqarni et al., Zenodo record 21630052, CC BY 4.0).

1,075 routine scalp EEG recordings (Rawalpindi, Pakistan; Contec KT88-2400, 200 Hz,
21 channels referred to A1/A2, ~17 min each): 962 reported normal and 113 abnormal.
For each abnormal recording a CSV lists events with start/end wall-clock time
(``HH:MM:SS[:mmm]``, relative to the ``File Start`` clock time), the channels
involved and a free-text comment: "spike and wave", "sharp and slow wave(s)",
"sharp wave(s)", "polyspikes (and wave)" - **epileptiform** discharges - and
"delta (slow) waves" - **non-epileptiform** slowing.

The dataset is a single 10.3 GB zip. :class:`NMTEventsClient` reads it with HTTP
range requests, so only the members actually needed (annotations, the records of
one split) are downloaded; each member is checked against the CRC-32 stored in
the zip directory (``zipfile`` verifies it while extracting). The zip-level MD5
published by Zenodo can only be checked on a full download.

Note: the dataset description states an age range of 6-93 years, but some CSVs
give younger ages (e.g. "3 years", "8 Months"); the CSV value is used as given.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ZENODO_RECORD = "21630052"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
ZIP_KEY = "NMT-Events.zip"
ZIP_URL = f"{ZENODO_API}/files/{ZIP_KEY}/content"
ZIP_SIZE = 10283989823
ZIP_MD5 = "dee076d56da0c2f144f0ef9c9d06ee39"
CITATION = ("Alqarni MA, Masood H, Khan HA, Kamboh AM, Shafait S, Shafait F. NMT-Events: "
            "A Scalp EEG Dataset with Fine-Grained Temporal and Channel-Level Annotations "
            "of Abnormal Events. Zenodo, 2024. doi:10.5281/zenodo.21630052.")
LICENSE = "Creative Commons Attribution 4.0 International"

# The EDF headers label the samples "uV" with a physical range of about ±2.7, but the
# band-passed signal then has a robust SD of ~0.17 "µV" (awake adult scalp EEG: ~10-30
# µV). Unit errors are decimal; x10 gives 1.7 µV (still implausible), x100 gives ~17 µV,
# x1000 ~170 µV. The samples are therefore treated as units of 10 µV (D43). This was
# decided from training-split normal recordings only, without using any annotation.
AMPLITUDE_SCALE = 100.0

EPILEPTIFORM = "epileptiform"
SLOWING = "slowing"
OTHER = "other"


# ---------------------------------------------------------------------------
# annotations
# ---------------------------------------------------------------------------

@dataclass
class NMTEvent:
    onset_s: float
    offset_s: float
    channels: list[str]
    comment: str
    category: str


@dataclass
class NMTAnnotation:
    record_id: int
    gender: str = ""
    age_years: Optional[float] = None
    file_start_s: float = 0.0
    events: list[NMTEvent] = field(default_factory=list)

    def intervals(self, category: str = EPILEPTIFORM) -> list[tuple[float, float]]:
        return [(e.onset_s, e.offset_s) for e in self.events if e.category == category]


def category(comment: str) -> str:
    """'spike', 'sharp' or 'poly(spike)' anywhere -> epileptiform; otherwise slow/delta
    -> slowing. "sharp and delta slow waves" is therefore epileptiform."""
    c = comment.strip().lower()
    if any(w in c for w in ("spike", "sharp", "poly")):
        return EPILEPTIFORM
    if "slow" in c or "delta" in c:
        return SLOWING
    return OTHER


def parse_clock(s: str) -> float:
    """``HH:MM:SS`` or ``HH:MM:SS:mmm`` -> seconds since midnight."""
    p = s.strip().split(":")
    if len(p) < 3:
        raise ValueError(f"bad clock time {s!r}")
    sec = int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])
    if len(p) > 3 and p[3].strip():
        sec += float(p[3]) / 1000.0
    return sec


def parse_age(s: str) -> Optional[float]:
    m = re.match(r"\s*([\d.]+)\s*(year|month|week|day)", s.strip().lower())
    if not m:
        return None
    v = float(m.group(1))
    return {"year": v, "month": v / 12.0, "week": v / 52.0, "day": v / 365.0}[m.group(2)]


def parse_events_csv(text: str, record_id: int) -> NMTAnnotation:
    rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
    if not rows:
        return NMTAnnotation(record_id)
    head = [h.strip().lower() for h in rows[0]]
    col = {name: head.index(name) for name in
           ("gender", "age", "file start", "start time", "end time", "channel names", "comment")}
    first = rows[1] if len(rows) > 1 else []
    ann = NMTAnnotation(
        record_id,
        gender=first[col["gender"]].strip() if first else "",
        age_years=parse_age(first[col["age"]]) if first else None,
        file_start_s=parse_clock(first[col["file start"]]) if first else 0.0,
    )
    for r in rows[1:]:
        a = parse_clock(r[col["start time"]]) - ann.file_start_s
        b = parse_clock(r[col["end time"]]) - ann.file_start_s
        if a < -43200:  # recording crossed midnight
            a += 86400
        if b < -43200:
            b += 86400
        if b < a:  # a few rows have end < start (annotation slip): keep as a point event
            b = a
        comment = r[col["comment"]].strip() if col["comment"] < len(r) else ""
        ann.events.append(NMTEvent(round(a, 3), round(b, 3),
                                   r[col["channel names"]].split(), comment, category(comment)))
    return ann


# ---------------------------------------------------------------------------
# split (fixed before any EEG was analysed - increment-15 pre-registration)
# ---------------------------------------------------------------------------

def split_by_rule(record_id: int, normal: bool) -> Optional[str]:
    """Abnormal: even id -> train, odd -> test. Normal: id % 8 == 0 -> train,
    id % 8 == 1 -> test, otherwise unused (keeps download and disk bounded)."""
    if not normal:
        return "train" if record_id % 2 == 0 else "test"
    return {0: "train", 1: "test"}.get(record_id % 8)


# ---------------------------------------------------------------------------
# remote zip access
# ---------------------------------------------------------------------------

class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file through HTTP range requests."""

    def __init__(self, url: str, size: int, retries: int = 5, timeout: float = 120.0):
        self.url, self.size, self.pos = url, size, 0
        self.retries, self.timeout = retries, timeout

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b) -> int:
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    if r.status != 206:
                        raise IOError(f"range request not honoured (HTTP {r.status})")
                    data = r.read()
                break
            except Exception:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** (attempt + 1))
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


class NMTEventsClient:
    """Fetches individual members of the NMT-Events zip into ``cache_dir``."""

    def __init__(self, cache_dir: str | Path = "data/nmt_events", url: str = ZIP_URL,
                 size: int = ZIP_SIZE):
        self.cache = Path(cache_dir)
        self.url, self.size = url, size
        self._zip: Optional[zipfile.ZipFile] = None

    def _open(self) -> zipfile.ZipFile:
        if self._zip is None:
            raw = HttpRangeFile(self.url, self.size)
            self._zip = zipfile.ZipFile(io.BufferedReader(raw, buffer_size=1 << 20))
        return self._zip

    def index(self) -> dict[str, list[int]]:
        """Member name -> [size, crc32]; cached as ``index.json``."""
        p = self.cache / "index.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        idx = {i.filename: [i.file_size, i.CRC] for i in self._open().infolist()}
        self.cache.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(idx, indent=0, sort_keys=True), encoding="utf-8")
        return idx

    def records(self) -> list[tuple[int, bool, str]]:
        """(record id, is_normal, member name) for every EDF."""
        out = []
        for name in self.index():
            if name.lower().endswith(".edf"):
                out.append((int(Path(name).stem), "Normal EDF" in name, name))
        return sorted(out)

    def local_path(self, member: str) -> Path:
        parts = Path(member).parts
        sub = "normal" if "Normal EDF" in member else ("abnormal" if "Abnormal EDF" in member else "csv")
        return self.cache / sub / parts[-1]

    def fetch(self, member: str) -> Path:
        dest = self.local_path(member)
        size = self.index()[member][0]
        if dest.exists() and dest.stat().st_size == size:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with self._open().open(member) as src, open(tmp, "wb") as out:  # CRC checked at EOF
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp, dest)
        return dest

    def annotations(self) -> dict[int, NMTAnnotation]:
        out = {}
        for name in self.index():
            if name.lower().endswith(".csv"):
                p = self.fetch(name)
                rid = int(Path(name).stem)
                out[rid] = parse_events_csv(p.read_text(encoding="utf-8-sig", errors="replace"), rid)
        return out

    def split_members(self, split: str) -> list[tuple[int, bool, str]]:
        return [(rid, normal, m) for rid, normal, m in self.records()
                if split_by_rule(rid, normal) == split]
