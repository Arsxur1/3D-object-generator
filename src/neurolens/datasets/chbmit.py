"""CHB-MIT Scalp EEG summary parser (physionet.org/content/chbmit/1.0.0).

Each subject folder has ``chbNN-summary.txt`` listing, per EDF file, the number
of seizures and their onset/offset in seconds from the file start. Format
variants handled: "Seizure Start Time:" and "Seizure 1 Start Time:", and
channel-list blocks that change mid-summary (montage changes in some subjects).
"""

from __future__ import annotations

import re

from .annotations import RecordAnnotation, SeizureInterval

_FS = re.compile(r"Data Sampling Rate:\s*([\d.]+)\s*Hz", re.I)
_CHAN = re.compile(r"^Channel\s+\d+\s*:\s*(.+?)\s*$", re.I)
_FILE = re.compile(r"^File Name:\s*(\S+)", re.I)
_START = re.compile(r"^Seizure(?:\s+\d+)?\s+Start Time:\s*([\d.]+)\s*seconds", re.I)
_END = re.compile(r"^Seizure(?:\s+\d+)?\s+End Time:\s*([\d.]+)\s*seconds", re.I)
_NSZ = re.compile(r"^Number of Seizures in File:\s*(\d+)", re.I)


def parse_chbmit_summary(text: str, subject: str) -> list[RecordAnnotation]:
    fs: float | None = None
    channels: list[str] = []
    collecting_channels = False
    records: list[RecordAnnotation] = []
    cur: RecordAnnotation | None = None
    pending_start: float | None = None
    expected: dict[str, int] = {}

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if m := _FS.search(line):
            fs = float(m.group(1))
            continue
        if line.lower().startswith("channels in edf"):
            channels = []
            collecting_channels = True
            continue
        if m := _CHAN.match(line):
            if collecting_channels:
                channels.append(m.group(1))
            continue
        if line.startswith("*"):
            continue
        if m := _FILE.match(line):
            collecting_channels = False
            cur = RecordAnnotation(
                database="chbmit", subject=subject, file=f"{subject}/{m.group(1)}",
                channels=list(channels), sampling_rate_hz=fs,
            )
            records.append(cur)
            pending_start = None
            continue
        if cur is None:
            continue
        if m := _NSZ.match(line):
            expected[cur.file] = int(m.group(1))
        elif m := _START.match(line):
            pending_start = float(m.group(1))
        elif m := _END.match(line):
            end = float(m.group(1))
            if pending_start is None:
                cur.warnings.append(f"seizure end {end}s without start")
            else:
                cur.seizures.append(SeizureInterval(pending_start, end))
            pending_start = None

    for r in records:
        n = expected.get(r.file)
        if n is not None and n != len(r.seizures):
            r.warnings.append(f"summary declares {n} seizures, parsed {len(r.seizures)}")
    return records
