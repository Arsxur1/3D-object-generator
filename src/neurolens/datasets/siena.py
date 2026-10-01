"""Siena Scalp EEG seizure-list parser (physionet.org/content/siena-scalp-eeg/1.0.0).

``Seizures-list-PNxx.txt`` gives wall-clock times ("19.58.36") for the
registration start and each seizure; we convert to seconds from file start,
handling midnight roll-over. Annotation typos present in the published lists
(e.g. an end time one hour off) are clamped and reported as warnings rather
than silently trusted.
"""

from __future__ import annotations

import re

from .annotations import RecordAnnotation, SeizureInterval

_FILE = re.compile(r"File name:\s*(\S+)", re.I)
_REG_START = re.compile(r"Registration start time:\s*([\d.:]+)", re.I)
_REG_END = re.compile(r"Registration end time:\s*([\d.:]+)", re.I)
_SZ_START = re.compile(r"Seizure start time:\s*([\d.:]+)", re.I)
_SZ_END = re.compile(r"Seizure end time:\s*([\d.:]+)", re.I)
_FS = re.compile(r"Data Sampling Rate:\s*([\d.]+)\s*Hz", re.I)

MAX_SEIZURE_S = 15 * 60  # longer annotated "seizures" are treated as typos


def _clock(s: str) -> int:
    parts = [int(p) for p in re.split(r"[.:]", s.strip()) if p != ""]
    h, m, sec = (parts + [0, 0, 0])[:3]
    return h * 3600 + m * 60 + sec


def _rel(t: int, ref: int) -> int:
    d = t - ref
    return d + 86400 if d < 0 else d


def parse_siena_seizure_list(text: str, subject: str) -> list[RecordAnnotation]:
    fs_m = _FS.search(text)
    fs = float(fs_m.group(1)) if fs_m else None
    by_file: dict[str, RecordAnnotation] = {}
    reg_start: dict[str, int] = {}
    cur_file: str | None = None
    sz_start: int | None = None

    for raw in text.splitlines():
        line = raw.strip()
        if m := _FILE.search(line):
            cur_file = m.group(1)
            if cur_file not in by_file:
                by_file[cur_file] = RecordAnnotation(
                    database="siena", subject=subject, file=f"{subject}/{cur_file}",
                    sampling_rate_hz=fs,
                )
            continue
        if cur_file is None:
            continue
        rec = by_file[cur_file]
        if m := _REG_START.search(line):
            reg_start[cur_file] = _clock(m.group(1))
        elif m := _REG_END.search(line):
            if cur_file in reg_start:
                rec.duration_s = float(_rel(_clock(m.group(1)), reg_start[cur_file]))
        elif m := _SZ_START.search(line):
            sz_start = _clock(m.group(1))
        elif m := _SZ_END.search(line):
            if sz_start is None or cur_file not in reg_start:
                rec.warnings.append("seizure end without start/registration time")
                continue
            ref = reg_start[cur_file]
            on = _rel(sz_start, ref)
            off = _rel(_clock(m.group(1)), ref)
            if off <= on or off - on > MAX_SEIZURE_S:
                # common typo: hour digit wrong -> try same minutes/seconds
                fixed = on + ((_clock(m.group(1)) - sz_start) % 3600)
                rec.warnings.append(
                    f"implausible seizure end ({m.group(1)}); corrected to +{fixed - on}s"
                )
                off = fixed
            rec.seizures.append(SeizureInterval(float(on), float(off)))
            sz_start = None
    return list(by_file.values())
