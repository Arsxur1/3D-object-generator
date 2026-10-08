"""Helsinki neonatal EEG seizure dataset (Stevenson et al. 2019, Zenodo record 2547147).

79 term neonates (NICU, Helsinki University Hospital), one EDF each (~74 min
median), seizures annotated per second by three experts (A, B, C) in
``annotations_2017_{A,B,C}.csv``: one column per neonate, one row per second,
1 = seizure; shorter recordings are padded with empty cells.

Reference standard used by NeuroLens (documented in the increment-12
pre-registration): **consensus** seconds (all three experts), runs separated by
less than ``MERGE_GAP_S`` merged, runs shorter than ``MIN_SEIZURE_S`` dropped
(neonatal seizures are defined as >= 10 s). Seconds marked by only one or two
experts are kept as ``ambiguous`` intervals for secondary analyses.

License CC BY 4.0. Citation: Stevenson NJ, Tapani K, Lauronen L, Vanhatalo S.
A dataset of neonatal EEG recordings with seizure annotations. Sci Data 6:190039, 2019.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

from .annotations import RecordAnnotation, SeizureInterval

ZENODO_RECORD = "2547147"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
EXPERTS = ("A", "B", "C")
MIN_SEIZURE_S = 10.0
MERGE_GAP_S = 0.0  # consensus runs are taken as annotated (no gap bridging)
CITATION = ("Stevenson NJ, Tapani K, Lauronen L, Vanhatalo S. A dataset of neonatal EEG "
            "recordings with seizure annotations. Scientific Data 6:190039, 2019 "
            "(Zenodo 10.5281/zenodo.2547147).")
LICENSE = "Creative Commons Attribution 4.0 International"


def parse_expert_csv(text: str) -> dict[int, list[int]]:
    """Per-neonate list of per-second 0/1 labels (padding removed)."""
    rows = list(csv.reader(io.StringIO(text)))
    header = [int(h) for h in rows[0]]
    out: dict[int, list[int]] = {}
    for j, nid in enumerate(header):
        col = []
        for r in rows[1:]:
            if j < len(r) and r[j].strip() != "":
                col.append(1 if float(r[j]) > 0.5 else 0)
        out[nid] = col
    return out


def _runs(labels: list[int]) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, v in enumerate(labels + [0]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    return runs


def consensus_intervals(labels: list[list[int]], min_s: float = MIN_SEIZURE_S,
                        merge_gap_s: float = MERGE_GAP_S) -> list[SeizureInterval]:
    """Runs of seconds marked by every expert, merged over gaps < merge_gap_s,
    keeping runs >= min_s. Second i covers [i, i+1)."""
    n = min(len(x) for x in labels)
    cons = [int(all(x[i] for x in labels)) for i in range(n)]
    merged: list[list[int]] = []
    for a, b in _runs(cons):
        if merged and a - merged[-1][1] < merge_gap_s:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [SeizureInterval(float(a), float(b)) for a, b in merged if b - a >= min_s]


def any_expert_intervals(labels: list[list[int]]) -> list[SeizureInterval]:
    n = max(len(x) for x in labels)
    anyv = [int(any(i < len(x) and x[i] for x in labels)) for i in range(n)]
    return [SeizureInterval(float(a), float(b)) for a, b in _runs(anyv)]


def parse_helsinki(expert_texts: dict[str, str]) -> list[RecordAnnotation]:
    experts = {k: parse_expert_csv(v) for k, v in expert_texts.items()}
    ids = sorted(set.intersection(*(set(e) for e in experts.values())))
    anns = []
    for nid in ids:
        labels = [experts[k][nid] for k in sorted(experts)]
        cons = consensus_intervals(labels)
        rec = RecordAnnotation(database="helsinki", subject=f"eeg{nid}", file=f"eeg{nid}.edf",
                               seizures=cons, sampling_rate_hz=256.0,
                               duration_s=float(min(len(x) for x in labels)),
                               ambiguous=[iv for iv in any_expert_intervals(labels)
                                          if not any(c.onset_s < iv.offset_s and iv.onset_s < c.offset_s
                                                     for c in cons)])
        if len({len(x) for x in labels}) > 1:
            rec.warnings.append(f"expert label lengths differ: {[len(x) for x in labels]}")
        anns.append(rec)
    return anns


def reviewers_with_seizure(clinical_csv: str) -> dict[str, int]:
    """'Number of Reviewers Annotating Seizure' per neonate (0-3)."""
    out = {}
    for row in csv.DictReader(io.StringIO(clinical_csv)):
        try:
            out[row["EEG file"].strip()] = int(row["Number of Reviewers Annotating Seizure"])
        except (KeyError, ValueError):
            pass
    return out


def split_by_rule(n_reviewers: dict[str, int]) -> dict[str, list[str]]:
    """Deterministic development/test split fixed before any EEG is seen:
    within each group (consensus seizures = 3 reviewers; ambiguous = 1-2;
    seizure-free = 0), neonates in ID order alternate dev, test, dev, ..."""
    groups: dict[str, list[str]] = {"consensus": [], "ambiguous": [], "free": []}
    for sub, k in sorted(n_reviewers.items(), key=lambda kv: int(kv[0].removeprefix("eeg"))):
        groups["consensus" if k == 3 else "free" if k == 0 else "ambiguous"].append(sub)
    split: dict[str, list[str]] = {"dev": [], "test": []}
    for members in groups.values():
        for i, sub in enumerate(members):
            split["dev" if i % 2 == 0 else "test"].append(sub)
    return {k: sorted(v, key=lambda s: int(s.removeprefix("eeg"))) for k, v in split.items()}


class HelsinkiClient:
    """Same interface as :class:`PhysioNetClient` for the evaluation code:
    annotations, subjects, local_path, is_cached, remote_size, fetch_many.
    Files are verified against the MD5 checksums Zenodo publishes."""

    name = "helsinki"

    def __init__(self, cache_dir: str | Path | None = None, fetch_text=None, fetch_file=None):
        from .physionet import DEFAULT_CACHE, _http_get

        self.cache = Path(cache_dir or DEFAULT_CACHE) / "helsinki"
        self._fetch_text = fetch_text or (lambda url: _http_get(url))
        self._fetch_file = fetch_file or (lambda url, dest: _http_get(url, dest))
        self._meta: dict | None = None

    def _url(self, key: str) -> str:
        return f"{ZENODO_API}/files/{key}/content"

    def meta(self) -> dict[str, dict]:
        """File key -> {size, md5}, from the Zenodo record (cached locally)."""
        if self._meta is None:
            path = self.cache / "zenodo_files.json"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                rec = json.loads(self._fetch_text(ZENODO_API))
                files = {f["key"]: {"size": f["size"], "md5": f["checksum"].removeprefix("md5:")}
                         for f in rec["files"]}
                path.write_text(json.dumps(files, indent=1), encoding="utf-8")
            self._meta = json.loads(path.read_text(encoding="utf-8"))
        return self._meta

    def text(self, key: str) -> str:
        path = self.cache / key
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            self.fetch(key)
        return path.read_text(encoding="utf-8", errors="replace")

    def subjects(self) -> list[str]:
        return sorted((k.removesuffix(".edf") for k in self.meta() if k.endswith(".edf")),
                      key=lambda s: int(s.removeprefix("eeg")))

    def all_annotations(self) -> list[RecordAnnotation]:
        if getattr(self, "_anns", None) is None:
            self._anns = parse_helsinki({e: self.text(f"annotations_2017_{e}.csv") for e in EXPERTS})
        return self._anns

    def annotations(self, subject: str) -> list[RecordAnnotation]:
        return [a for a in self.all_annotations() if a.subject == subject]

    def subject_ages(self) -> dict[str, float]:
        return {s: 0.0 for s in self.subjects()}  # term neonates

    def split(self) -> dict[str, list[str]]:
        return split_by_rule(reviewers_with_seizure(self.text("clinical_information.csv")))

    def local_path(self, rel: str) -> Path:
        return self.cache / rel

    def is_cached(self, rel: str) -> bool:
        return self.local_path(rel).exists()

    def remote_size(self, rel: str) -> int | None:
        return self.meta().get(rel, {}).get("size")

    def fetch(self, rel: str, verify: bool = True, attempts: int = 3) -> Path:
        from .physionet import ChecksumError

        dest = self.local_path(rel)
        if dest.exists():
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        expected = self.meta().get(rel, {}).get("md5")
        if verify and expected is None:
            raise ChecksumError(f"{rel}: not listed in the Zenodo record")
        for i in range(attempts):
            part = dest.with_suffix(dest.suffix + ".part")
            self._fetch_file(self._url(rel), part)
            if not verify or md5_file(part) == expected:
                part.replace(dest)
                return dest
            part.unlink(missing_ok=True)
            if i == attempts - 1:
                raise ChecksumError(f"{rel}: md5 mismatch")
        raise AssertionError("unreachable")

    def fetch_many(self, rels: list[str], workers: int = 4, on_done=None) -> dict:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        self.meta()
        results: dict = {}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futs = {pool.submit(self.fetch, r): r for r in rels}
            for fut in as_completed(futs):
                rel = futs[fut]
                try:
                    results[rel] = fut.result()
                    err = None
                except Exception as exc:
                    results[rel] = err = exc
                if on_done:
                    on_done(rel, err)
        return results


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()
