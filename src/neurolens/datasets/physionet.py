"""Minimal PhysioNet open-access client: list, download, cache, verify.

Uses only the standard library (urllib) so the core install gains no
dependency. Downloads stream to ``<file>.part`` and are atomically renamed
after the SHA-256 matches the database's published ``SHA256SUMS.txt``; a
corrupt or truncated file is never left in the cache under its real name.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .annotations import RecordAnnotation
from .chbmit import parse_chbmit_summary
from .siena import parse_siena_seizure_list

BASE_URL = "https://physionet.org/files"
DEFAULT_CACHE = Path(os.environ.get("NEUROLENS_DATA", "data/physionet"))


@dataclass(frozen=True)
class DatabaseInfo:
    slug: str
    version: str
    citation: str
    license: str

    @property
    def root(self) -> str:
        return f"{self.slug}/{self.version}"


DATABASES: dict[str, DatabaseInfo] = {
    "chbmit": DatabaseInfo(
        "chbmit", "1.0.0",
        "Shoeb A. Application of Machine Learning to Epileptic Seizure Onset Detection "
        "and Treatment. PhD Thesis, MIT, 2009; Goldberger AL et al. PhysioBank, "
        "PhysioToolkit, and PhysioNet. Circulation 101(23):e215-e220, 2000.",
        "Open Data Commons Attribution License v1.0",
    ),
    "siena": DatabaseInfo(
        "siena-scalp-eeg", "1.0.0",
        "Detti P et al. Siena Scalp EEG Database (v1.0.0). PhysioNet, 2020; "
        "Detti P et al. EEG Synchronization Analysis for Seizure Prediction: "
        "A Study on Data of Noninvasive Recordings. Processes 8(7):846, 2020.",
        "Creative Commons Attribution 4.0 International",
    ),
}

Fetcher = Callable[[str], bytes]


class ChecksumError(RuntimeError):
    pass


SEGMENT_MIN_BYTES = 64 * 1024 * 1024  # split files larger than this
SEGMENTS = 4  # parallel byte ranges per large file
# Global cap on simultaneous HTTP transfers across all files and segments.
# PhysioNet stalls/resets connections when a client opens too many at once
# (observed at ~18), so parallelism is bounded here, not per call site.
MAX_CONNECTIONS = int(os.environ.get("NEUROLENS_MAX_CONNECTIONS", "6"))
_CONN = threading.BoundedSemaphore(MAX_CONNECTIONS)


class RangeUnsupported(RuntimeError):
    pass


def _http_get(url: str, dest: Optional[Path] = None, retries: int = 4) -> bytes:
    if dest is not None:
        size = _head_size(url)
        if size and size >= SEGMENT_MIN_BYTES and SEGMENTS > 1:
            try:
                _download_segmented(url, dest, size, SEGMENTS, retries)
                return b""
            except RangeUnsupported:
                pass
        _download_resumable(url, dest, retries)
        return b""
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "neurolens/0.1"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except Exception:
            if attempt == retries:
                raise
            time.sleep(delay)
            delay *= 2
    return b""  # unreachable


def _head_size(url: str) -> Optional[int]:
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "neurolens/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            n = resp.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:
        return None


def _download_resumable(
    url: str, dest: Path, retries: int = 8, max_resumes: int = 200,
    start: int = 0, end: Optional[int] = None,
) -> None:
    """Stream to ``dest``; when a proxy silently cuts a long transfer (stream ends
    early without an error), continue with an HTTP Range request from the bytes
    already on disk. With ``start``/``end`` only that byte range (inclusive) is
    fetched — the building block of segmented downloads. Integrity is checked by
    SHA-256 afterwards."""
    ranged = start > 0 or end is not None
    total: Optional[int] = None if end is None else end - start + 1
    dest.write_bytes(b"")
    failures = 0
    delay = 2.0
    for _ in range(max_resumes):
        have = dest.stat().st_size
        if total is not None and have >= total:
            return
        headers = {"User-Agent": "neurolens/0.1"}
        if have or ranged:
            headers["Range"] = f"bytes={start + have}-" + ("" if end is None else str(end))
        try:
            req = urllib.request.Request(url, headers=headers)
            with _CONN, urllib.request.urlopen(req, timeout=120) as resp:
                if (have or ranged) and resp.status != 206:
                    if ranged:
                        raise RangeUnsupported(url)
                    dest.write_bytes(b"")  # server ignored Range: restart
                    have = 0
                if total is None:
                    cl = resp.headers.get("Content-Length")
                    total = int(cl) + have if cl else None
                with open(dest, "ab") as fh:
                    # never write past the requested range, even if a server or
                    # proxy ignores the range end and keeps sending
                    left = None if total is None else total - have
                    while left is None or left > 0:
                        chunk = resp.read(1 << 20 if left is None else min(1 << 20, left))
                        if not chunk:
                            break
                        fh.write(chunk)
                        if left is not None:
                            left -= len(chunk)
            failures = 0
            if total is None:  # no length known: trust a clean end of stream
                return
        except RangeUnsupported:
            raise
        except Exception:
            failures += 1
            if failures > retries:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 120.0)  # a throttling server needs real back-off
    raise RuntimeError(f"{url}: download did not complete after {max_resumes} resumes")


def _download_segmented(url: str, dest: Path, size: int, segments: int = SEGMENTS,
                        retries: int = 4) -> None:
    """Fetch ``segments`` byte ranges in parallel (PhysioNet throttles per
    connection), each resumable, then concatenate in order."""
    from concurrent.futures import ThreadPoolExecutor

    step = -(-size // segments)
    bounds = [(i * step, min(size, (i + 1) * step) - 1) for i in range(segments) if i * step < size]
    parts = [dest.with_name(f"{dest.name}.seg{i}") for i in range(len(bounds))]
    try:
        with ThreadPoolExecutor(max_workers=len(bounds)) as pool:
            futs = [pool.submit(_download_resumable, url, p, retries, 200, a, b)
                    for p, (a, b) in zip(parts, bounds)]
            for f in futs:
                f.result()
        with open(dest, "wb") as out:
            for p in parts:
                with open(p, "rb") as fh:
                    while chunk := fh.read(1 << 20):
                        out.write(chunk)
    finally:
        for p in parts:
            p.unlink(missing_ok=True)


class PhysioNetClient:
    def __init__(
        self,
        database: str = "chbmit",
        cache_dir: str | Path | None = None,
        base_url: str = BASE_URL,
        fetch_text: Fetcher | None = None,
        fetch_file: Callable[[str, Path], None] | None = None,
    ):
        if database not in DATABASES:
            raise ValueError(f"unknown database {database!r}; known: {sorted(DATABASES)}")
        self.db = DATABASES[database]
        self.name = database
        self.cache = Path(cache_dir or DEFAULT_CACHE) / database
        self.base = f"{base_url}/{self.db.root}"
        self._fetch_text = fetch_text or (lambda url: _http_get(url))
        self._fetch_file = fetch_file or (lambda url, dest: _http_get(url, dest))
        self._sums: dict[str, str] | None = None

    # -- text resources (cached) -----------------------------------------
    def text(self, rel: str) -> str:
        path = self.cache / rel
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            data = self._fetch_text(f"{self.base}/{rel}")
            path.write_bytes(data)
        return path.read_text(encoding="utf-8", errors="replace")

    def checksums(self) -> dict[str, str]:
        if self._sums is None:
            sums: dict[str, str] = {}
            for line in self.text("SHA256SUMS.txt").splitlines():
                parts = line.split()
                if len(parts) == 2:
                    sums[parts[1]] = parts[0]
            self._sums = sums
        return self._sums

    def records(self) -> list[str]:
        return [r.strip() for r in self.text("RECORDS").splitlines() if r.strip()]

    def subjects(self) -> list[str]:
        return sorted({r.split("/")[0] for r in self.records()})

    # -- annotations -----------------------------------------------------
    def annotations(self, subject: str) -> list[RecordAnnotation]:
        if self.name == "chbmit":
            return parse_chbmit_summary(self.text(f"{subject}/{subject}-summary.txt"), subject)
        if self.name == "siena":
            anns = parse_siena_seizure_list(
                self.text(f"{subject}/Seizures-list-{subject}.txt"), subject
            )
            return _resolve_files(anns, [r for r in self.records() if r.startswith(subject + "/")])
        raise NotImplementedError(self.name)

    def subject_ages(self) -> dict[str, float]:
        """Patient age in years per subject (CHB-MIT SUBJECT-INFO, Siena
        subject_info.csv). Subjects without a listed age are absent."""
        ages: dict[str, float] = {}
        if self.name == "chbmit":
            for line in self.text("SUBJECT-INFO").splitlines():
                parts = line.split("\t")
                if len(parts) >= 3 and parts[0].strip().startswith("chb"):
                    try:
                        ages[parts[0].strip()] = float(parts[2])
                    except ValueError:
                        pass
        elif self.name == "siena":
            for line in self.text("subject_info.csv").splitlines()[1:]:
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 2 and parts[0].startswith("PN"):
                    try:
                        ages[parts[0]] = float(parts[1])
                    except ValueError:
                        pass
        return ages

    # -- binary files ------------------------------------------------------
    def local_path(self, rel: str) -> Path:
        return self.cache / rel

    def remote_size(self, rel: str) -> int | None:
        """Size in bytes via HTTP HEAD (None if unavailable)."""
        return _head_size(f"{self.base}/{rel}")

    def is_cached(self, rel: str) -> bool:
        return self.local_path(rel).exists()

    def fetch(self, rel: str, verify: bool = True, attempts: int = 3) -> Path:
        """Download one file; a checksum mismatch (typically a transfer cut by a
        proxy) is retried before giving up."""
        for i in range(attempts):
            try:
                return self._fetch_once(rel, verify)
            except ChecksumError as exc:
                if i == attempts - 1 or "not listed" in str(exc):
                    raise
        raise AssertionError("unreachable")

    def _fetch_once(self, rel: str, verify: bool) -> Path:
        dest = self.local_path(rel)
        if dest.exists():
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        self._fetch_file(f"{self.base}/{rel}", part)
        if verify:
            expected = self.checksums().get(rel)
            if expected is None:
                part.unlink(missing_ok=True)
                raise ChecksumError(f"{rel}: not listed in SHA256SUMS.txt")
            got = sha256_file(part)
            if got != expected:
                part.unlink(missing_ok=True)
                raise ChecksumError(f"{rel}: sha256 mismatch ({got[:12]} != {expected[:12]})")
        part.replace(dest)
        return dest

    def fetch_many(
        self, rels: list[str], workers: int = 6, on_done: Callable[[str, Exception | None], None] | None = None
    ) -> dict[str, Path | Exception]:
        """Parallel fetch (PhysioNet throttles per connection, not per client)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        self.checksums()  # load once before threads start
        results: dict[str, Path | Exception] = {}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futs = {pool.submit(self.fetch, r): r for r in rels}
            for fut in as_completed(futs):
                rel = futs[fut]
                try:
                    results[rel] = fut.result()
                    err = None
                except Exception as exc:  # report, keep going
                    results[rel] = err = exc
                if on_done:
                    on_done(rel, err)
        return results


def client_for(database: str, cache_dir: str | Path | None = None):
    """Dataset client by name: PhysioNet databases or the Helsinki neonatal set (Zenodo)."""
    if database == "helsinki":
        from .helsinki import HelsinkiClient

        return HelsinkiClient(cache_dir)
    return PhysioNetClient(database, cache_dir)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _resolve_files(anns: list[RecordAnnotation], records: list[str]) -> list[RecordAnnotation]:
    """Map annotated file names onto real RECORDS entries (lists contain typos,
    e.g. Siena "PN11-.edf" for "PN11-1.edf"). Unresolvable names are kept and warned."""
    import difflib

    real = set(records)
    claimed = {a.file for a in anns if a.file in real}
    for a in anns:
        if a.file in real or not records:
            continue
        free = [r for r in records if r not in claimed]
        match = difflib.get_close_matches(a.file, free, n=1, cutoff=0.8)
        if match:
            a.warnings.append(f"annotated file {a.file!r} resolved to {match[0]!r}")
            a.file = match[0]
            claimed.add(match[0])
        else:
            a.warnings.append(f"annotated file {a.file!r} not found in RECORDS")
    return anns
