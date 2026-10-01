"""Minimal PhysioNet open-access client: list, download, cache, verify.

Uses only the standard library (urllib) so the core install gains no
dependency. Downloads stream to ``<file>.part`` and are atomically renamed
after the SHA-256 matches the database's published ``SHA256SUMS.txt``; a
corrupt or truncated file is never left in the cache under its real name.
"""

from __future__ import annotations

import hashlib
import os
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


def _http_get(url: str, dest: Optional[Path] = None, retries: int = 4) -> bytes:
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "neurolens/0.1"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                if dest is None:
                    return resp.read()
                with open(dest, "wb") as fh:
                    while chunk := resp.read(1 << 20):
                        fh.write(chunk)
                return b""
        except Exception:
            if attempt == retries:
                raise
            time.sleep(delay)
            delay *= 2
    return b""  # unreachable


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
            return parse_siena_seizure_list(
                self.text(f"{subject}/Seizures-list-{subject}.txt"), subject
            )
        raise NotImplementedError(self.name)

    # -- binary files ------------------------------------------------------
    def local_path(self, rel: str) -> Path:
        return self.cache / rel

    def is_cached(self, rel: str) -> bool:
        return self.local_path(rel).exists()

    def fetch(self, rel: str, verify: bool = True) -> Path:
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()
