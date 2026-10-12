"""IFCN interictal epileptiform discharge (IED) criteria validation dataset
(Kural, Duez, Sejer Hansen, Larsson, Rampp, Schulz, Tankisi, Wennberg, Bibby, Scherg,
Beniczky; Neurology 2020; Zenodo record 4688758, CC0).

100 patients (one EEG segment each, Danish centre) whose EEG contained a sharp
transient. Gold standard = clinical diagnosis from video-EEG recording of the
patient's habitual paroxysmal events: 54 "Epileptic" (the transient is an IED),
46 "Non_Epileptic" (a benign sharp transient over-read as epileptiform is the
classic cause of misdiagnosis). Seven raters in the original study; expert scoring
reached 92% accuracy, five IFCN criteria 95.7% specificity / 81.5% sensitivity.

Used by NeuroLens as an **external** test of how well an IED detector separates
epileptiform from benign sharp transients (increment 16). The files were never
used for training.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ZENODO_RECORD = "4688758"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
GOLD_FILE = "Demographics_&_gold_standard.xlsx"
CITATION = ("Kural MA, Duez L, Sejer Hansen V, et al. Criteria for defining interictal "
            "epileptiform discharges in EEG: a clinical validation study. Neurology "
            "2020;94:e2139-e2147. Data: Zenodo 10.5281/zenodo.4688758.")
LICENSE = "Creative Commons Zero v1.0 Universal"

_M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


@dataclass
class KuralCase:
    sample: str            # "S1" ... "S100"
    file: str              # EDF member name (note: S80 is stored as "S80-.edf")
    gender: str
    age_years: Optional[float]
    epileptic: bool


def read_xlsx_rows(data: bytes, sheet: str = "xl/worksheets/sheet1.xml") -> list[list[Optional[str]]]:
    """Minimal .xlsx reader (shared strings + values) - avoids an openpyxl dependency."""
    z = zipfile.ZipFile(__import__("io").BytesIO(data))
    strings: list[str] = []
    if "xl/sharedStrings.xml" in z.namelist():
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        strings = ["".join(t.text or "" for t in si.iter(_M + "t")) for si in root.findall(_M + "si")]
    rows = []
    for row in ET.fromstring(z.read(sheet)).iter(_M + "row"):
        vals: list[Optional[str]] = []
        for c in row.findall(_M + "c"):
            v = c.find(_M + "v")
            if c.get("t") == "inlineStr":
                vals.append("".join(t.text or "" for t in c.iter(_M + "t")))
            elif v is None:
                vals.append(None)
            else:
                vals.append(strings[int(v.text)] if c.get("t") == "s" else v.text)
        rows.append(vals)
    return rows


def edf_name(sample: str, available: set[str]) -> str:
    """Sample "S1" -> "S01.edf"; tolerates the dataset's "S80-.edf" spelling."""
    n = int(sample.lstrip("Ss"))
    for cand in (f"S{n:02d}.edf", f"S{n}.edf", f"S{n:02d}-.edf", f"S{n}-.edf"):
        if cand in available:
            return cand
    raise KeyError(f"no EDF for {sample}")


def parse_gold(data: bytes, available: set[str]) -> list[KuralCase]:
    rows = read_xlsx_rows(data)
    head = [(h or "").strip().lower() for h in rows[0]]
    i_s, i_g, i_a, i_gold = (head.index(k) for k in ("samples", "gender", "age", "gold standard"))
    out = []
    for r in rows[1:]:
        if not r or not r[i_s]:
            continue
        gold = (r[i_gold] or "").strip().lower()
        if gold not in ("epileptic", "non_epileptic"):
            raise ValueError(f"unexpected gold standard {r[i_gold]!r}")
        age = float(r[i_a]) if r[i_a] not in (None, "") else None
        out.append(KuralCase(r[i_s].strip(), edf_name(r[i_s], available), (r[i_g] or "").strip(),
                             age, gold == "epileptic"))
    return out


class KuralClient:
    """Downloads the record's EDFs and gold standard with MD5 verification and retries."""

    def __init__(self, cache_dir: str | Path = "data/kural_ied", retries: int = 6):
        self.cache = Path(cache_dir)
        self.retries = retries
        self._meta: Optional[dict] = None

    def meta(self) -> dict:
        p = self.cache / "record.json"
        if self._meta is None:
            if p.exists():
                self._meta = json.loads(p.read_text(encoding="utf-8"))
            else:
                self._meta = json.loads(self._get(ZENODO_API))
                self.cache.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(self._meta), encoding="utf-8")
        return self._meta

    def _get(self, url: str) -> bytes:
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    return r.read()
            except Exception:
                if attempt == self.retries - 1:
                    raise
                time.sleep(min(30, 2 ** (attempt + 1)))
        raise RuntimeError("unreachable")

    def fetch(self, key: str) -> Path:
        f = next(x for x in self.meta()["files"] if x["key"] == key)
        md5 = f["checksum"].split(":", 1)[1]
        dest = self.cache / key
        if dest.exists() and hashlib.md5(dest.read_bytes()).hexdigest() == md5:
            return dest
        for attempt in range(self.retries):
            data = self._get(f["links"]["self"])
            if hashlib.md5(data).hexdigest() == md5:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                return dest
            time.sleep(2)
        raise IOError(f"MD5 mismatch for {key}")

    def edf_keys(self) -> set[str]:
        return {f["key"] for f in self.meta()["files"] if f["key"].lower().endswith(".edf")}

    def cases(self) -> list[KuralCase]:
        return parse_gold(self.fetch(GOLD_FILE).read_bytes(), self.edf_keys())
