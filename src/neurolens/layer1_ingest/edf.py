"""EDF/EDF+ ingestor (TZ §4). Mandatory MVP input format.

Uses ``pyedflib`` (lightweight) to read samples and header, then normalizes
channel names, detects the electrode system, and builds a ``UnifiedSignal``
in microvolts. Channels with differing sample rates are resampled to the
maximum rate via linear interpolation (adequate for the skeleton; MNE-grade
resampling is a v2 option).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyedflib

from ..contracts.signal import (
    ClinicalContext,
    ElectrodeSystem,
    PatientInfo,
    Provenance,
    UnifiedSignal,
)
from .base import Ingestor
from .bipolar import is_bipolar_recording, parse_bipolar_label, reconstruct_average_reference
from .electrodes import detect_electrode_system, is_eeg_channel, normalize_channel_name

# EDF physical-dimension strings -> multiplier to microvolts.
_UNIT_TO_UV = {
    "uv": 1.0,
    "µv": 1.0,
    "μv": 1.0,
    "mv": 1e3,
    "v": 1e6,
}


def _to_microvolts(data: np.ndarray, dimension: str) -> np.ndarray:
    mult = _UNIT_TO_UV.get(dimension.strip().lower(), 1.0)
    return data * mult


class EdfIngestor(Ingestor):
    extensions = (".edf", ".bdf")
    source_name = "edf"

    def read(
        self,
        path: str | Path,
        *,
        patient: PatientInfo | None = None,
        context: ClinicalContext | None = None,
    ) -> UnifiedSignal:
        path = Path(path)
        # pyedflib rejects e.g. EDF+ headers without the annotation signal (Helsinki
        # eeg50.edf) or a record count beyond the data (CAP brux1.edf): plain EDF then
        reader = open_edf(path)
        try:
            n = reader.signals_in_file
            labels = reader.getSignalLabels()
            sample_rates = [reader.getSampleFrequency(i) for i in range(n)]
            dims = [reader.getPhysicalDimension(i) for i in range(n)]
            target_fs = float(max(sample_rates))

            channel_names: list[str] = []
            channels: list[np.ndarray] = []
            for i in range(n):
                raw = reader.readSignal(i)
                sig_uv = _to_microvolts(np.asarray(raw, dtype=np.float64), dims[i])
                if sample_rates[i] != target_fs:
                    sig_uv = _resample_linear(sig_uv, sample_rates[i], target_fs)
                channels.append(sig_uv.astype(np.float32))
                channel_names.append(normalize_channel_name(labels[i]))

            n_samples = min(len(c) for c in channels)
            signal = np.stack([c[:n_samples] for c in channels], axis=0)
            flags: list[str] = []
            reference = _guess_reference(labels)

            if is_bipolar_recording(labels):
                signal, channel_names, flags = _from_bipolar(signal, labels)
                reference = "average_from_bipolar"

            electrode_system = detect_electrode_system(channel_names)
        finally:
            reader.close()

        out = UnifiedSignal(
            signal=signal,
            sampling_rate_hz=target_fs,
            channel_names=channel_names,
            electrode_system=electrode_system,
            reference=reference,
            source=self.source_name,
            patient=patient or PatientInfo(),
            context=context or ClinicalContext(),
            provenance=Provenance(
                digitized=False, confidence=1.0, source_file=path.name,
                pipeline_version="0.1.0",
            ),
        )
        out.quality.flags.extend(flags)
        return out


def open_edf(path: str | Path):
    """pyedflib reader, or :class:`PlainEdfReader` when pyedflib rejects the file
    (non-compliant EDF+ header, or a header record count larger than the data)."""
    try:
        return pyedflib.EdfReader(str(path))
    except OSError:
        return PlainEdfReader(path)


class PlainEdfReader:
    """Minimal EDF reader (header + 16-bit data records), used only when pyedflib
    rejects a file whose samples are standard EDF but whose header is not strictly
    compliant. Same subset of the pyedflib API as used above; 'EDF Annotations'
    signals are skipped."""

    def __init__(self, path: str | Path):
        raw = Path(path).read_bytes()
        hdr_bytes = int(raw[184:192].decode("ascii").strip())
        n_records = int(raw[236:244].decode("ascii").strip())
        ns = int(raw[252:256].decode("ascii").strip())

        def field(offset: int, width: int) -> list[str]:
            base = 256 + offset * ns
            return [raw[base + i * width: base + (i + 1) * width].decode("latin-1").strip() for i in range(ns)]

        # per-signal fields, each stored for all signals in turn (byte widths:
        # label 16, transducer 80, dimension 8, pmin/pmax/dmin/dmax 8, prefilter 80, samples 8)
        labels, dims = field(0, 16), field(96, 8)
        pmin, pmax = [float(x) for x in field(104, 8)], [float(x) for x in field(112, 8)]
        dmin, dmax = [float(x) for x in field(120, 8)], [float(x) for x in field(128, 8)]
        nsamp = [int(x) for x in field(216, 8)]
        duration = float(raw[244:252].decode("ascii").strip() or 1.0)
        rec_len = sum(nsamp)
        avail = (len(raw) - hdr_bytes) // (2 * rec_len)
        n_records = avail if n_records < 0 else min(n_records, avail)
        data = np.frombuffer(raw, dtype="<i2", count=n_records * rec_len, offset=hdr_bytes)
        data = data.reshape(n_records, rec_len)
        keep = [i for i in range(ns) if labels[i] != "EDF Annotations"]
        starts = np.concatenate([[0], np.cumsum(nsamp)])
        self._labels = [labels[i] for i in keep]
        self._dims = [dims[i] for i in keep]
        self._fs = [nsamp[i] / duration for i in keep]
        self._signals = []
        for i in keep:
            dig = data[:, starts[i]:starts[i + 1]].reshape(-1).astype(np.float64)
            gain = (pmax[i] - pmin[i]) / (dmax[i] - dmin[i])
            self._signals.append((dig - dmin[i]) * gain + pmin[i])
        self.signals_in_file = len(keep)
        self.truncated_records = max(0, int(raw[236:244].decode("ascii").strip()) - n_records)
        self._start = (raw[168:176].decode("ascii"), raw[176:184].decode("ascii"))

    def getStartdatetime(self):
        from datetime import datetime

        d, t = self._start
        day, month, year = (int(x) for x in d.split("."))
        hh, mm, ss = (int(x) for x in t.split("."))
        return datetime(2000 + year if year < 85 else 1900 + year, month, day, hh, mm, ss)

    def getSampleFrequencies(self):
        return list(self._fs)

    def getSignalLabels(self) -> list[str]:
        return list(self._labels)

    def getSampleFrequency(self, i: int) -> float:
        return self._fs[i]

    def getPhysicalDimension(self, i: int) -> str:
        return self._dims[i]

    def readSignal(self, i: int) -> np.ndarray:
        return self._signals[i]

    def close(self) -> None:
        self._signals = []


def _from_bipolar(
    data: np.ndarray, labels: list[str]
) -> tuple[np.ndarray, list[str], list[str]]:
    """Bipolar-only archive -> average reference + passthrough of non-EEG channels."""
    rec = reconstruct_average_reference(data, labels)
    rows = [rec.data]
    names = list(rec.electrodes)
    for i, lb in enumerate(labels):
        if parse_bipolar_label(lb):
            continue
        name = normalize_channel_name(lb)
        # keep auxiliary physiologic channels (ECG...) — dummy/unknown ones are dropped
        if name == "ECG" and not is_eeg_channel(name) and name not in names:
            rows.append(data[i : i + 1].astype(np.float32))
            names.append(name)
    flags = [f"reconstructed_from_bipolar:{len(rec.used_derivations)}_derivations"]
    if rec.dropped_electrodes:
        flags.append(f"bipolar_unreferenced_dropped:{','.join(rec.dropped_electrodes)}")
    if rec.residual_rms_uv > 1.0:
        flags.append(f"bipolar_inconsistency_rms_uv:{rec.residual_rms_uv:.1f}")
    return np.vstack(rows).astype(np.float32), names, flags


def _resample_linear(sig: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if fs_in == fs_out:
        return sig
    n_out = int(round(len(sig) * fs_out / fs_in))
    x_old = np.linspace(0.0, 1.0, num=len(sig), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, sig)


def _guess_reference(labels: list[str]) -> str:
    joined = " ".join(labels).lower()
    if "-ref" in joined or "ref" in joined:
        return "unknown_ref"
    if "-le" in joined or "linked" in joined:
        return "linked_ears"
    if "-avg" in joined or "average" in joined:
        return "average"
    return "unknown"
