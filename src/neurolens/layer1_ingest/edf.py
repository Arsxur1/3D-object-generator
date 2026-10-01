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
from .electrodes import detect_electrode_system, normalize_channel_name

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
        reader = pyedflib.EdfReader(str(path))
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

            electrode_system = detect_electrode_system(channel_names)
            reference = _guess_reference(labels)
        finally:
            reader.close()

        return UnifiedSignal(
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
