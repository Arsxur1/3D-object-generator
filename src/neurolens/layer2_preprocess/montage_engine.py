"""Montage engine (TZ §5, first-class).

Builds referential and bipolar derivations from the referential signal WITHOUT
recomputing detection: detectors run on the referential/average data; montages
are a *view* used for display and localization. Switching montages is therefore
just re-deriving channel pairs — cheap and non-destructive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts.config import MontageConfig, MontageType
from ..contracts.signal import UnifiedSignal
from ..layer1_ingest.electrodes import is_eeg_channel


@dataclass
class MontagedSignal:
    """A montage view: derived traces + their labels (not a UnifiedSignal)."""

    montage_name: str
    montage_type: MontageType
    derivation_names: list[str]
    data: np.ndarray  # [derivations][samples], microvolts
    sampling_rate_hz: float

    @property
    def n_derivations(self) -> int:
        return self.data.shape[0]

    def trace(self, name: str) -> np.ndarray:
        return self.data[self.derivation_names.index(name)]


class MontageEngine:
    """Applies a :class:`MontageConfig` to a :class:`UnifiedSignal`."""

    def __init__(self, signal: UnifiedSignal):
        self.signal = signal
        self._index = {n: i for i, n in enumerate(signal.channel_names)}

    def _channel(self, name: str) -> np.ndarray | None:
        idx = self._index.get(name)
        if idx is None:
            return None
        return self.signal.signal[idx].astype(np.float64)

    def apply(self, montage: MontageConfig) -> MontagedSignal:
        traces: list[np.ndarray] = []
        names: list[str] = []
        n_samples = self.signal.n_samples

        if montage.type == MontageType.REFERENTIAL:
            ref_vec = self._reference_vector(montage.reference)
            for pair in montage.pairs:
                a = self._channel(pair.anode)
                if a is None:
                    continue
                traces.append(a - ref_vec)
                names.append(pair.name)
        else:  # BIPOLAR
            for pair in montage.pairs:
                a = self._channel(pair.anode)
                c = self._channel(pair.cathode) if pair.cathode else None
                if a is None or c is None:
                    continue
                traces.append(a - c)
                names.append(pair.name)

        if not traces:
            raise ValueError(
                f"montage {montage.name!r} produced no derivations; "
                "channel names may not match the recording"
            )
        data = np.stack(traces, axis=0)[:, :n_samples].astype(np.float32)
        return MontagedSignal(
            montage_name=montage.name,
            montage_type=montage.type,
            derivation_names=names,
            data=data,
            sampling_rate_hz=self.signal.sampling_rate_hz,
        )

    def _reference_vector(self, reference: str | None) -> np.ndarray:
        n_samples = self.signal.n_samples
        if reference in (None, "average"):
            eeg_idx = [
                i for i, n in enumerate(self.signal.channel_names) if is_eeg_channel(n)
            ]
            if not eeg_idx:
                return np.zeros(n_samples)
            return self.signal.signal[eeg_idx].astype(np.float64).mean(axis=0)
        if reference in ("linked_ears", "linked_mastoids"):
            a_idx = [
                i for i, n in enumerate(self.signal.channel_names)
                if n.upper() in ("A1", "A2", "M1", "M2")
            ]
            if a_idx:
                return self.signal.signal[a_idx].astype(np.float64).mean(axis=0)
            return np.zeros(n_samples)
        ch = self._channel(reference)
        return ch if ch is not None else np.zeros(n_samples)
