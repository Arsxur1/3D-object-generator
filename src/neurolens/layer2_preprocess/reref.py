"""Re-referencing (TZ §5): average, linked-ears, or a single channel.

REST is reserved for the optional MNE backend (v2). Only EEG channels are
used to build the average; non-EEG channels (ECG/EOG) are left untouched.
"""

from __future__ import annotations

import numpy as np

from ..contracts.signal import UnifiedSignal
from ..layer1_ingest.electrodes import is_eeg_channel


def rereference(sig: UnifiedSignal, scheme: str = "average") -> UnifiedSignal:
    data = sig.signal.astype(np.float64).copy()
    names = sig.channel_names
    eeg_idx = [i for i, n in enumerate(names) if is_eeg_channel(n)]

    if scheme == "average":
        if eeg_idx:
            ref = data[eeg_idx].mean(axis=0, keepdims=True)
            data[eeg_idx] -= ref
        new_ref = "average"
    elif scheme in ("linked_ears", "linked_mastoids"):
        a_idx = [i for i, n in enumerate(names) if n.upper() in ("A1", "A2", "M1", "M2")]
        if a_idx:
            ref = data[a_idx].mean(axis=0, keepdims=True)
            data[eeg_idx] -= ref
        new_ref = "linked_ears"
    else:  # treat scheme as a channel name
        try:
            ch = sig.channel_index(scheme)
        except KeyError as exc:
            raise ValueError(f"unknown reference scheme/channel {scheme!r}") from exc
        data[eeg_idx] -= data[ch : ch + 1]
        new_ref = scheme

    return sig.with_signal(data.astype(np.float32), reference=new_ref)
