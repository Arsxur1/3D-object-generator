"""Bipolar-input recordings -> reconstructed average reference (TZ §4).

Some archives (e.g. CHB-MIT) store only bipolar derivations ("FP1-F7",
"F7-T7", ...). The rest of NeuroLens works on referential data so that any
montage can be re-derived. Bipolar channels are exact potential differences
V_a - V_b, so on every *connected* set of electrodes the potentials are
determined up to one additive constant; fixing the component mean to zero
yields exactly the average reference over that component's electrodes
(least-squares solve; redundant/duplicate derivations are averaged).

Electrodes in components smaller than ``min_component`` (e.g. an isolated
Fz-Cz-Pz chain whose offset relative to the rest is unknowable) are dropped
and reported, never silently mixed with the main reference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .electrodes import is_eeg_channel, normalize_channel_name

# "FP1-F7", "EEG T8-P8", "T8-P8-0" (duplicate index suffix), "FZ-CZ"
_BIPOLAR = re.compile(
    r"^(?:EEG\s*)?([A-Za-z]{1,3}\d{0,2})\s*-\s*([A-Za-z]{1,3}\d{0,2})(?:-\d+)?$", re.I
)


def parse_bipolar_label(label: str) -> tuple[str, str] | None:
    """Return canonical (anode, cathode) for an EEG bipolar label, else None."""
    m = _BIPOLAR.match(label.strip())
    if not m:
        return None
    a, b = normalize_channel_name(m.group(1)), normalize_channel_name(m.group(2))
    if a == b or not (is_eeg_channel(a) and is_eeg_channel(b)):
        return None
    # "Fp1-Ref", "C3-A1" style referential labels are not bipolar pairs between
    # scalp electrodes: treat ear/mastoid references as referential.
    if b in ("A1", "A2"):
        return None
    return a, b


def is_bipolar_recording(labels: list[str], min_fraction: float = 0.6) -> bool:
    eeg_like = [lb for lb in labels if not _is_dummy(lb)]
    pairs = [lb for lb in eeg_like if parse_bipolar_label(lb)]
    return len(pairs) >= 4 and len(pairs) >= min_fraction * max(1, len(eeg_like))


def _is_dummy(label: str) -> bool:
    return label.strip().strip("-0123456789 ") == ""


@dataclass
class BipolarReconstruction:
    data: np.ndarray  # (n_electrodes, n_samples), average reference per component
    electrodes: list[str]
    dropped_electrodes: list[str] = field(default_factory=list)
    used_derivations: list[str] = field(default_factory=list)
    residual_rms_uv: float = 0.0


def reconstruct_average_reference(
    data: np.ndarray, labels: list[str], min_component: int = 6
) -> BipolarReconstruction:
    """Solve V (electrodes x samples) from bipolar rows D = A @ V, mean(V)=0."""
    pairs: list[tuple[int, str, str]] = []
    for i, lb in enumerate(labels):
        p = parse_bipolar_label(lb)
        if p:
            pairs.append((i, *p))
    if not pairs:
        raise ValueError("no bipolar EEG derivations found")

    electrodes = sorted({e for _, a, b in pairs for e in (a, b)})
    idx = {e: k for k, e in enumerate(electrodes)}

    # connected components (union-find)
    parent = list(range(len(electrodes)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, a, b in pairs:
        parent[find(idx[a])] = find(idx[b])
    comps: dict[int, list[int]] = {}
    for k in range(len(electrodes)):
        comps.setdefault(find(k), []).append(k)

    keep_comps = [c for c in comps.values() if len(c) >= min_component]
    if not keep_comps:  # fall back to the largest component
        keep_comps = [max(comps.values(), key=len)]
    keep = sorted(k for c in keep_comps for k in c)
    dropped = [electrodes[k] for k in range(len(electrodes)) if k not in set(keep)]

    sub = {k: j for j, k in enumerate(keep)}
    rows = [(i, a, b) for i, a, b in pairs if idx[a] in sub and idx[b] in sub]
    n_e = len(keep)
    A = np.zeros((len(rows) + len(keep_comps), n_e))
    for r, (_, a, b) in enumerate(rows):
        A[r, sub[idx[a]]] = 1.0
        A[r, sub[idx[b]]] = -1.0
    for c_i, comp in enumerate(keep_comps):  # mean-zero constraint per component
        for k in comp:
            A[len(rows) + c_i, sub[k]] = 1.0
    D = np.vstack([
        data[[i for i, _, _ in rows]].astype(np.float64),
        np.zeros((len(keep_comps), data.shape[1])),
    ])
    pinv = np.linalg.pinv(A)
    V = pinv @ D
    resid = A[: len(rows)] @ V - D[: len(rows)]
    return BipolarReconstruction(
        data=V.astype(np.float32),
        electrodes=[electrodes[k] for k in keep],
        dropped_electrodes=dropped,
        used_derivations=[labels[i] for i, _, _ in rows],
        residual_rms_uv=float(np.sqrt(np.mean(resid ** 2))) if resid.size else 0.0,
    )
