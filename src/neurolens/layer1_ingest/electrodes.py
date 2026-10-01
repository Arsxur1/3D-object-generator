"""Electrode/channel-name normalization and electrode-system detection (TZ §4).

Instrument exports vary wildly ("EEG Fp1-Ref", "FP1", "T7"...). We normalize to
the canonical 10-20 labels used throughout NeuroLens and detect which electrode
system a recording uses.
"""

from __future__ import annotations

import re

from ..contracts.signal import ElectrodeSystem

# Canonical 10-20 set used across NeuroLens (classic T3/T4/T5/T6 nomenclature).
CANONICAL_10_20 = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz", "A1", "A2",
]

# 10-10 -> classic aliases (so modern exports map onto our canonical set).
_ALIASES = {
    "T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6",
    "M1": "A1", "M2": "A2",
}

# Extra electrodes that only exist in extended (10-10/10-5) systems.
_EXTENDED_ONLY = {
    "AF3", "AF4", "FC1", "FC2", "FC5", "FC6", "CP1", "CP2", "CP5", "CP6",
    "PO3", "PO4", "F1", "F2", "C1", "C2", "P1", "P2", "FT7", "FT8", "TP7", "TP8",
}


def normalize_channel_name(raw: str) -> str:
    """Map a raw channel label to a canonical electrode name.

    Strips common prefixes/suffixes ("EEG ", "-Ref", "-LE"), fixes case, and
    applies 10-10 -> classic aliases. Non-EEG channels (ECG/EKG/EOG/EMG) are
    passed through in a canonical upper-case form.
    """
    s = raw.strip()
    s = re.sub(r"(?i)^eeg[\s:]*", "", s)  # drop leading "EEG "
    # Drop a trailing reference tag: "-Ref", "-REF", "-LE", "-A1", "-AVG"...
    s = re.sub(r"(?i)[-_ ](ref|le|avg|average|a1|a2|m1|m2|cz|linked.*)$", "", s)
    s = s.strip(" -_")

    up = s.upper()
    # Non-EEG modality channels
    for mod in ("ECG", "EKG", "EOG", "EMG", "RESP", "SPO2", "PHOTIC", "TRIG"):
        if up.startswith(mod):
            return "ECG" if up.startswith("EKG") else up

    # Canonical case: first letter(s) capital, keep digits, e.g. "FP1"->"Fp1"
    m = re.match(r"^([A-Za-z]+)(\d*)$", s)
    if m:
        letters, digits = m.group(1), m.group(2)
        if len(letters) >= 2:
            canon = letters[0].upper() + letters[1:].lower()
        else:
            canon = letters.upper()
        name = f"{canon}{digits}"
    else:
        name = s

    return _ALIASES.get(name.upper(), name)


def detect_electrode_system(channel_names: list[str]) -> ElectrodeSystem:
    """Infer the electrode system from the set of EEG channel names."""
    names = {n for n in channel_names}
    upper = {n.upper() for n in names}

    if upper & {n.upper() for n in _EXTENDED_ONLY}:
        return ElectrodeSystem.TEN_TEN

    core = {"FP1", "FP2", "C3", "C4", "O1", "O2"}
    overlap = len(upper & core)
    eeg_like = sum(1 for n in upper if re.match(r"^(FP|F|C|P|O|T|A)\d|^[FCPO]Z$", n))

    if overlap >= 4:
        # Reduced neonatal montages have few electrodes.
        if eeg_like <= 12:
            return ElectrodeSystem.NEONATAL_REDUCED if eeg_like <= 9 else ElectrodeSystem.TEN_TWENTY
        return ElectrodeSystem.TEN_TWENTY
    return ElectrodeSystem.UNKNOWN


def is_eeg_channel(name: str) -> bool:
    """True for scalp-EEG electrodes (excludes ECG/EOG/EMG/etc.)."""
    up = name.upper()
    if any(up.startswith(m) for m in ("ECG", "EKG", "EOG", "EMG", "RESP", "SPO2", "PHOTIC", "TRIG")):
        return False
    return bool(re.match(r"^(FP|AF|F|FC|FT|C|CP|T|TP|P|PO|O|A|M)\d|^[FCPO]Z$", up))
