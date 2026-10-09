"""CAP Sleep Database (PhysioNet capslpdb 1.0.0): clinical PSG with R&K hypnograms.

108 recordings: 16 healthy controls (n) and patients with nocturnal frontal lobe
epilepsy (nfle), REM behaviour disorder (rbd), periodic leg movements (plm),
insomnia (ins), narcolepsy (narco), sleep-disordered breathing (sdb) and
bruxism (brux). Montages are clinical (bipolar parasagittal chains, mastoid
references), recorded at 128-512 Hz; scoring in ``<record>.txt`` (RemLogic
export: clock time + "SLEEP-S0..S4 / SLEEP-REM / SLEEP-MT / SLEEP-UNSCORED").

Used in increment 13b as a zero-shot transfer test of the frozen Sleep-EDF
stager: the Sleep-EDF derivations are approximated from the right parasagittal
chain (Fpz-Cz ~ Fp2-C4 = (Fp2-F4) + (F4-C4); Pz-Oz ~ P4-O2), left chain if the
right one is absent; horizontal EOG ~ ROC-LOC.

Citation: Terzano MG et al. Atlas, rules, and recording techniques for the scoring
of cyclic alternating pattern (CAP) in human sleep. Sleep Med 2(6):537-553, 2001;
Goldberger AL et al., PhysioNet, Circulation 2000. License: ODC-By 1.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .sleepedf import EPOCH_S, STAGES, analysis_window

_RK = {"SLEEP-S0": "W", "SLEEP-S1": "N1", "SLEEP-S2": "N2", "SLEEP-S3": "N3", "SLEEP-S4": "N3",
       "SLEEP-REM": "REM"}
GROUPS = ("n", "brux", "ins", "narco", "nfle", "plm", "rbd", "sdb")
PER_GROUP = 4
RIGHT = ("FP2-F4", "F4-C4", "P4-O2")
LEFT = ("FP1-F3", "F3-C3", "P3-O1")
EOG = "ROC-LOC"


def usable(labels: list[str]) -> bool:
    up = {lb.strip().upper() for lb in labels}
    return (set(RIGHT) <= up or set(LEFT) <= up) and EOG in up


def select_records(headers: dict[str, list[str]], per_group: int = PER_GROUP) -> list[str]:
    """Pre-registered rule: usable records (chain + ROC-LOC), the first ``per_group``
    of each diagnostic group in numeric order."""
    by_group: dict[str, list[tuple[int, str]]] = {}
    for rec, labels in headers.items():
        m = re.match(r"([a-z]+)(\d+)\.edf$", rec)
        if m and usable(labels):
            by_group.setdefault(m.group(1), []).append((int(m.group(2)), rec))
    out = []
    for g in GROUPS:
        out += [r for _, r in sorted(by_group.get(g, []))[:per_group]]
    return out


def _clock(s: str) -> int:
    h, m, sec = (int(float(x)) for x in re.split(r"[:.]", s.strip())[:3])
    return h * 3600 + m * 60 + sec


def parse_scoring(text: str, edf_start: str, n_epochs: int) -> list:
    """Per-epoch stage indices (None = unscored/movement) aligned to the EDF start."""
    lines = text.splitlines()
    head = next(i for i, l in enumerate(lines) if "Time [hh:mm:ss]" in l and "Event" in l)
    cols = [c.strip() for c in lines[head].split("\t")]
    ti, ei = cols.index("Time [hh:mm:ss]"), cols.index("Event")
    di = cols.index("Duration[s]") if "Duration[s]" in cols else None
    t0 = _clock(edf_start)
    out: list = [None] * n_epochs
    for l in lines[head + 1:]:
        parts = [c.strip() for c in l.split("\t")]
        if len(parts) <= max(ti, ei) or parts[ei] not in _RK and not parts[ei].startswith("SLEEP-"):
            continue
        ev = parts[ei]
        if not ev.startswith("SLEEP-"):
            continue
        on = (_clock(parts[ti]) - t0) % 86400
        dur = float(parts[di]) if di is not None and di < len(parts) and parts[di] else EPOCH_S
        lab = STAGES.index(_RK[ev]) if ev in _RK else None
        a, b = int(round(on / EPOCH_S)), int(round((on + dur) / EPOCH_S))
        for i in range(max(a, 0), min(b, n_epochs)):
            out[i] = lab
    return out


@dataclass
class CapRecord:
    name: str
    group: str
    stages: list = field(default_factory=list)
    start_epoch: int = 0
    end_epoch: int = 0


def read_cap(edf_path: str | Path, txt: str, target_fs: float = 100.0) -> tuple[dict, CapRecord]:
    """Approximate Sleep-EDF derivations at ``target_fs`` + expert stages."""
    from fractions import Fraction

    import pyedflib
    from scipy.signal import resample_poly

    r = pyedflib.EdfReader(str(edf_path))
    try:
        labels = [lb.strip().upper() for lb in r.getSignalLabels()]
        chain = RIGHT if set(RIGHT) <= set(labels) else LEFT

        def get(name):
            i = labels.index(name)
            x = r.readSignal(i)
            fr = Fraction(target_fs / r.getSampleFrequency(i)).limit_denominator(1000)
            return resample_poly(x, fr.numerator, fr.denominator).astype(np.float32)

        a, b, c = (get(n) for n in chain)
        eog = get(EOG)
        start = r.getStartdatetime().strftime("%H:%M:%S")
    finally:
        r.close()
    n = min(len(a), len(b), len(c), len(eog))
    sig = {"frontal": a[:n] + b[:n], "parietal": c[:n], "eog": eog[:n], "chain": "right" if chain == RIGHT else "left"}
    name = Path(edf_path).stem
    rec = CapRecord(name, re.match(r"[a-z]+", name).group(0))
    rec.stages = parse_scoring(txt, start, int(n / target_fs // EPOCH_S))
    rec.start_epoch, rec.end_epoch = analysis_window(rec.stages)
    return sig, rec
