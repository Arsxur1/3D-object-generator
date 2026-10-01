"""Deterministic synthetic EEG generator (TZ §18.4, offline demo/test input).

Produces a 21-channel 10-20 EDF+ at 256 Hz with clearly-injected, labeled
segments so the whole pipeline yields non-trivial, reproducible output:

    0-60s    normal posterior alpha background
    60-120s  diffuse slowing (generalized delta)
    120-180s left-lateralized asymmetry (extra left slow activity)
    180-240s ictal rhythmic discharge (evolving freq, rising amplitude, left)
    240-300s frontal triphasic-like ~2 Hz waves
    300-360s burst-suppression (bursts separated by suppression)

An ECG channel with ~60 bpm QRS is added and coupled into the temporal channels
to create an ECG artifact (and pseudo-epileptiform transients) for the
artifact-vs-IED discrimination demo.

Not physiological truth — a controlled fixture. Real EDF files also work.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyedflib

FS = 256
DURATION_S = 360
SEED = 42

CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
]
LEFT = {"Fp1", "F3", "C3", "P3", "O1", "F7", "T3", "T5"}
POSTERIOR = {"O1", "O2", "P3", "P4"}
FRONTAL = {"Fp1", "Fp2", "F3", "F4", "Fz", "F7", "F8"}


def _pink(rng: np.random.Generator, n: int) -> np.ndarray:
    """True pink (1/f) noise via spectral shaping of white noise."""
    white = rng.standard_normal(n)
    X = np.fft.rfft(white)
    f = np.fft.rfftfreq(n)
    f[0] = f[1] if len(f) > 1 else 1.0
    X = X / np.sqrt(f)  # 1/f power spectrum
    x = np.fft.irfft(X, n=n)
    x = (x - x.mean()) / (x.std() + 1e-9)
    return x


def _seg_mask(t: np.ndarray, t0: float, t1: float) -> np.ndarray:
    return (t >= t0) & (t < t1)


def _qrs_template(fs: int) -> np.ndarray:
    """A short QRS-like biphasic spike (~100 ms)."""
    n = int(0.1 * fs)
    tt = np.linspace(-1, 1, n)
    q = np.exp(-((tt) ** 2) / 0.02) * np.sign(tt)  # sharp biphasic
    q = q / (np.max(np.abs(q)) + 1e-9)
    return q


def make_synthetic_edf(path: str | Path, seed: int = SEED, duration_s: int = DURATION_S) -> Path:
    rng = np.random.default_rng(seed)
    n = duration_s * FS
    t = np.arange(n) / FS

    data: dict[str, np.ndarray] = {}
    for ch in CHANNELS:
        ci = CHANNELS.index(ch)
        sig = 18.0 * _pink(rng, n)  # background (comfortably above the ~10uV floor)

        # posterior alpha (awake) in the first minute + occipital emphasis
        if ch in POSTERIOR:
            sig += 30.0 * np.sin(2 * np.pi * 10.0 * t) * _seg_mask(t, 0, 60)

        # diffuse slowing 60-120s (per-channel phase/amplitude so it is not pure
        # common-mode, i.e. it survives average re-referencing as real slowing does)
        sig += (48.0 + 8.0 * (ci % 4)) * np.sin(2 * np.pi * 2.0 * t + 0.6 * ci) * _seg_mask(t, 60, 120)

        # left-lateralized asymmetry 120-180s (extra slow left)
        if ch in LEFT:
            sig += 45.0 * np.sin(2 * np.pi * 2.5 * t + 0.3) * _seg_mask(t, 120, 180)

        # ictal rhythmic discharge 180-240s, evolving 8->4 Hz, amp 55->110 uV, left
        if ch in LEFT:
            m = _seg_mask(t, 180, 240)
            local = (t - 180) / 60.0
            freq = 8.0 - 4.0 * local
            amp = 55.0 + 55.0 * local
            phase = 2 * np.pi * np.cumsum(freq * m) / FS
            sig += (amp * np.sin(phase)) * m

        # frontal triphasic-like ~2 Hz 240-300s
        if ch in FRONTAL:
            m = _seg_mask(t, 240, 300)
            base = np.sin(2 * np.pi * 2.0 * t)
            triph = base - 0.4 * np.sin(2 * np.pi * 4.0 * t)  # add harmonic for shape
            sig += 60.0 * triph * m

        # burst-suppression 300-360s
        m_bs = _seg_mask(t, 300, 360)
        if m_bs.any():
            # 1s burst every 3s: burst when (t mod 3) < 1
            tb = (t - 300) % 3.0
            burst = (tb < 1.0).astype(float)
            bs_sig = burst * (70.0 * np.sin(2 * np.pi * 6.0 * t) + 25.0 * _pink(rng, n))
            # suppress background during this window
            sig = np.where(m_bs, bs_sig, sig)

        data[ch] = sig.astype(np.float64)

    # --- ECG channel + coupling into temporal channels ---
    ecg = np.zeros(n)
    q = _qrs_template(FS)
    rr = int(FS * 1.0)  # 60 bpm
    for start in range(0, n - len(q), rr):
        ecg[start:start + len(q)] += q
    ecg *= 300.0  # ECG amplitude on its own channel
    # couple a scaled, sharp QRS into temporal channels -> ECG artifact + pseudo-spikes
    coupling = np.zeros(n)
    for start in range(0, n - len(q), rr):
        coupling[start:start + len(q)] += q
    data["T3"] = data["T3"] + 45.0 * coupling
    data["T4"] = data["T4"] + 40.0 * coupling

    # assemble writer
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    all_labels = CHANNELS + ["ECG"]
    all_data = [data[ch] for ch in CHANNELS] + [ecg]

    writer = pyedflib.EdfWriter(str(path), len(all_labels), file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        headers = []
        for label in all_labels:
            pmax = 2000.0
            headers.append({
                "label": label,
                "dimension": "uV",
                "sample_frequency": FS,
                "physical_max": pmax,
                "physical_min": -pmax,
                "digital_max": 32767,
                "digital_min": -32768,
                "transducer": "",
                "prefilter": "",
            })
        writer.setSignalHeaders(headers)
        writer.writeSamples([np.clip(d, -1999, 1999) for d in all_data])
    finally:
        writer.close()
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a synthetic demo EDF.")
    default_out = Path(__file__).resolve().parent / "demo.edf"
    ap.add_argument("-o", "--out", default=str(default_out))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--duration", type=int, default=DURATION_S)
    args = ap.parse_args()
    out = make_synthetic_edf(args.out, seed=args.seed, duration_s=args.duration)
    print(f"wrote {out} ({args.duration}s, {FS}Hz, {len(CHANNELS)+1} channels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
