"""Layer 2 preprocessing tests."""

from __future__ import annotations

import numpy as np

from neurolens.contracts.config import FilterConfig, MontageConfig, MontageType, MontagePair
from neurolens.contracts.signal import UnifiedSignal
from neurolens.layer2_preprocess.filters import apply_filters
from neurolens.layer2_preprocess.montage_engine import MontageEngine
from neurolens.layer2_preprocess.reref import rereference


def _signal(data, names, fs=256.0):
    return UnifiedSignal(signal=np.asarray(data, dtype=np.float32),
                         sampling_rate_hz=fs, channel_names=list(names),
                         reference="unknown")


def test_notch_attenuates_line_noise():
    fs = 256.0
    n = int(fs * 6)
    t = np.arange(n) / fs
    # 10 Hz signal + strong 50 Hz mains
    clean = 30 * np.sin(2 * np.pi * 10 * t)
    noisy = clean + 40 * np.sin(2 * np.pi * 50 * t)
    sig = _signal([noisy, noisy], ["C3", "C4"])
    out = apply_filters(sig, FilterConfig(notch_hz=50.0, notch_enabled=True))

    def power_at(x, f0):
        freqs = np.fft.rfftfreq(len(x), 1 / fs)
        sp = np.abs(np.fft.rfft(x))
        return sp[np.argmin(np.abs(freqs - f0))]

    assert power_at(out.signal[0], 50.0) < 0.2 * power_at(noisy, 50.0)


def test_double_banana_bipolar_difference():
    fs = 256.0
    a = np.ones(100) * 5.0
    b = np.ones(100) * 2.0
    sig = _signal([a, b], ["Fp1", "F7"])
    montage = MontageConfig(
        name="mini", type=MontageType.BIPOLAR,
        pairs=[MontagePair(name="Fp1-F7", anode="Fp1", cathode="F7")],
    )
    m = MontageEngine(sig).apply(montage)
    assert m.derivation_names == ["Fp1-F7"]
    assert np.allclose(m.trace("Fp1-F7"), 3.0)


def test_average_reference_zero_sum():
    fs = 256.0
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 500)).astype(np.float32) * 20
    sig = _signal(data, ["C3", "C4", "P3", "P4"])
    out = rereference(sig, "average")
    # average across EEG channels should be ~0 at each sample
    assert np.allclose(out.signal.mean(axis=0), 0.0, atol=1e-3)
