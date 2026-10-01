"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "data" / "synthetic"))

from make_synthetic_edf import make_synthetic_edf  # noqa: E402

from neurolens.contracts.signal import ClinicalContext, PatientInfo  # noqa: E402
from neurolens.pipeline.config_loader import load_configs  # noqa: E402


@pytest.fixture(scope="session")
def demo_edf(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("edf")
    return make_synthetic_edf(d / "demo.edf")


@pytest.fixture(scope="session")
def configs():
    return load_configs()


@pytest.fixture()
def demo_context() -> ClinicalContext:
    return ClinicalContext(sedatives=["propofol"], temperature_c=36.5,
                           clinical_question="ICU coma")


@pytest.fixture()
def adult_patient() -> PatientInfo:
    return PatientInfo(age_years=55)


def make_sine_signal(freq: float, fs: float = 256.0, seconds: float = 8.0,
                     amp: float = 30.0, channels=None):
    """Build a UnifiedSignal of pure sinusoids for feature tests."""
    from neurolens.contracts.signal import UnifiedSignal

    channels = channels or ["O1", "O2", "C3", "C4"]
    n = int(fs * seconds)
    t = np.arange(n) / fs
    data = np.stack([amp * np.sin(2 * np.pi * freq * t) for _ in channels]).astype(np.float32)
    return UnifiedSignal(
        signal=data, sampling_rate_hz=fs, channel_names=list(channels),
        electrode_system="10-20", reference="average",
    )


CH21 = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
        "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]
LEFT_CH = {"Fp1", "F3", "C3", "P3", "O1", "F7", "T3", "T5"}
FRONTAL_CH = {"Fp1", "Fp2", "F3", "F4", "Fz", "F7", "F8"}


def build_signal(channel_data: dict, fs: float = 256.0):
    """Build a UnifiedSignal directly from {channel: array} (no re-referencing)."""
    from neurolens.contracts.signal import UnifiedSignal

    names = list(channel_data)
    data = np.stack([np.asarray(channel_data[c], dtype=np.float32) for c in names])
    return UnifiedSignal(
        signal=data, sampling_rate_hz=fs, channel_names=names,
        electrode_system="10-20", reference="linked_ears",
    )


def sharp_train(t: np.ndarray, fs: float, freq: float, amp: float, t0=0.0, t1=None) -> np.ndarray:
    """Periodic SHARP biphasic discharges at ``freq`` Hz (for periodic-discharge tests)."""
    t1 = t[-1] if t1 is None else t1
    out = np.zeros_like(t)
    w = int(0.06 * fs)
    tt = np.linspace(-1, 1, w)
    pulse = np.exp(-(tt ** 2) / 0.05) * np.sign(tt)
    pulse = pulse / (np.max(np.abs(pulse)) + 1e-9)
    step = 1.0 / freq
    time = t0
    while time < t1:
        s = int(time * fs)
        if s + w <= len(out):
            out[s:s + w] += amp * pulse
        time += step
    return out
