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
