"""Layer 1 ingestion tests."""

from __future__ import annotations

from neurolens.contracts.signal import ElectrodeSystem
from neurolens.layer1_ingest.electrodes import (
    detect_electrode_system,
    is_eeg_channel,
    normalize_channel_name,
)
from neurolens.layer1_ingest.registry import ingest


def test_normalize_channel_names():
    assert normalize_channel_name("EEG Fp1-Ref") == "Fp1"
    assert normalize_channel_name("FP1") == "Fp1"
    assert normalize_channel_name("T7") == "T3"  # 10-10 -> classic alias
    assert normalize_channel_name("P8") == "T6"
    assert normalize_channel_name("EKG") == "ECG"


def test_detect_system_10_20():
    ch = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
          "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]
    assert detect_electrode_system(ch) == ElectrodeSystem.TEN_TWENTY


def test_is_eeg_channel():
    assert is_eeg_channel("C3")
    assert is_eeg_channel("Fz")
    assert not is_eeg_channel("ECG")
    assert not is_eeg_channel("EOG")


def test_ingest_synthetic_edf(demo_edf):
    sig = ingest(demo_edf)
    assert sig.sampling_rate_hz == 256
    assert "Fp1" in sig.channel_names
    assert "ECG" in sig.channel_names
    assert sig.electrode_system == ElectrodeSystem.TEN_TWENTY
    assert sig.n_samples == 360 * 256
    assert sig.units == "uV"
