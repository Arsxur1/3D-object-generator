"""Synthetic EDF generator determinism."""

from __future__ import annotations

import numpy as np
from make_synthetic_edf import CHANNELS, make_synthetic_edf

from neurolens.layer1_ingest.registry import ingest


def test_generation_is_deterministic(tmp_path):
    p1 = make_synthetic_edf(tmp_path / "a.edf", seed=42, duration_s=30)
    p2 = make_synthetic_edf(tmp_path / "b.edf", seed=42, duration_s=30)
    s1 = ingest(p1).signal
    s2 = ingest(p2).signal
    assert s1.shape == s2.shape
    assert np.allclose(s1, s2, atol=1e-3)


def test_channels_present(tmp_path):
    p = make_synthetic_edf(tmp_path / "c.edf", duration_s=10)
    sig = ingest(p)
    for ch in CHANNELS:
        assert ch in sig.channel_names
