"""Periodic & rhythmic ACNS pattern detector tests (in-memory, clean patterns)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import CH21, FRONTAL_CH, LEFT_CH, build_signal, sharp_train

from neurolens.layer2_preprocess.filters import apply_filters
from neurolens.layer3_features.feature_set import compute_features
from neurolens.layer4_detect.periodic import PeriodicPatternDetector
from neurolens.layer4_detect.special import ExtremeDeltaBrushDetector, FirdaDetector

FS = 256.0
SECONDS = 40.0


def _base(rng, n):
    return 6.0 * rng.standard_normal(n)


def _prep(sig, configs):
    """Filter (as the pipeline does) and compute features."""
    filt = apply_filters(sig, configs.filters)
    return filt, compute_features(filt, configs.filters)


def _run_periodic(sig, configs):
    filt, feats = _prep(sig, configs)
    return PeriodicPatternDetector().detect(filt, feats, configs.thresholds)


def test_lpds_lateralized_left(configs):
    rng = np.random.default_rng(0)
    n = int(FS * SECONDS)
    t = np.arange(n) / FS
    data = {}
    for ch in CH21:
        x = _base(rng, n)
        if ch in LEFT_CH:
            x = x + sharp_train(t, FS, 2.0, 70.0, t0=2.0, t1=30.0)
        data[ch] = x
    events = _run_periodic(build_signal(data, FS), configs)
    lpds = [e for e in events if e.code == "lpds"]
    assert lpds, f"expected LPDs, got {[e.code for e in events]}"
    assert lpds[0].localization.lateralization == "left"
    assert lpds[0].acns and abs(lpds[0].acns.frequency_hz - 2.0) < 0.3
    assert lpds[0].acns.iic is True  # 2 Hz is in the IIC band


def test_gpds_generalized(configs):
    rng = np.random.default_rng(1)
    n = int(FS * SECONDS)
    t = np.arange(n) / FS
    train = sharp_train(t, FS, 1.5, 70.0, t0=2.0, t1=30.0)
    data = {ch: _base(rng, n) + train for ch in CH21}  # synchronous on all channels
    events = _run_periodic(build_signal(data, FS), configs)
    gpds = [e for e in events if e.code == "gpds"]
    assert gpds, f"expected GPDs, got {[e.code for e in events]}"
    assert gpds[0].localization.lateralization == "generalized"


def test_grda_generalized(configs):
    rng = np.random.default_rng(2)
    n = int(FS * SECONDS)
    t = np.arange(n) / FS
    data = {}
    for i, ch in enumerate(CH21):
        data[ch] = _base(rng, n) + 45.0 * np.sin(2 * np.pi * 2.5 * t + 0.4 * i)
    events = _run_periodic(build_signal(data, FS), configs)
    grda = [e for e in events if e.code == "grda"]
    assert grda, f"expected GRDA, got {[e.code for e in events]}"
    assert grda[0].localization.lateralization == "generalized"


def test_lrda_lateralized(configs):
    rng = np.random.default_rng(3)
    n = int(FS * SECONDS)
    t = np.arange(n) / FS
    data = {}
    for i, ch in enumerate(CH21):
        x = _base(rng, n)
        if ch in LEFT_CH:
            x = x + 45.0 * np.sin(2 * np.pi * 2.0 * t + 0.2 * i)
        data[ch] = x
    events = _run_periodic(build_signal(data, FS), configs)
    lrda = [e for e in events if e.code == "lrda"]
    assert lrda, f"expected LRDA, got {[e.code for e in events]}"
    assert lrda[0].localization.lateralization == "left"


def test_firda_intermittent_frontal(configs):
    rng = np.random.default_rng(4)
    n = int(FS * SECONDS)
    t = np.arange(n) / FS
    burst = (((t % 8.0) < 4.0)).astype(float)  # intermittent 4s on / 4s off
    data = {}
    for i, ch in enumerate(CH21):
        x = _base(rng, n)
        if ch in FRONTAL_CH:
            x = x + 50.0 * np.sin(2 * np.pi * 2.0 * t + 0.3 * i) * burst
        data[ch] = x
    filt, feats = _prep(build_signal(data, FS), configs)
    events = FirdaDetector().detect(filt, feats, configs.thresholds)
    assert any(e.code == "firda" for e in events), f"expected FIRDA, got {[e.code for e in events]}"


def test_extreme_delta_brush(configs):
    rng = np.random.default_rng(5)
    n = int(FS * SECONDS)
    t = np.arange(n) / FS
    delta = np.sin(2 * np.pi * 2.0 * t)
    brush = np.clip(delta, 0, None) * np.sin(2 * np.pi * 24.0 * t)
    data = {}
    for i, ch in enumerate(CH21):
        d = np.sin(2 * np.pi * 2.0 * t + 0.2 * i)
        b = np.clip(d, 0, None) * np.sin(2 * np.pi * 24.0 * t)
        data[ch] = 6.0 * rng.standard_normal(n) + 45.0 * d + 35.0 * b
    filt, feats = _prep(build_signal(data, FS), configs)
    events = ExtremeDeltaBrushDetector().detect(filt, feats, configs.thresholds)
    assert any(e.code == "extreme_delta_brush" for e in events), \
        f"expected extreme_delta_brush, got {[e.code for e in events]}"
