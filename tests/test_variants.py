"""Benign-variant detection & resolution tests (TZ §8.2)."""

from __future__ import annotations

import numpy as np

from conftest import CH21, build_signal

from neurolens.contracts.events import DetectionResult, Event, Localization
from neurolens.contracts.signal import PatientInfo
from neurolens.layer2_preprocess.filters import apply_filters
from neurolens.layer3_features.feature_set import compute_features
from neurolens.layer3_features.norms import NormsEngine
from neurolens.layer4_detect.variants import BenignVariantDetector
from neurolens.layer5_reasoning.physiology import PhysiologyEngine
from neurolens.layer5_reasoning.plausibility import check_plausibility

FS = 256.0
_TEMPORAL = {"T3", "T4", "F7", "F8", "T5", "T6"}


def test_wicket_detected_over_temporal(configs):
    rng = np.random.default_rng(7)
    n = int(FS * 30)
    t = np.arange(n) / FS
    data = {}
    for ch in CH21:
        x = 8.0 * rng.standard_normal(n)
        if ch in _TEMPORAL:
            x = x + 32.0 * (np.sin(2 * np.pi * 9.0 * t) + 0.3 * np.sin(2 * np.pi * 18.0 * t))
        data[ch] = x
    filt = apply_filters(build_signal(data, FS), configs.filters)
    feats = compute_features(filt, configs.filters)
    events = BenignVariantDetector().detect(filt, feats, configs.thresholds)
    assert any(e.code == "wicket" for e in events), f"expected wicket, got {[e.code for e in events]}"


def _spike(chan, t0):
    return Event(code="spike", label_ru="s", label_uz="s", group="ied",
                 localization=Localization(channels=[chan], region="temporal"),
                 t_start=t0, t_end=t0 + 0.1, confidence=0.55)


def _wicket(chan, t0, t1):
    return Event(code="wicket", label_ru="w", label_uz="w", group="variant",
                 localization=Localization(channels=[chan], region="temporal"),
                 t_start=t0, t_end=t1, confidence=0.6)


def test_plausibility_resolves_spike_as_variant():
    spike = _spike("T3", 10.0)
    wicket = _wicket("T3", 0.0, 30.0)
    det = DetectionResult(events=[spike, wicket])
    res = check_plausibility(det, artifacts=None)["spike_resolutions"]
    assert res.get(id(spike), {}).get("kind") == "variant"


def test_age_gated_posterior_slow_waves(configs):
    engine = PhysiologyEngine(NormsEngine(configs.norms))
    child = PatientInfo(age_years=10)
    ev = Event(code="focal_slowing", label_ru="x", label_uz="x", group="background",
               localization=Localization(region="occipital"), t_start=0, t_end=5, confidence=0.6)
    assert engine.is_age_physiologic_variant(ev, child) is True
    # an adult with the same finding is NOT a physiologic variant
    assert engine.is_age_physiologic_variant(ev, PatientInfo(age_years=45)) is False
