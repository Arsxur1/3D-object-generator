"""Layer 4 detector tests (run against the synthetic recording)."""

from __future__ import annotations

import pytest

from neurolens.layer1_ingest.registry import ingest
from neurolens.layer2_preprocess.artifacts import flag_artifacts
from neurolens.layer2_preprocess.filters import apply_filters
from neurolens.layer2_preprocess.reref import rereference
from neurolens.layer3_features.feature_set import compute_features
from neurolens.layer4_detect.registry import run_detectors


@pytest.fixture(scope="module")
def analyzed(request):
    demo_edf = request.getfixturevalue("demo_edf")
    configs = request.getfixturevalue("configs")
    sig = ingest(demo_edf)
    filt = apply_filters(sig, configs.filters)
    art = flag_artifacts(filt, mains_hz=configs.filters.notch_hz)
    ana = rereference(filt, "average")
    feats = compute_features(ana, configs.filters)
    det = run_detectors(ana, feats, configs.thresholds, art)
    return det, art


def test_detects_diffuse_slowing(analyzed):
    det, _ = analyzed
    assert "diffuse_slowing" in det.codes()


def test_detects_ictal(analyzed):
    det, _ = analyzed
    ictal = det.by_group("ictal")
    assert ictal, "expected an ictal event"
    assert ictal[0].duration_s > 20  # sustained


def test_detects_burst_suppression(analyzed):
    det, _ = analyzed
    assert "burst_suppression" in det.codes()


def test_detects_ecg_artifact(analyzed):
    det, art = analyzed
    assert art.ecg_present
    assert "ecg_artifact" in det.codes()


def test_ecg_artifact_marked_as_hypothesis(analyzed):
    det, _ = analyzed
    ecg = [e for e in det.events if e.code == "ecg_artifact"][0]
    assert ecg.is_artifact_hypothesis is True
