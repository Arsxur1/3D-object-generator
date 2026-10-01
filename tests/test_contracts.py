"""Contract / schema tests."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from neurolens.contracts.causal import CausalGraph, CausalNode, NodeKind, PhysiologyLabel
from neurolens.contracts.events import Event, EventEvidence, Localization
from neurolens.contracts.export_schemas import SCHEMAS
from neurolens.contracts.signal import UnifiedSignal


def test_unified_signal_shape_validation():
    sig = UnifiedSignal(
        signal=np.zeros((3, 100), dtype=np.float32),
        sampling_rate_hz=256, channel_names=["C3", "C4", "Cz"],
    )
    assert sig.n_channels == 3
    assert sig.duration_s == pytest.approx(100 / 256)

    with pytest.raises(ValidationError):
        UnifiedSignal(signal=np.zeros((2, 10)), sampling_rate_hz=256,
                      channel_names=["only_one"])


def test_unified_signal_metadata_excludes_samples():
    sig = UnifiedSignal(signal=np.zeros((2, 10)), sampling_rate_hz=256,
                        channel_names=["C3", "C4"])
    md = sig.metadata_dict()
    assert "signal" not in md
    assert md["channel_names"] == ["C3", "C4"]


def test_event_roundtrip():
    ev = Event(
        code="diffuse_slowing", label_ru="x", label_uz="y", group="background",
        localization=Localization(region="generalized"),
        t_start=0.0, t_end=10.0, confidence=0.7,
        evidence=[EventEvidence(feature="rel_delta", value=0.6)],
    )
    dumped = ev.model_dump(mode="json")
    ev2 = Event(**dumped)
    assert ev2.code == "diffuse_slowing"
    assert ev2.duration_s == 10.0


def test_causal_graph_grounding_keys():
    g = CausalGraph(
        nodes=[CausalNode(id="evt:a:0", kind=NodeKind.EVENT, label_ru="a", label_uz="a",
                          confidence=0.8, event_code="diffuse_slowing",
                          evidence_refs=["rel_delta"])],
    )
    keys = g.grounding_keys()
    assert "evt:a:0" in keys and "diffuse_slowing" in keys and "rel_delta" in keys


def test_all_schemas_export():
    for name, model in SCHEMAS.items():
        schema = model.model_json_schema()
        assert schema.get("type") == "object" or "$defs" in schema
