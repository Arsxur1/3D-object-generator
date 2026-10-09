"""Helsinki neonatal dataset: annotation consensus, rule-based split, Zenodo client,
and neonatal routing of the seizure detector (offline + monitor)."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from neurolens.contracts.config import LearnedDetectorMode, NeonatalPolicy
from neurolens.datasets.helsinki import (
    HelsinkiClient,
    consensus_intervals,
    parse_expert_csv,
    parse_helsinki,
    split_by_rule,
)


def _csv(cols: dict[int, list[int]]) -> str:
    n = max(len(v) for v in cols.values())
    lines = [",".join(str(k) for k in cols)]
    for i in range(n):
        lines.append(",".join(str(v[i]) if i < len(v) else "" for v in cols.values()))
    return "\n".join(lines) + "\n"


def test_parse_expert_csv_strips_padding():
    got = parse_expert_csv(_csv({1: [0, 1, 1], 2: [0, 0, 0, 0, 1]}))
    assert got == {1: [0, 1, 1], 2: [0, 0, 0, 0, 1]}


def test_consensus_requires_all_experts_and_min_duration():
    a = [0] * 5 + [1] * 15 + [0] * 5 + [1] * 8 + [0] * 5
    b = [0] * 7 + [1] * 13 + [0] * 5 + [1] * 8 + [0] * 5
    c = [0] * 5 + [1] * 15 + [0] * 18
    iv = consensus_intervals([a, b, c])
    assert [(s.onset_s, s.offset_s) for s in iv] == [(7.0, 20.0)]  # 8-s run: not all experts


def test_parse_helsinki_ambiguous_excludes_consensus():
    a = {1: [0] * 5 + [1] * 12 + [0] * 10 + [1] * 12 + [0] * 3}
    b = {1: [0] * 5 + [1] * 12 + [0] * 25 + [0] * 2}
    c = {1: [0] * 5 + [1] * 12 + [0] * 27}
    (rec,) = parse_helsinki({k: _csv(v) for k, v in (("A", a), ("B", b), ("C", c))})
    assert rec.file == "eeg1.edf" and [(s.onset_s, s.offset_s) for s in rec.seizures] == [(5.0, 17.0)]
    assert [(s.onset_s, s.offset_s) for s in rec.ambiguous] == [(27.0, 39.0)]  # expert A only


def test_split_alternates_within_groups():
    reviewers = {f"eeg{i}": k for i, k in enumerate([3, 3, 3, 0, 0, 1, 2, 3], start=1)}
    s = split_by_rule(reviewers)
    assert s["dev"] == ["eeg1", "eeg3", "eeg4", "eeg6"]
    assert s["test"] == ["eeg2", "eeg5", "eeg7", "eeg8"]


def test_client_verifies_md5(tmp_path):
    payload = b"EDF-bytes"
    meta = {"files": [{"key": "eeg1.edf", "size": len(payload),
                       "checksum": "md5:" + hashlib.md5(payload).hexdigest()},
                      {"key": "eeg2.edf", "size": 3, "checksum": "md5:" + "0" * 32}]}

    def fetch_file(url, dest):
        dest.write_bytes(payload)

    c = HelsinkiClient(tmp_path, fetch_text=lambda url: json.dumps(meta).encode(), fetch_file=fetch_file)
    assert c.subjects() == ["eeg1", "eeg2"]
    assert c.fetch("eeg1.edf").read_bytes() == payload
    with pytest.raises(Exception, match="md5"):
        c.fetch("eeg2.edf")
    assert not c.is_cached("eeg2.edf")


def test_select_routes_neonates(configs):
    ml = copy.deepcopy(configs.ml)
    assert ml.select("offline") is ml.offline  # adults: unchanged
    ml.neonatal = NeonatalPolicy(offline="threshold", realtime="general")
    assert ml.select("offline", neonate=True) is None
    assert ml.select("realtime", neonate=True) is ml.realtime
    neo = LearnedDetectorMode(enabled=True, model=ml.offline.model, threshold=0.5, min_epochs=7)
    ml.neonatal = NeonatalPolicy(offline="neonatal", offline_model=neo)
    assert ml.select("offline", neonate=True).min_epochs == 7
    assert ml.select("offline", neonate=False) is ml.offline


def test_pipeline_and_monitor_neonatal_routing(demo_edf, configs):
    from neurolens.contracts.signal import PatientInfo
    from neurolens.pipeline.pipeline import Pipeline
    from neurolens.realtime.monitor import RealtimeMonitor

    cfg = copy.deepcopy(configs)
    cfg.ml.neonatal = NeonatalPolicy(offline="threshold", realtime="threshold")
    neonate = PatientInfo(age_years=0.0, postmenstrual_age_weeks=40)
    pipe = Pipeline(config=cfg, provider_pref="deterministic", run_ica=False)
    out = pipe.analyze_file(demo_edf, patient=neonate)
    assert "ictal_rhythm" in out.detection.detectors_run and "ictal_ml" not in out.detection.detectors_run
    adult = pipe.analyze_file(demo_edf, patient=PatientInfo(age_years=40))
    assert "ictal_ml" in adult.detection.detectors_run
    assert RealtimeMonitor(cfg, patient=neonate)._ml is None
    assert RealtimeMonitor(cfg, patient=PatientInfo(age_years=40))._ml is not None
    assert RealtimeMonitor(cfg, patient=neonate, learned=True)._ml is not None  # explicit wins


def test_prereg12_decision_rules_and_apply(tmp_path):
    import shutil

    import yaml

    from neurolens.evaluation.prereg12 import apply_decision, assess, markdown_report

    def rec(tp, fp, n=10, fps=()):
        return {"file": "eeg4.edf", "hours": 10.0, "n_seizures": n, "tp": tp, "fp": fp,
                "latencies_s": [5.0] * tp, "false_positives": [list(x) for x in fps]}

    def metrics(neo, gen, thr=(8, 30)):
        res = []
        for p in ("offline", "realtime"):
            res += [{"mode": f"{p}-threshold-default", "records": [rec(*thr, fps=[(100, 110)] * thr[1])]},
                    {"mode": f"{p}-ml-A", "records": [rec(*neo)]},
                    {"mode": f"{p}-cmp-A", "records": [rec(*gen)]}]
        return {"results": res}

    a = assess(metrics(neo=(9, 20), gen=(5, 10)), ambiguous={"eeg4.edf": [(90.0, 120.0)]})
    assert a["decision"] == {"offline_neonatal": "neonatal", "realtime_neonatal": "neonatal"}
    assert a["secondary"]["fa_per_hour_excluding_ambiguous"]["offline-threshold-default"] == 0.0
    b = assess(metrics(neo=(9, 40), gen=(8, 25)))  # neonatal: more FA than threshold
    assert b["decision"]["offline_neonatal"] == "general"
    c = assess(metrics(neo=(3, 5), gen=(2, 5)))
    assert c["decision"]["realtime_neonatal"] == "threshold"
    d = assess(metrics(neo=(9, 28), gen=(9, 10)))  # both ok, neonatal noisier than general
    assert d["decision"]["offline_neonatal"] == "general"
    assert "неонатальная модель" in markdown_report(a)

    cfg = tmp_path / "ml.yaml"
    shutil.copy("configs/ml.yaml", cfg)
    fz = {p: {"A_replacement": {"params": {"threshold": 0.6, "min_epochs": 8}},
              "model": {"path": f"configs/models/ictal_gbm_{p}_neo1.json"},
              "calibration": {"path": f"configs/calibration.ml_{p}_neo1.json"}} for p in ("offline", "realtime")}
    apply_decision(a, fz, str(cfg))
    doc = yaml.safe_load(cfg.read_text())
    assert doc["neonatal"]["offline"] == "neonatal" and doc["neonatal"]["realtime_model"]["min_epochs"] == 8
    apply_decision(b, fz, str(cfg))  # re-applying replaces the block
    doc = yaml.safe_load(cfg.read_text())
    assert doc["neonatal"] == {"offline": "general", "realtime": "general"}
    assert doc["offline"]["model"].endswith("offline_v2.json") and "provenance" in doc
    from neurolens.contracts.config import LearnedDetectorConfig

    LearnedDetectorConfig(**doc)  # valid against the contract


def test_plain_edf_fallback_matches_pyedflib_and_reads_noncompliant_header(demo_edf, tmp_path):
    """Helsinki eeg50.edf declares EDF+C without an annotation signal; pyedflib
    rejects it, the fallback reads the samples identically."""
    import numpy as np
    import pyedflib

    from neurolens.layer1_ingest.edf import PlainEdfReader
    from neurolens.layer1_ingest.registry import ingest

    ref = pyedflib.EdfReader(str(demo_edf))
    alt = PlainEdfReader(demo_edf)
    labels = [lb for lb in ref.getSignalLabels() if lb != "EDF Annotations"]
    assert alt.getSignalLabels() == labels
    for i in range(alt.signals_in_file):
        j = ref.getSignalLabels().index(labels[i])
        assert np.allclose(alt.readSignal(i), ref.readSignal(j), atol=1e-6)
        assert alt.getSampleFrequency(i) == ref.getSampleFrequency(j)
    ref.close()

    plain = tmp_path / "plain.edf"
    names = ["EEG Fp1-REF", "EEG Fp2-REF", "EEG C3-REF", "EEG C4-REF", "ECG EKG-REF"]
    w = pyedflib.EdfWriter(str(plain), len(names), file_type=pyedflib.FILETYPE_EDF)
    w.setSignalHeaders([{"label": n, "dimension": "uV", "sample_frequency": 256, "physical_min": -500,
                         "physical_max": 500, "digital_min": -32768, "digital_max": 32767} for n in names])
    rng = np.random.default_rng(0)
    w.writeSamples([rng.normal(0, 30, 256 * 20) for _ in names])
    w.close()
    raw = bytearray(plain.read_bytes())
    raw[192:197] = b"EDF+C"  # declare EDF+ without the mandatory annotation signal
    bad = tmp_path / "bad.edf"
    bad.write_bytes(bytes(raw))
    with pytest.raises(OSError):
        pyedflib.EdfReader(str(bad))
    sig = ingest(bad)
    good = pyedflib.EdfReader(str(plain))
    assert sig.channel_names[:4] == ["Fp1", "Fp2", "C3", "C4"]
    assert np.allclose(sig.signal[0], good.readSignal(0), atol=0.02)
    good.close()


def test_default_config_neonatal_decision(configs):
    """Increment-12 decision in configs/ml.yaml: neonates -> threshold detector offline,
    neonatal model in the monitor; the general v2 model is never used for neonates."""
    ml = configs.ml
    assert ml.select("offline", neonate=True) is None
    rt = ml.select("realtime", neonate=True)
    assert rt is not None and rt.model.endswith("ictal_gbm_realtime_neo1.json")
    assert ml.select("offline").model.endswith("offline_v2.json")


def test_open_edf_reads_file_shorter_than_header(tmp_path):
    """CAP brux1.edf: header record count exceeds the data on disk; the fallback
    reads the records that exist and reports how many are missing."""
    import numpy as np
    import pyedflib

    from neurolens.layer1_ingest.edf import PlainEdfReader, open_edf

    plain = tmp_path / "plain.edf"
    w = pyedflib.EdfWriter(str(plain), 2, file_type=pyedflib.FILETYPE_EDF)
    w.setSignalHeaders([{"label": f"C{i}", "dimension": "uV", "sample_frequency": 100, "physical_min": -500,
                         "physical_max": 500, "digital_min": -32768, "digital_max": 32767} for i in range(2)])
    w.writeSamples([np.random.default_rng(i).normal(0, 20, 1000) for i in range(2)])
    w.close()
    raw = plain.read_bytes()
    cut = tmp_path / "cut.edf"
    cut.write_bytes(raw[: len(raw) - 2 * 2 * 100 * 3])  # drop the last 3 one-second records
    r = open_edf(cut)
    assert isinstance(r, PlainEdfReader) and r.truncated_records == 3
    assert len(r.readSignal(0)) == 700 and r.getStartdatetime().year >= 2000
