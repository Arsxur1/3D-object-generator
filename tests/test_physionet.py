"""PhysioNet integration — offline tests (no network): parsers, client cache/
checksum, bipolar reconstruction, event-level scoring."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pyedflib
import pytest

from neurolens.datasets import PhysioNetClient, parse_chbmit_summary, parse_siena_seizure_list
from neurolens.datasets.annotations import SeizureInterval
from neurolens.datasets.physionet import ChecksumError
from neurolens.evaluation.matching import Detection, aggregate, score_record
from neurolens.layer1_ingest.bipolar import (
    is_bipolar_recording,
    parse_bipolar_label,
    reconstruct_average_reference,
)
from neurolens.layer1_ingest.registry import ingest

CHB_SUMMARY = """Data Sampling Rate: 256 Hz
*************************

Channels in EDF Files:
**********************
Channel 1: FP1-F7
Channel 2: F7-T7

File Name: chb01_01.edf
File Start Time: 11:42:54
File End Time: 12:42:54
Number of Seizures in File: 0

File Name: chb01_03.edf
File Start Time: 13:43:04
File End Time: 14:43:04
Number of Seizures in File: 1
Seizure Start Time: 2996 seconds
Seizure End Time: 3036 seconds

Channels changed:
Channels in EDF Files:
Channel 1: FP1-F7

File Name: chb01_04.edf
Number of Seizures in File: 2
Seizure 1 Start Time: 10 seconds
Seizure 1 End Time: 40 seconds
Seizure 2 Start Time: 100 seconds
Seizure 2 End Time: 130 seconds
"""

SIENA_LIST = """Data Sampling Rate: 512 Hz

Seizure n 1
File name: PN00-1.edf
Registration start time: 19.39.33
Registration end time:  20.22.58
Seizure start time: 19.58.36
Seizure end time: 19.59.46

Seizure n 3
File name: PN00-3.edf
Registration start time: 23.50.00
Registration end time: 00.40.00
Seizure start time: 00.10.00
Seizure end time: 01.11.00
"""

CHB_LABELS = (
    "FP1-F7 F7-T7 T7-P7 P7-O1 FP1-F3 F3-C3 C3-P3 P3-O1 FP2-F4 F4-C4 C4-P4 P4-O2 "
    "FP2-F8 F8-T8 T8-P8-0 P8-O2 FZ-CZ CZ-PZ P7-T7 T7-FT9 FT9-FT10 FT10-T8 T8-P8-1"
).split()


def test_chbmit_summary_parser():
    recs = parse_chbmit_summary(CHB_SUMMARY, "chb01")
    assert [r.file for r in recs] == ["chb01/chb01_01.edf", "chb01/chb01_03.edf", "chb01/chb01_04.edf"]
    assert recs[0].seizures == []
    assert recs[1].seizures == [SeizureInterval(2996, 3036)]
    assert len(recs[2].seizures) == 2 and recs[2].seizures[1].onset_s == 100
    assert recs[1].channels == ["FP1-F7", "F7-T7"] and recs[2].channels == ["FP1-F7"]
    assert recs[1].sampling_rate_hz == 256
    assert not any(r.warnings for r in recs)


def test_siena_parser_midnight_and_typo():
    recs = parse_siena_seizure_list(SIENA_LIST, "PN00")
    assert recs[0].seizures == [SeizureInterval(1143, 1213)]
    assert recs[0].duration_s == 43 * 60 + 25
    # midnight roll-over: onset 20 min after 23:50 ; end hour typo -> +60 s
    assert recs[1].seizures[0].onset_s == 20 * 60
    assert recs[1].seizures[0].duration_s == 60
    assert recs[1].warnings


def test_bipolar_label_parsing():
    assert parse_bipolar_label("FP1-F7") == ("Fp1", "F7")
    assert parse_bipolar_label("T8-P8-0") == ("T4", "T6")
    assert parse_bipolar_label("EEG FT9-FT10") == ("Ft9", "Ft10")
    for lb in ("-", "ECG", "Fp1-Ref", "C3-A1", "LOC-ROC", "VNS"):
        assert parse_bipolar_label(lb) is None, lb
    assert is_bipolar_recording(CHB_LABELS + ["-", "ECG"])
    assert not is_bipolar_recording(["Fp1", "Fp2", "C3", "C4", "O1", "O2"])


def test_bipolar_reconstruction_is_exact_average_reference():
    rng = np.random.default_rng(1)
    pairs = [parse_bipolar_label(lb) for lb in CHB_LABELS]
    names = sorted({e for p in pairs for e in p})
    V = rng.normal(size=(len(names), 500)) + 7.0
    pot = dict(zip(names, V))
    D = np.array([pot[a] - pot[b] for a, b in pairs])
    rec = reconstruct_average_reference(D, CHB_LABELS)
    # Fz-Cz-Pz chain is disconnected from the rest -> dropped, not guessed
    assert rec.dropped_electrodes == ["Cz", "Fz", "Pz"]
    keep = np.array([pot[e] for e in rec.electrodes])
    np.testing.assert_allclose(rec.data, keep - keep.mean(0), atol=1e-4)
    assert rec.residual_rms_uv < 1e-6


def test_bipolar_edf_ingest(tmp_path):
    fs, n = 256, 256 * 4
    rng = np.random.default_rng(2)
    pairs = [parse_bipolar_label(lb) for lb in CHB_LABELS]
    names = sorted({e for p in pairs for e in p})
    pot = {e: 20 * rng.normal(size=n) for e in names}
    labels = CHB_LABELS[:15] + ["T8-P8"] + CHB_LABELS[15:22] + ["ECG"]
    sigs = [pot[a] - pot[b] for a, b in (parse_bipolar_label(lb) for lb in labels[:-1])]
    sigs.append(100 * np.sin(np.arange(n) / fs * 2 * np.pi * 1.2))
    path = tmp_path / "bip.edf"
    w = pyedflib.EdfWriter(str(path), len(labels), file_type=pyedflib.FILETYPE_EDFPLUS)
    w.setSignalHeaders([
        {"label": lb, "dimension": "uV", "sample_frequency": fs, "physical_max": 1000,
         "physical_min": -1000, "digital_max": 32767, "digital_min": -32768}
        for lb in labels
    ])
    w.writeSamples([np.asarray(s, dtype=np.float64) for s in sigs])
    w.close()

    sig = ingest(path)
    assert sig.reference == "average_from_bipolar"
    assert "Fz" not in sig.channel_names and "ECG" in sig.channel_names
    assert any(f.startswith("reconstructed_from_bipolar") for f in sig.quality.flags)
    i = sig.channel_names.index("Fp1")
    eeg = [c for c in sig.channel_names if c != "ECG"]
    truth = np.array([pot[e] for e in eeg])
    truth -= truth.mean(0)
    assert np.corrcoef(sig.signal[i], truth[eeg.index("Fp1")])[0, 1] > 0.999


def test_scoring_ovlp():
    truth = [SeizureInterval(100, 140), SeizureInterval(1000, 1050)]
    dets = [
        Detection(110, 130, 0.9),        # TP (seizure 1), latency +10
        Detection(132, 150, 0.8),        # merged into the same detection
        Detection(2000, 2020, 0.7),      # FP
        Detection(2025, 2030, 0.7),      # merged with previous FP (gap 5 s)
    ]
    sc = score_record("r", truth, dets, duration_s=3600)
    assert (sc.tp, sc.fn, sc.fp) == (1, 1, 1)
    assert sc.latencies_s == [10]
    # pre-onset tolerance: detection 20 s before onset counts, latency negative
    sc2 = score_record("r", [SeizureInterval(100, 140)], [Detection(75, 85)], 3600)
    assert sc2.tp == 1 and sc2.latencies_s == [-25]
    agg = aggregate([sc, sc2])
    assert agg.n_seizures == 3 and agg.tp == 2 and agg.fa_per_hour == pytest.approx(0.5)
    assert agg.sensitivity == pytest.approx(2 / 3)


def test_client_cache_and_checksum(tmp_path):
    payload = b"0" * 1000
    good = hashlib.sha256(payload).hexdigest()
    texts = {
        "SHA256SUMS.txt": f"{good} chb01/chb01_01.edf\n{'f' * 64} chb01/chb01_02.edf\n".encode(),
        "RECORDS": b"chb01/chb01_01.edf\nchb01/chb01_02.edf\n",
    }
    calls = []

    def fetch_text(url):
        calls.append(url)
        return texts[url.split("1.0.0/")[1]]

    def fetch_file(url, dest: Path):
        calls.append(url)
        dest.write_bytes(payload)

    c = PhysioNetClient("chbmit", tmp_path, fetch_text=fetch_text, fetch_file=fetch_file)
    assert c.subjects() == ["chb01"]
    p = c.fetch("chb01/chb01_01.edf")
    assert p.read_bytes() == payload and c.is_cached("chb01/chb01_01.edf")
    n = len(calls)
    c.fetch("chb01/chb01_01.edf")  # cache hit: no network
    assert len(calls) == n
    with pytest.raises(ChecksumError):
        c.fetch("chb01/chb01_02.edf")
    assert not c.is_cached("chb01/chb01_02.edf")
    assert not list(tmp_path.rglob("*.part"))


def test_overrides_and_tuning_on_synthetic(tmp_path, demo_edf, configs):
    import copy

    import yaml

    from neurolens.datasets.annotations import RecordAnnotation
    from neurolens.evaluation.runner import evaluate_offline, prepare_records
    from neurolens.evaluation.tune import fit_ictal_calibration, grid_search, write_overrides
    from neurolens.pipeline.config_loader import apply_overrides

    cfg = copy.deepcopy(configs)
    ann = RecordAnnotation("synthetic", "s0", "s0/demo.edf", [SeizureInterval(180, 240)])
    cache = tmp_path / "feat"
    recs = prepare_records([(ann, demo_edf)], cfg, cache)
    assert list(cache.glob("*.pkl"))
    recs2 = prepare_records([(ann, demo_edf)], cfg, cache)  # served from cache
    assert recs2[0].duration_s == recs[0].duration_s

    base = evaluate_offline(recs, cfg.thresholds).total
    assert base.tp == 1  # the synthetic seizure is found

    grid = {"ictal_min_channels": [1, 3], "ictal_min_duration_s": [8.0, 12.0]}
    res = grid_search(recs, cfg.thresholds, grid=grid, fa_target=1.0)
    assert len(res.trials) == 4 and res.best.score.sensitivity == 1.0

    out = write_overrides(tmp_path / "ovr.yaml", res.best.params, {"database": "synthetic"})
    data = yaml.safe_load(out.read_text())
    assert set(data["thresholds"]) == set(grid)
    apply_overrides(cfg, out)
    for k, v in res.best.params.items():
        assert getattr(cfg.thresholds, k) == v

    cal, metrics = fit_ictal_calibration(recs, cfg.thresholds)
    assert "n" in metrics  # too few events here -> skipped, but reported


def test_overrides_reject_unknown_keys(tmp_path, configs):
    import copy

    from pydantic import ValidationError

    from neurolens.pipeline.config_loader import apply_overrides

    p = tmp_path / "bad.yaml"
    p.write_text("thresholds:\n  not_a_threshold: 3\n")
    with pytest.raises(ValidationError):
        apply_overrides(copy.deepcopy(configs), p)
    p.write_text("realtime:\n  alarms:\n    seizure:\n      persistence_windows: 3\n")
    cfg = apply_overrides(copy.deepcopy(configs), p)
    assert cfg.realtime.alarms["seizure"].persistence_windows == 3
    assert cfg.realtime.alarms["seizure"].refractory_s == configs.realtime.alarms["seizure"].refractory_s


def test_siena_mixed_separators_and_file_resolution():
    from neurolens.datasets.physionet import _resolve_files

    text = """Seizure n 1:
File name: PN11-.edf
Registration start time: 15.51.31
Seizure start time: 16:13.23
Seizure end time:16.14.26
"""
    anns = parse_siena_seizure_list(text, "PN11")
    assert anns[0].seizures == [SeizureInterval(1312, 1375)]
    out = _resolve_files(anns, ["PN11/PN11-1.edf"])
    assert out[0].file == "PN11/PN11-1.edf" and out[0].warnings


def test_resumable_download_survives_cut_transfers(tmp_path, monkeypatch):
    import io

    import neurolens.datasets.physionet as pn

    payload = bytes(range(256)) * 8  # 2048 bytes
    seen_ranges = []

    class Resp(io.BytesIO):
        def __init__(self, body, status, length):
            super().__init__(body)
            self.status = status
            self.headers = {"Content-Length": str(length)}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        rng = req.headers.get("Range")
        seen_ranges.append(rng)
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        rest = payload[start:]
        # the "proxy" cuts every response after 300 bytes, without an error
        return Resp(rest[:300], 206 if rng else 200, len(rest))

    monkeypatch.setattr(pn.urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "f.part"
    pn._download_resumable("https://x/f", dest)
    assert dest.read_bytes() == payload
    assert seen_ranges[0] is None and seen_ranges[1] == "bytes=300-"
    assert len(seen_ranges) == 7


def test_realtime_replay_matches_live_monitor_and_overrides_roundtrip(tmp_path, demo_edf, configs):
    import copy

    from neurolens.datasets.annotations import RecordAnnotation
    from neurolens.evaluation.realtime_replay import (
        build_traces,
        evaluate_replay,
        replay,
        with_realtime_params,
    )
    from neurolens.evaluation.runner import realtime_detections
    from neurolens.evaluation.tune import grid_search_realtime, write_overrides
    from neurolens.pipeline.config_loader import apply_overrides

    cfg = copy.deepcopy(configs)
    ann = RecordAnnotation("synthetic", "s0", "s0/demo.edf", [SeizureInterval(180, 240)])
    tr = build_traces([(ann, demo_edf)], cfg, tmp_path / "c", workers=1)[0]
    assert tr.windows and len(tr.windows[0]) == 4

    def key(ds):
        return [(d.t_start, d.t_end, d.confidence) for d in ds]

    for params in ({}, {"ictal_min_channels": 2, "persistence_windows": 1}):
        c2 = copy.deepcopy(cfg)
        c2.realtime = with_realtime_params(cfg.realtime, params)
        live, _ = realtime_detections(demo_edf, c2)
        assert key(replay(tr, c2.thresholds, c2.realtime)) == key(live)

    res = grid_search_realtime([tr], cfg.thresholds, cfg.realtime,
                               grid={"ictal_min_channels": [1, 2], "persistence_windows": [1, 2]})
    assert len(res.trials) == 4 and res.best.score.sensitivity == 1.0
    out = write_overrides(tmp_path / "o.yaml", {"ictal_min_channels": 3}, {}, realtime_params=res.best.params)
    c3 = apply_overrides(copy.deepcopy(cfg), out)
    assert c3.thresholds.ictal_min_channels == 3
    assert c3.realtime.alarms["seizure"].persistence_windows == res.best.params["persistence_windows"]
    assert c3.realtime.threshold_overrides["ictal_min_channels"] == res.best.params["ictal_min_channels"]
    assert evaluate_replay([tr], c3.thresholds, c3.realtime).total.tp == 1


def _trial(params, sens, fa, lat=5.0):
    from neurolens.evaluation.matching import AggregateScore
    from neurolens.evaluation.tune import Trial

    return Trial(params, AggregateScore(1, 10.0, 10, int(sens * 10), 10 - int(sens * 10), int(fa * 10),
                                        sens, fa, lat, lat, 0.0))


def test_robust_selector_avoids_sharp_corner():
    from neurolens.evaluation.tune import TuneResult

    grid = {"a": [1, 2, 3], "b": [1, 2]}
    # (a=3,b=2) has zero FA but its neighbours lose sensitivity -> sharp corner;
    # (a=1,b=1) is the only point whose whole neighbourhood keeps full sensitivity
    # ((1,2) has neighbour (2,2) at 0.8).
    table = {
        (1, 1): (1.0, 0.6), (1, 2): (1.0, 0.4), (2, 1): (1.0, 0.5),
        (2, 2): (0.8, 0.2), (3, 1): (0.8, 0.1), (3, 2): (1.0, 0.0),
    }
    trials = [_trial({"a": a, "b": b}, *v) for (a, b), v in table.items()]
    res = TuneResult(best=trials[0], baseline=trials[0], trials=trials, fa_target=1.0, grid=grid)
    assert res.select("strict").params == {"a": 3, "b": 2}
    assert res.select("robust").params == {"a": 1, "b": 1}
    # margin: own FA in budget + full own sensitivity, then best worst-neighbour sensitivity
    assert res.select("margin").params == {"a": 1, "b": 1}
    with pytest.raises(ValueError):
        res.select("nope")


def test_segmented_download_reassembles_with_cuts(tmp_path, monkeypatch):
    import io

    import neurolens.datasets.physionet as pn

    payload = bytes((i * 7) % 251 for i in range(10_000))
    ranges = []

    class Resp(io.BytesIO):
        def __init__(self, body, status, length):
            super().__init__(body)
            self.status, self.headers = status, {"Content-Length": str(length)}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        rng = req.headers.get("Range")
        ranges.append(rng)
        a, b = rng.split("=")[1].split("-")
        a, b = int(a), (int(b) if b else len(payload) - 1)
        body = payload[a:b + 1]
        return Resp(body[:700], 206, len(body))  # proxy cuts every response at 700 B

    monkeypatch.setattr(pn.urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "f.part"
    pn._download_segmented("https://x/f", dest, len(payload), segments=4)
    assert dest.read_bytes() == payload
    assert not list(tmp_path.glob("*.seg*"))
    assert all(r and "-" in r for r in ranges) and len(ranges) > 4


def test_prepare_records_parallel_matches_serial(tmp_path, demo_edf, configs):
    import shutil

    import numpy as np

    from neurolens.datasets.annotations import RecordAnnotation
    from neurolens.evaluation.runner import prepare_records

    other = tmp_path / "demo2.edf"
    shutil.copy(demo_edf, other)
    items = [(RecordAnnotation("s", "s", "s/a.edf"), demo_edf),
             (RecordAnnotation("s", "s", "s/b.edf"), other)]
    par = prepare_records(items, configs, tmp_path / "c1", workers=2)
    ser = prepare_records(items, configs, None, workers=1)
    assert [r.annotation.file for r in par] == ["s/a.edf", "s/b.edf"]
    np.testing.assert_allclose(par[1].features.epoch_rms, ser[1].features.epoch_rms)


def test_prereg_assessment_logic():
    from neurolens.evaluation.prereg import assess, clopper_pearson

    lo, hi = clopper_pearson(24, 24)
    assert hi == 1.0 and abs(lo - 0.8575) < 1e-3

    def rec(file, n, tp, fp, hours=1.0):
        return {"file": file, "hours": hours, "n_seizures": n, "tp": tp, "fp": fp,
                "latencies_s": [5.0] * tp}

    def mode(name, tp_c, fp_c, tp_s, fp_s):
        return {"mode": name, "records": [rec("chb02/a.edf", 10, tp_c, fp_c, 5.0),
                                          rec("PN05/b.edf", 10, tp_s, fp_s, 5.0)]}

    metrics = {"results": [
        mode("offline-default", 10, 40, 10, 40), mode("offline-strict", 9, 1, 8, 1),
        mode("offline-robust", 8, 1, 8, 0), mode("offline-margin", 10, 4, 9, 4),
        mode("realtime-default", 10, 20, 10, 15), mode("realtime-strict", 10, 3, 9, 3),
        mode("realtime-robust", 9, 1, 8, 1), mode("realtime-margin", 10, 3, 9, 3),
    ]}
    a = assess(metrics)
    assert a["hypotheses"]["H1_offline_margin_sens_ge_strict"] is True
    assert a["hypotheses"]["H2_offline_margin_fa_le_budget"] is True     # 8 FA / 10 h
    assert a["hypotheses"]["H3_default_most_sensitive_but_over_budget"] is True
    assert a["hypotheses"]["H4_realtime_margin_sens_ge_0.9_and_fa_le_budget"] is True
    assert a["decision"]["adopt_realtime_margin_as_monitor_default"] is True  # 0.95 >= 0.95
    assert a["by_population"]["offline-margin"]["siena"]["tp"] == 9


def test_run_memory_bounded_respects_budget(tmp_path):
    from neurolens.evaluation.runner import run_memory_bounded

    got = []
    # budget 10: tasks 8 and 7 can never overlap; 2-unit tasks may run alongside
    run_memory_bounded(_sleep_echo, [(i,) for i in range(5)], [8, 7, 2, 2, 2], 3,
                       got.append, budget=10)
    assert sorted(got) == [0, 1, 2, 3, 4]


def _sleep_echo(i):
    import time

    time.sleep(0.05)
    return i


def test_segmented_download_ignores_overlong_range_responses(tmp_path, monkeypatch):
    import io

    import neurolens.datasets.physionet as pn

    payload = bytes((i * 13) % 251 for i in range(8_000))

    class Resp(io.BytesIO):
        def __init__(self, body, status, length):
            super().__init__(body)
            self.status, self.headers = status, {"Content-Length": str(length)}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        a = int(req.headers.get("Range").split("=")[1].split("-")[0])
        body = payload[a:]  # server ignores the range END and sends to EOF
        return Resp(body, 206, len(body))

    monkeypatch.setattr(pn.urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "f.part"
    pn._download_segmented("https://x/f", dest, len(payload), segments=4)
    assert dest.read_bytes() == payload
