"""Increment 16 plumbing: Natus/NK label prefixes, EDF+ TAL parsing, the IFCN dataset
reader, AUC/bootstrap, the pre-registered rule and candidate annotation."""

import io
import zipfile

import numpy as np
import pytest

from neurolens.datasets.kural_ied import edf_name, parse_gold
from neurolens.evaluation import kural_eval as K
from neurolens.evaluation import prereg16
from neurolens.layer1_ingest.edf import parse_tal_record
from neurolens.layer1_ingest.electrodes import is_eeg_channel, normalize_channel_name


@pytest.mark.parametrize("raw,canon", [
    ("E FP1-Ref", "Fp1"), ("E T7-Ref", "T3"), ("E P10-Ref", "P10"), ("P EKG", "ECG"),
    ("POL ECG", "ECG"), ("EEG Fp1-REF", "Fp1"), ("Pz", "Pz"), ("P3", "P3"), ("E Fz-Ref", "Fz"),
])
def test_type_prefixes(raw, canon):
    assert normalize_channel_name(raw) == canon


def test_prefixed_eeg_counts_as_eeg():
    assert is_eeg_channel(normalize_channel_name("E C3-Ref"))
    assert not is_eeg_channel(normalize_channel_name("P EKG"))


def test_parse_tal_record():
    rec = b"+1\x14\x14\x00+6.876146\x14Marker: Annotation\x14\x00+2.5\x150.5\x14A\x14B\x14\x00\x00\x00"
    onset, anns = parse_tal_record(rec)
    assert onset == 1.0
    assert (6.876146, 0.0, "Marker: Annotation") in anns
    assert (2.5, 0.5, "A") in anns and (2.5, 0.5, "B") in anns


def _xlsx(rows):
    def cell(ref, v):
        return f'<c r="{ref}" t="inlineStr"><is><t>{v}</t></is></c>'
    body = "".join("<row>" + "".join(cell(f"{chr(65 + j)}{i + 1}", v) for j, v in enumerate(r)) + "</row>"
                   for i, r in enumerate(rows))
    xml = ('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
           f"<sheetData>{body}</sheetData></worksheet>")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", xml)
    return buf.getvalue()


def test_parse_gold_and_edf_names():
    data = _xlsx([["Samples", "Gender", "Age", "Gold Standard"], ["S1", "F", "34", "Non_Epileptic"],
                  ["S80", "M", "12", "Epileptic"]])
    cases = parse_gold(data, {"S01.edf", "S80-.edf"})
    assert [c.file for c in cases] == ["S01.edf", "S80-.edf"]
    assert [c.epileptic for c in cases] == [False, True] and cases[0].age_years == 34
    with pytest.raises(KeyError):
        edf_name("S7", {"S01.edf"})


def test_auc_and_bootstrap():
    y = np.array([True, True, False, False])
    assert K.auc(np.array([0.9, 0.8, 0.1, 0.2]), y) == 1.0
    assert K.auc(np.array([0.5, 0.5, 0.5, 0.5]), y) == 0.5
    b = K.bootstrap_auc(np.array([0.9, 0.8, 0.1, 0.2]), np.array([0.1, 0.2, 0.9, 0.8]), y, n=200)
    assert b["ci95"][0] == 1.0 and b["diff_ci95"][0] == 2.0 - 1.0


def _summary(auc_l, ci_lo, auc_h, spec):
    b = {"sensitivity": 0.7, "sensitivity_ci95": [0.5, 0.8], "specificity": spec,
         "specificity_ci95": [spec - 0.1, min(1, spec + 0.1)], "accuracy": 0.8}
    return {"n": 100, "n_epileptic": 54, "n_non_epileptic": 46,
            "learned": {"auc": auc_l, "auc_ci95": [ci_lo, 0.9], "at_frozen_threshold": b},
            "heuristic": {"auc": auc_h, "auc_ci95": [0.4, 0.7], "at_default_z": b},
            "auc_difference_learned_minus_heuristic": auc_l - auc_h, "auc_difference_ci95": [0.0, 0.3]}


def test_prereg16_rule_and_apply(tmp_path):
    good = prereg16.assess(_summary(0.8, 0.7, 0.6, 0.85))
    assert all(good["hypotheses"].values()) and good["decision"]["annotate_heuristic_candidates_with_learned_probability"]
    assert not prereg16.assess(_summary(0.8, 0.45, 0.6, 0.85))["decision"][
        "annotate_heuristic_candidates_with_learned_probability"]
    assert not prereg16.assess(_summary(0.8, 0.7, 0.78, 0.85))["hypotheses"]["H2_learned_auc_exceeds_heuristic_by_0.05"]
    assert not prereg16.assess(_summary(0.8, 0.7, 0.6, 0.7))["hypotheses"][
        "H3_learned_specificity_at_threshold_at_least_0.80"]
    assert "подтверждена" in prereg16.markdown_report(good)

    import yaml

    from neurolens.contracts.config import IEDConfig

    y = tmp_path / "ied.yaml"
    y.write_text("enabled: false\nmodel: m.json\nthreshold: 0.95\nmax_events: 200\n"
                 "apply_to_neonates: false\nprovenance:\n  report: docs/increment15_report.md\n")
    prereg16.apply_decision(good, str(y))
    cfg = IEDConfig(**yaml.safe_load(y.read_text()))
    assert cfg.annotate_candidates and not cfg.enabled
    assert cfg.provenance["external_validation"] == "docs/increment16_report.md"
    prereg16.apply_decision(prereg16.assess(_summary(0.6, 0.4, 0.6, 0.5)), str(y))
    assert IEDConfig(**yaml.safe_load(y.read_text())).annotate_candidates is False


def test_annotate_candidates_adds_probability():
    from neurolens.contracts.events import Event, Localization
    from neurolens.layer4_detect import ied_ml

    fs, n = 200.0, 6000
    names = list(ied_ml.STANDARD_19)
    rng = np.random.default_rng(1)
    x = rng.normal(0, 10, (len(names), n))

    class Sig:
        signal, sampling_rate_hz, channel_names = x, fs, names

    class Const:
        def __init__(self, v):
            self.v = v

        def predict_proba(self, X):
            return np.full(X.shape[0], self.v)

    ev = Event(code="spike", label_ru="с", label_uz="s", group="ied",
               localization=Localization(channels=["C3"]), t_start=10.0, t_end=10.1, confidence=0.5)
    other = Event(code="ictal_rhythm", label_ru="и", label_uz="i", group="ictal",
                  localization=Localization(channels=["C3"]), t_start=1.0, t_end=5.0, confidence=0.5)
    assert ied_ml.annotate_candidates(Const(0.97), Sig, [ev, other], 0.95) == 1
    assert ev.metadata["ied_probability_learned"] == 0.97
    assert "ied_probability_learned" not in other.metadata
    ev2 = ev.model_copy(deep=True)
    ev2.metadata = {}
    assert ied_ml.annotate_candidates(Const(0.2), Sig, [ev2], 0.95) == 0
