"""REST API tests (TZ §12, §14) via FastAPI TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from neurolens.api.app import create_app
from neurolens.api.deid import deidentify_result, hash_identifier
from neurolens.feedback.log import FeedbackLogger


@pytest.fixture()
def client_and_paths(configs, tmp_path):
    audit = tmp_path / "audit.jsonl"
    fb = tmp_path / "feedback.jsonl"
    app = create_app(config=configs, audit_path=audit, feedback_path=fb)
    return TestClient(app), audit, fb


def test_health_and_version(client_and_paths):
    client, _, _ = client_and_paths
    assert client.get("/health").json()["status"] == "ok"
    v = client.get("/version").json()
    assert v["neurolens_version"]
    assert v["rule_base_version"]
    assert "event" in v["schemas"]


def test_montages(client_and_paths):
    client, _, _ = client_and_paths
    names = {m["name"] for m in client.get("/config/montages").json()}
    assert {"double_banana", "referential", "transverse"} <= names


def test_analyze_path_deidentified(client_and_paths, demo_edf):
    client, audit, _ = client_and_paths
    r = client.post("/analyze", json={"path": str(demo_edf), "provider": "deterministic", "age_years": 55})
    assert r.status_code == 200
    j = r.json()
    assert len(j["events"]) > 0
    # deidentified: source filename scrubbed, subject hash present
    assert j["recording"]["provenance"]["source_file"] is None
    assert j["subject_id"] and len(j["subject_id"]) == 16
    assert audit.exists()


def test_analyze_missing_file(client_and_paths):
    client, _, _ = client_and_paths
    assert client.post("/analyze", json={"path": "/no/such.edf"}).status_code == 404


def test_analyze_upload(client_and_paths, demo_edf):
    client, _, _ = client_and_paths
    with open(demo_edf, "rb") as fh:
        r = client.post(
            "/analyze/upload",
            files={"file": ("rec.edf", fh, "application/octet-stream")},
            data={"provider": "deterministic", "mode": "A"},
        )
    assert r.status_code == 200
    assert len(r.json()["events"]) > 0


def test_monitor(client_and_paths, demo_edf):
    client, _, _ = client_and_paths
    r = client.post("/monitor", json={"path": str(demo_edf)})
    assert r.status_code == 200
    j = r.json()
    assert "alarms" in j and j["n_windows"] > 0
    assert j["seizure_burden"] is not None
    # source deidentified to the subject hash
    assert j["source"] == j["subject_id"]


def test_feedback_recorded(client_and_paths):
    client, _, fb = client_and_paths
    r = client.post("/feedback", json={
        "recording_id": "rec1", "target_kind": "event",
        "target_ref": "diffuse_slowing", "action": "confirm",
    })
    assert r.status_code == 200
    records = FeedbackLogger(fb).read_all()
    assert len(records) == 1 and records[0].target_ref == "diffuse_slowing"


def test_schema_endpoint(client_and_paths):
    client, _, _ = client_and_paths
    assert client.get("/schemas/event").json()["title"] == "Event"
    assert client.get("/schemas/does_not_exist").status_code == 404


def test_audit_log_written(client_and_paths, demo_edf):
    client, audit, _ = client_and_paths
    client.post("/analyze", json={"path": str(demo_edf), "provider": "deterministic"})
    lines = audit.read_text(encoding="utf-8").splitlines()
    assert lines
    import json
    entry = json.loads(lines[-1])
    assert entry["endpoint"] == "/analyze" and entry["subject_id"]
    assert entry["neurolens_version"]


def test_deid_unit():
    result = {"recording": {"provenance": {"source_file": "patient_john.edf"}}}
    out, sid = deidentify_result(result)
    assert out["recording"]["provenance"]["source_file"] is None
    assert out["subject_id"] == sid == hash_identifier("patient_john.edf")
    # original untouched (deep copy)
    assert result["recording"]["provenance"]["source_file"] == "patient_john.edf"
