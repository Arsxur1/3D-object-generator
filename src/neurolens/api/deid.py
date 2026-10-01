"""PHI deidentification (TZ §12).

Before a result leaves the core (returned to a client or sent to the cloud/LLM),
direct identifiers are removed. The recording is tagged with a stable, one-way
``subject_id`` (a salted hash of the source identifier) so records can be
correlated for serial comparison without exposing PHI. Clinical fields (age,
postmenstrual age, context) are retained — they are needed for interpretation
and are not directly identifying.
"""

from __future__ import annotations

import copy
import hashlib
import os
from typing import Any

_SALT = os.environ.get("NEUROLENS_DEID_SALT", "neurolens")


def hash_identifier(value: str) -> str:
    """One-way, salted 16-hex-char subject id."""
    h = hashlib.sha256((_SALT + "|" + value).encode("utf-8")).hexdigest()
    return h[:16]


def deidentify_result(result: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Return (deidentified_copy, subject_id).

    Removes ``recording.provenance.source_file`` and stamps a ``subject_id``.
    """
    out = copy.deepcopy(result)
    rec = out.get("recording", {}) or {}
    prov = rec.get("provenance", {}) or {}
    source = prov.get("source_file") or rec.get("source") or "unknown"
    subject_id = hash_identifier(str(source))

    if "source_file" in prov:
        prov["source_file"] = None
    rec["provenance"] = prov
    rec["subject_id"] = subject_id
    out["recording"] = rec
    out["subject_id"] = subject_id
    return out, subject_id
