"""Open annotated EEG databases (PhysioNet) for validation and tuning (TZ §16).

NeuroLens never downloads data implicitly: records are fetched on explicit
request (``neurolens physionet fetch``) into a local, gitignored cache and
verified against the database's published SHA-256 checksums.

Supported databases:
  * CHB-MIT Scalp EEG (``chbmit``) — pediatric, 256 Hz, bipolar, seizure labels
  * Siena Scalp EEG (``siena``)    — adult, 512 Hz, referential, seizure labels
"""

from __future__ import annotations

from .annotations import RecordAnnotation, SeizureInterval
from .chbmit import parse_chbmit_summary
from .physionet import DATABASES, PhysioNetClient
from .siena import parse_siena_seizure_list

__all__ = [
    "DATABASES",
    "PhysioNetClient",
    "RecordAnnotation",
    "SeizureInterval",
    "parse_chbmit_summary",
    "parse_siena_seizure_list",
]
