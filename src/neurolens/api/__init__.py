"""REST API for NeuroLens (TZ §12 deployment, §14 FastAPI).

A thin service over the batch pipeline and the real-time monitor, with the
deployment requirements of §12: PHI deidentification before results leave the
core, an append-only audit log, and version stamping. FastAPI is an optional
dependency (`pip install -e ".[api]"`); the core stays lean.
"""

from __future__ import annotations

from .deid import deidentify_result, hash_identifier
from .audit import AuditLogger

__all__ = ["deidentify_result", "hash_identifier", "AuditLogger", "create_app"]


def create_app(*args, **kwargs):
    """Lazy factory so importing the package doesn't require FastAPI."""
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)
