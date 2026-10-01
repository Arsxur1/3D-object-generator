"""ECG-artifact event detector (TZ §7, §8.3).

Turns the Layer-2 ECG :class:`ArtifactReport` into an event so Layer 5 can use
it to resolve pseudo-epileptiform transients (ECG artifact mimicking sharp
transients). Marked as an artifact hypothesis.
"""

from __future__ import annotations

from ..contracts.events import Event, EventEvidence, Localization
from .base import Detector


class EcgArtifactDetector(Detector):
    code = "ecg_artifact"
    group = "artifact"

    def detect(self, sig, features, thresholds, artifacts=None) -> list[Event]:
        if artifacts is None or not artifacts.ecg_present:
            return []
        ru, uz = self.labels(self.code)
        hr = artifacts.heart_rate_hz or 0.0
        chans = artifacts.ecg_contaminated_channels
        conf = 0.6 if not chans else min(0.9, 0.6 + 0.05 * len(chans))
        return [
            Event(
                code=self.code,
                label_ru=ru,
                label_uz=uz,
                group=self.group,
                localization=Localization(channels=chans, region="widespread"),
                t_start=0.0,
                t_end=sig.duration_s or 0.0,
                confidence=float(conf),
                is_artifact_hypothesis=True,
                evidence=[
                    EventEvidence(feature="heart_rate_hz", value=round(hr, 2), unit="Hz",
                                  reference="QRS periodicity"),
                    EventEvidence(feature="ecg_contaminated_channels",
                                  value=float(len(chans)),
                                  note=",".join(chans[:8])),
                ],
                metadata={"qrs_count": len(artifacts.ecg_qrs_times_s)},
            )
        ]
