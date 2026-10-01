"""Ictal (seizure) rhythmic-pattern detector (TZ §7.3).

Classic-signal heuristic: a seizure appears as a sustained run of epochs with
high spectral concentration (rhythmicity) AND amplitude elevated above the
channel's own baseline, with evolution of the dominant frequency. This is an
MVP detector; ML seizure detectors (braindecode) come in v2. Feeds seizure
burden accounting for cEEG.
"""

from __future__ import annotations

import numpy as np

from ..contracts.events import Event, EventEvidence, Localization
from ..layer4_detect.background_abn import _region_of
from .base import Detector


class IctalRhythmDetector(Detector):
    code = "ictal_rhythm"
    group = "ictal"

    def detect(self, sig, features, thresholds, artifacts=None) -> list[Event]:
        times = features.epoch_times
        if times.size == 0:
            return []
        dt_epoch = float(np.median(np.diff(times))) if times.size > 1 else features.epoch_s
        min_epochs = max(2, int(round(thresholds.ictal_min_duration_s / max(dt_epoch, 1e-6))))

        events: list[Event] = []
        rms = features.epoch_rms
        conc = features.epoch_band_conc
        domf = features.epoch_domfreq

        # Optional external (rolling) baseline for streaming: when a seizure fills
        # the analysis window there is no quiet reference, so the monitor supplies
        # a baseline from stream history aligned to features.eeg_channels.
        ext_baseline = getattr(self, "external_baseline", None)

        for ci, chan in enumerate(features.eeg_channels):
            if ext_baseline is not None and ci < len(ext_baseline):
                baseline = float(ext_baseline[ci]) + 1e-9
            else:
                baseline = float(np.median(rms[:, ci])) + 1e-9
            active = (conc[:, ci] >= thresholds.ictal_rhythmicity) & (
                rms[:, ci] >= thresholds.ictal_amplitude_factor * baseline
            )
            for a, b in _runs(active):
                if (b - a) < min_epochs:
                    continue
                # Distinguish ictal from rhythmic delta (RDA/IIC): a seizure is
                # typically >3 Hz OR shows clear frequency evolution. A pure,
                # non-evolving <=3 Hz rhythm is rhythmic delta, not ictal.
                seg_f = domf[a:b, ci]
                mean_f = float(np.mean(seg_f))
                evolution = abs(float(seg_f[-1]) - float(seg_f[0]))
                if mean_f <= 3.0 and evolution < 1.5:
                    continue
                events.append(self._make_event(features, ci, chan, a, b, baseline))

        if not events:
            return []
        # Cluster per-channel runs that overlap in time: each cluster is one
        # seizure (long recordings contain several). Per cluster keep the
        # highest-confidence run, spanning the cluster, localized to all its
        # channels.
        out: list[Event] = []
        min_ch = getattr(thresholds, "ictal_min_channels", 1)
        for cluster in _time_clusters(events):
            cluster.sort(key=lambda e: (e.confidence, e.duration_s), reverse=True)
            best = cluster[0]
            chans = sorted({e.localization.channels[0] for e in cluster if e.localization.channels})
            if len(chans) < min_ch:
                continue
            # spatial recruitment is evidence: a multi-channel discharge is more
            # likely ictal than a focal rhythmic run of the same morphology
            best.confidence = float(np.clip(best.confidence + 0.01 * (len(chans) - 1), 0.5, 0.99))
            best.t_start = min(e.t_start for e in cluster)
            best.t_end = max(e.t_end for e in cluster)
            best.localization.channels = chans
            best.localization.lateralization = _lateralization(chans)
            best.localization.region = _region_of(chans[0]) if chans else None
            best.metadata["seizure_burden_s"] = round(best.duration_s, 1)
            best.metadata["n_channels_involved"] = len(chans)
            out.append(best)
        out.sort(key=lambda e: (e.confidence, e.duration_s), reverse=True)
        return out

    def _make_event(self, features, ci, chan, a, b, baseline) -> Event:
        times = features.epoch_times
        t0, t1 = float(times[a]), float(times[b - 1])
        seg_conc = float(np.mean(features.epoch_band_conc[a:b, ci]))
        seg_amp = float(np.mean(features.epoch_rms[a:b, ci]))
        f_start = float(features.epoch_domfreq[a, ci])
        f_end = float(features.epoch_domfreq[b - 1, ci])
        evolution = abs(f_end - f_start)
        ru, uz = self.labels(self.code)
        conf = float(np.clip(0.5 + 0.4 * seg_conc + 0.05 * evolution, 0.5, 0.95))
        return Event(
            code=self.code,
            label_ru=ru,
            label_uz=uz,
            group=self.group,
            localization=Localization(channels=[chan]),
            t_start=t0,
            t_end=t1,
            confidence=conf,
            evidence=[
                EventEvidence(feature="rhythmicity", value=round(seg_conc, 3),
                              reference="sustained narrow-band rhythm", channels=[chan]),
                EventEvidence(feature="amplitude_rms_uv", value=round(seg_amp, 1), unit="uV",
                              reference=f"> baseline {baseline:.1f}uV", channels=[chan]),
                EventEvidence(feature="freq_evolution_hz",
                              value=round(evolution, 2), unit="Hz",
                              note=f"{f_start:.1f}Hz -> {f_end:.1f}Hz"),
            ],
        )


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.size == 0:
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    stops = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        stops = stops + [mask.size]
    return list(zip(starts, stops))


def _time_clusters(events: list[Event]) -> list[list[Event]]:
    s = sorted(events, key=lambda e: e.t_start)
    clusters: list[list[Event]] = [[s[0]]]
    end = s[0].t_end
    for e in s[1:]:
        if e.t_start <= end:
            clusters[-1].append(e)
            end = max(end, e.t_end)
        else:
            clusters.append([e])
            end = e.t_end
    return clusters


def _overlaps(e1: Event, e2: Event) -> bool:
    return not (e1.t_end < e2.t_start or e2.t_end < e1.t_start)


def _lateralization(channels: list[str]) -> str:
    left = sum(1 for c in channels if c and c[-1].isdigit() and int(c[-1]) % 2 == 1)
    right = sum(1 for c in channels if c and c[-1].isdigit() and int(c[-1]) % 2 == 0)
    if left and not right:
        return "left"
    if right and not left:
        return "right"
    return "bilateral"
