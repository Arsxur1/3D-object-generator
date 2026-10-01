"""Real-time cEEG monitor (TZ §2 sub-mode, §10, §16).

Consumes a stream, runs a lightweight per-window analysis (filter -> features ->
seizure/suppression/IIC detectors), and raises alarms via the AlarmManager.
Accumulates seizure burden and per-window latency across the session, and
attaches an EEG fragment (montage PNG) to every alarm for verification.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np

from ..contracts.alarms import AlarmSeverity, AlarmType, MonitorSummary
from ..contracts.events import DetectionResult, SeizureBurden
from ..contracts.report import Bilingual
from ..contracts.signal import UnifiedSignal
from ..layer2_preprocess.filters import apply_filters
from ..layer2_preprocess.montage_engine import MontageEngine
from ..layer2_preprocess.reref import rereference
from ..layer3_features.feature_set import compute_features
from ..layer4_detect.base import Detector
from ..layer4_detect.ictal import IctalRhythmDetector
from ..layer4_detect.periodic import PeriodicPatternDetector
from ..layer4_detect.registry import run_detectors
from ..layer4_detect.suppression_events import BurstSuppressionDetector
from ..outputs.plots import plot_montage_with_events
from ..pipeline.config_loader import ConfigBundle, load_configs
from .alarms import AlarmCandidate, AlarmManager
from .buffer import WindowBuffer
from .stream import StreamSource

STATUS_DURATION_S = 30.0

_MSG = {
    AlarmType.SEIZURE: Bilingual(
        ru="Зарегистрирована судорожная активность — требуется верификация врачом.",
        uz="Tutqanoq faolligi qayd etildi — shifokor tekshiruvi zarur.",
    ),
    AlarmType.STATUS_EPILEPTICUS: Bilingual(
        ru="Подозрение на эпилептический статус (накопленная судорожная нагрузка).",
        uz="Epileptik status shubhasi (to‘plangan tutqanoq yuki).",
    ),
    AlarmType.IIC: Bilingual(
        ru="Периодический паттерн на иктально-интериктальном континууме (IIC).",
        uz="Iktal-interiktal kontinuumdagi davriy pattern (IIC).",
    ),
    AlarmType.BURST_SUPPRESSION: Bilingual(
        ru="Паттерн «вспышка-подавление» — тяжёлое повреждение против седации/гипотермии.",
        uz="“Portlash-bostirish” patterni — og‘ir shikast yoki sedatsiya/gipotermiya.",
    ),
}


class RealtimeMonitor:
    def __init__(
        self,
        config: ConfigBundle | None = None,
        montage_name: str = "double_banana",
        detectors: list[Detector] | None = None,
        out_dir: str | Path | None = None,
    ):
        self.cfg = config or load_configs()
        self.montage_name = montage_name
        self.detectors = detectors or [
            IctalRhythmDetector(), BurstSuppressionDetector(), PeriodicPatternDetector(),
        ]
        self.out_dir = Path(out_dir) if out_dir else None
        self._seizure_intervals: list[tuple[float, float]] = []
        # rolling per-channel amplitude baseline for streaming seizure detection
        self._baseline: Optional[np.ndarray] = None
        self._baseline_alpha = 0.05  # slow EMA: a seizure does not quickly inflate it
        self._ictal = next((d for d in self.detectors if isinstance(d, IctalRhythmDetector)), None)

    def run(self, source: StreamSource) -> MonitorSummary:
        rt = self.cfg.realtime
        fs = source.sampling_rate_hz
        buf = WindowBuffer(len(source.channel_names), fs, rt.window_s, rt.step_s)
        mgr = AlarmManager(rt)
        alarms = []
        latencies: list[float] = []
        n_windows = 0
        last_t1 = 0.0

        for chunk in source.chunks():
            for window_data, start_sample in buf.push(chunk.data):
                t0 = start_sample / fs
                t1 = t0 + rt.window_s
                last_t1 = t1

                t_perf = perf_counter()
                det, analysis = self._analyze(window_data, source)
                latency_ms = (perf_counter() - t_perf) * 1000.0
                latencies.append(latency_ms)

                candidates = self._candidates(det, t0)
                fired = mgr.process(candidates, n_windows, t0, t1, latency_ms)
                for al in fired:
                    self._attach_fragment(al, analysis, det, rt.window_s)
                alarms.extend(fired)
                n_windows += 1

        return self._summary(source, alarms, mgr, latencies, n_windows, last_t1)

    # -- per-window analysis ---------------------------------------------
    def _analyze(self, window_data: np.ndarray, source: StreamSource):
        sig = UnifiedSignal(
            signal=window_data.astype(np.float32),
            sampling_rate_hz=source.sampling_rate_hz,
            channel_names=list(source.channel_names),
            reference=source.reference,
        )
        filt = apply_filters(sig, self.cfg.filters)
        analysis = rereference(filt, "average")
        feats = compute_features(analysis, self.cfg.filters)

        # streaming baseline: use history (pre-update) for this window's seizure
        # detection, then slowly update it toward the current window.
        win_med = (
            np.median(feats.epoch_rms, axis=0)
            if feats.epoch_rms.size else np.zeros(len(feats.eeg_channels))
        )
        if self._baseline is None or self._baseline.shape != win_med.shape:
            self._baseline = win_med.copy()
        if self._ictal is not None:
            self._ictal.external_baseline = self._baseline.copy()

        det = run_detectors(analysis, feats, self.cfg.thresholds, detectors=self.detectors)

        self._baseline = (1 - self._baseline_alpha) * self._baseline + self._baseline_alpha * win_med
        return det, analysis

    # -- candidate generation --------------------------------------------
    def _candidates(self, det: DetectionResult, t0: float) -> list[AlarmCandidate]:
        cands: list[AlarmCandidate] = []

        ictal = det.by_group("ictal")
        for e in ictal:
            self._seizure_intervals.append((t0 + e.t_start, t0 + e.t_end))
            cands.append(AlarmCandidate(
                type=AlarmType.SEIZURE, severity=AlarmSeverity.CRITICAL,
                message=_MSG[AlarmType.SEIZURE], confidence=e.confidence,
                channels=e.localization.channels, evidence=[e.code], is_critical=True,
            ))
        self._seizure_intervals = _merge(self._seizure_intervals)
        if _total(self._seizure_intervals) >= STATUS_DURATION_S:
            cands.append(AlarmCandidate(
                type=AlarmType.STATUS_EPILEPTICUS, severity=AlarmSeverity.CRITICAL,
                message=_MSG[AlarmType.STATUS_EPILEPTICUS], confidence=0.9,
                evidence=["seizure_burden"], is_critical=True,
            ))

        for e in det.by_group("suppression"):
            cands.append(AlarmCandidate(
                type=AlarmType.BURST_SUPPRESSION, severity=AlarmSeverity.CRITICAL,
                message=_MSG[AlarmType.BURST_SUPPRESSION], confidence=e.confidence,
                channels=e.localization.channels, evidence=[e.code], is_critical=True,
            ))

        for e in det.by_group("periodic"):
            if e.acns and e.acns.iic:
                cands.append(AlarmCandidate(
                    type=AlarmType.IIC, severity=AlarmSeverity.WARNING,
                    message=_MSG[AlarmType.IIC], confidence=e.confidence,
                    channels=e.localization.channels, evidence=[e.code], is_critical=True,
                ))
        return cands

    # -- fragment for verification (TZ §10) ------------------------------
    def _attach_fragment(self, alarm, analysis, det: DetectionResult, window_s: float) -> None:
        if self.out_dir is None:
            return
        try:
            montaged = MontageEngine(analysis).apply(self.cfg.montage(self.montage_name))
            frag_dir = self.out_dir / "fragments"
            path = frag_dir / f"{alarm.id}_{alarm.type.value}.png"
            plot_montage_with_events(montaged, det.events, path, window_s=(0.0, window_s))
            alarm.fragment_path = str(path)
        except Exception:  # a fragment failure must not drop the alarm
            alarm.fragment_path = None

    # -- summary ----------------------------------------------------------
    def _summary(self, source, alarms, mgr, latencies, n_windows, duration_s) -> MonitorSummary:
        rt = self.cfg.realtime
        hours = duration_s / 3600.0 if duration_s > 0 else 0.0
        per_hour = (mgr.n_raised / hours) if hours > 0 else 0.0
        lat = np.asarray(latencies) if latencies else np.asarray([0.0])
        merged = _merge(self._seizure_intervals)
        total_sz = _total(merged)
        burden = SeizureBurden(
            n_seizures=len(merged),
            total_seizure_time_s=round(total_sz, 1),
            recording_duration_s=round(duration_s, 1),
            seizure_fraction=round(min(1.0, total_sz / duration_s) if duration_s else 0.0, 4),
            seizures_per_hour=round(len(merged) / hours, 2) if hours > 0 else 0.0,
            longest_seizure_s=round(max((b - a for a, b in merged), default=0.0), 1),
            status_epilepticus_suspected=total_sz >= STATUS_DURATION_S,
        )
        p95 = float(np.percentile(lat, 95))
        return MonitorSummary(
            source=getattr(source, "source_name", "stream"),
            montage=self.montage_name,
            duration_s=round(duration_s, 1),
            n_windows=n_windows,
            window_s=rt.window_s,
            step_s=rt.step_s,
            alarms=alarms,
            n_alarms_raised=mgr.n_raised,
            n_alarms_suppressed=mgr.n_suppressed,
            alarms_per_hour=round(per_hour, 2),
            false_alarm_budget_per_hour=rt.max_false_alarms_per_hour,
            within_budget=per_hour <= rt.max_false_alarms_per_hour,
            seizure_burden=burden,
            latency_ms_mean=round(float(np.mean(lat)), 1),
            latency_ms_p95=round(p95, 1),
            realtime_feasible=p95 < rt.step_s * 1000.0,
        )


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    s = sorted(intervals)
    out = [s[0]]
    for a, b in s[1:]:
        la, lb = out[-1]
        if a <= lb:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out


def _total(intervals: list[tuple[float, float]]) -> float:
    return sum(b - a for a, b in intervals)
