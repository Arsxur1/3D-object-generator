"""Artifact handling (TZ §5).

Two responsibilities:
  1. *Flag* artifacts (mark, don't silently remove): ECG (cardiac field),
     EMG (muscle), electrode "pop". The ECG hypothesis is passed to Layer 5 so
     it can distinguish true IEDs from cardiac transients (TZ §8.3).
  2. Provide a swappable removal step. ``ICAStep`` runs MNE's ICA when the
     optional ``mne`` backend is installed; otherwise it is a no-op that records
     a TODO flag (full ICLabel/ASR is v2). The step interface lets any removal
     method slot in behind ``ArtifactStep``.

Nothing here is destructive by default: detectors run on flagged-but-intact
signal, and the artifact hypotheses travel forward as evidence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks, welch

from ..contracts.signal import UnifiedSignal
from ..layer1_ingest.electrodes import is_eeg_channel


@dataclass
class ArtifactReport:
    """Structured artifact findings carried into Layer 4/5."""

    ecg_present: bool = False
    heart_rate_hz: float | None = None
    ecg_qrs_times_s: list[float] = field(default_factory=list)
    ecg_contaminated_channels: list[str] = field(default_factory=list)
    emg_channels: list[str] = field(default_factory=list)
    pop_channels: list[str] = field(default_factory=list)
    artifact_fraction: float = 0.0
    flags: list[str] = field(default_factory=list)


def _detect_qrs(ecg: np.ndarray, fs: float) -> tuple[float | None, list[float]]:
    """Very small QRS detector: peaks on the normalized |ECG|.

    Returns (heart_rate_hz, qrs_times_s). Adequate to establish periodicity for
    the ECG-vs-IED discrimination; not a clinical ECG analyzer.
    """
    x = np.abs(ecg - np.median(ecg))
    if np.std(x) < 1e-9:
        return None, []
    x = x / (np.std(x) + 1e-12)
    min_dist = int(0.33 * fs)  # <=180 bpm
    peaks, _ = find_peaks(x, height=2.5, distance=max(1, min_dist))
    if len(peaks) < 3:
        return None, []
    times = (peaks / fs).tolist()
    rr = np.diff(peaks) / fs
    hr = float(1.0 / np.median(rr)) if len(rr) else None
    return hr, times


def flag_artifacts(sig: UnifiedSignal, mains_hz: float = 50.0) -> ArtifactReport:
    """Detect and flag ECG/EMG/pop artifacts; append flags to signal quality."""
    data = sig.signal.astype(np.float64)
    fs = sig.sampling_rate_hz
    names = sig.channel_names
    report = ArtifactReport()

    # --- ECG artifact: find an ECG channel, detect QRS, correlate with EEG ---
    ecg_idx = next((i for i, n in enumerate(names) if n.upper() in ("ECG", "EKG")), None)
    if ecg_idx is not None:
        hr, qrs = _detect_qrs(data[ecg_idx], fs)
        if hr is not None:
            report.ecg_present = True
            report.heart_rate_hz = hr
            report.ecg_qrs_times_s = qrs
            ecg_sig = data[ecg_idx]
            contaminated = []
            for i, n in enumerate(names):
                if not is_eeg_channel(n):
                    continue
                if np.std(data[i]) < 1e-9:
                    continue
                r = float(np.corrcoef(data[i], ecg_sig)[0, 1])
                if abs(r) > 0.25:
                    contaminated.append(n)
            report.ecg_contaminated_channels = contaminated
            report.flags.append(
                f"ecg_artifact:hr={hr:.2f}Hz,channels={len(contaminated)}"
            )

    # --- EMG: excessive high-frequency (>30 Hz) power fraction ---
    emg = []
    for i, n in enumerate(names):
        if not is_eeg_channel(n) or fs <= 80:
            continue
        x = data[i]
        nper = int(min(len(x), fs * 2))
        if nper < 32:
            continue
        f, p = welch(x, fs=fs, nperseg=nper)
        total = np.trapezoid(p, f) + 1e-12
        hf = np.trapezoid(p[f > 30], f[f > 30]) / total
        if hf > 0.5:
            emg.append(n)
    report.emg_channels = emg
    if emg:
        report.flags.append(f"emg_artifact:channels={len(emg)}")

    # --- electrode pop: brief large step transients ---
    pops = []
    for i, n in enumerate(names):
        if not is_eeg_channel(n):
            continue
        d = np.abs(np.diff(data[i]))
        if d.size and np.max(d) > 100.0 and np.percentile(d, 99.9) > 50.0:
            pops.append(n)
    report.pop_channels = pops
    if pops:
        report.flags.append(f"electrode_pop:channels={len(pops)}")

    # crude artifact fraction: share of EEG channels flagged in any category
    eeg_names = [n for n in names if is_eeg_channel(n)]
    flagged = set(report.ecg_contaminated_channels) | set(emg) | set(pops)
    report.artifact_fraction = (
        len(flagged) / len(eeg_names) if eeg_names else 0.0
    )

    sig.quality.flags.extend(report.flags)
    return report


class ArtifactStep(ABC):
    """Swappable artifact-removal step (TZ §5: ICA/ASR/regression)."""

    name: str = "artifact_step"

    @abstractmethod
    def apply(self, sig: UnifiedSignal) -> UnifiedSignal:
        ...


class ICAStep(ArtifactStep):
    """ICA-based removal via the optional MNE backend.

    If ``mne`` is not installed, this is a no-op that records a TODO flag,
    keeping the skeleton runnable on the lean core. Full component
    auto-classification (ICLabel) is a v2 extension.
    """

    name = "ica"

    def __init__(self, n_components: int | None = None, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state

    def apply(self, sig: UnifiedSignal) -> UnifiedSignal:
        try:
            import mne  # noqa: F401
        except Exception:
            sig.quality.flags.append("ica:skipped(mne_not_installed;TODO_v2)")
            return sig
        return self._apply_mne(sig)

    def _apply_mne(self, sig: UnifiedSignal) -> UnifiedSignal:  # pragma: no cover
        import mne

        eeg_names = [n for n in sig.channel_names]
        info = mne.create_info(eeg_names, sig.sampling_rate_hz, ch_types="eeg")
        raw = mne.io.RawArray(sig.signal.astype(np.float64) * 1e-6, info, verbose="ERROR")
        ica = mne.preprocessing.ICA(
            n_components=self.n_components, random_state=self.random_state, verbose="ERROR"
        )
        ica.fit(raw)
        raw = ica.apply(raw, verbose="ERROR")
        cleaned = (raw.get_data() * 1e6).astype(np.float32)
        sig.quality.flags.append("ica:applied(mne)")
        return sig.with_signal(cleaned)
