"""Learned interictal epileptiform discharge (IED) detector (increment 15).

Windows of 0.5 s (hop 0.25 s) on the 19 standard 10-20 scalp electrodes of the
average-referenced, filtered signal. For every channel and window: peak-to-peak
amplitude, maximal slope, maximal sharpness (|2nd derivative|), line length, the
1-4 Hz amplitude of the *next* window (after-going slow wave) and the share of
20-70 Hz activity (muscle). Each is expressed relative to the channel's median
over the record (background) and relative to its median over the surrounding
±10 s (local prominence). Cross-channel aggregates (largest, second largest,
mean of the top three, median, number of channels above 3x / 5x background),
a frontal-polar dominance index (eye blinks) and the same features of the
neighbouring windows form the input of a gradient-boosted tree model
(:class:`~neurolens.layer4_detect.ml_ictal.TreeEnsembleModel`).

Detections are runs of windows whose probability is >= the threshold (merged
across a one-window gap); each run is one discharge event, confidence = its
maximal probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt

from ..contracts.events import Event, EventEvidence, Localization
from .background_abn import _region_of
from .base import Detector

WIN_S = 0.5
HOP_S = 0.25
LOCAL_WINDOWS = 41  # ±20 hops = ±5 s each side of a 0.5 s window -> ~10 s neighbourhood
STANDARD_19 = ("FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
               "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ")
_ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
METRICS = ("ptp", "slope", "sharp", "ll", "delta_next")
AGGS = ("max", "second", "top3", "median", "n_gt3", "n_gt5")


def standard_channels(names: list[str]) -> list[int]:
    """Indices of the 10-20 scalp electrodes (ears/mastoids and non-EEG excluded)."""
    idx = []
    for i, n in enumerate(names):
        up = _ALIASES.get(n.upper(), n.upper())
        if up in STANDARD_19:
            idx.append(i)
    return idx


def _windows(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """(C, N) -> (C, K, win) strided view."""
    return sliding_window_view(x, win, axis=1)[:, ::hop, :]


def _bandpass(x: np.ndarray, fs: float, lo: float, hi: Optional[float]) -> np.ndarray:
    nyq = fs / 2.0
    if hi is None or hi >= nyq * 0.95:
        sos = butter(4, lo / nyq, btype="highpass", output="sos")
    else:
        sos = butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x, axis=1)


def channel_metrics(x: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    """Per-channel, per-window raw metrics, each (C, K)."""
    win, hop = int(round(WIN_S * fs)), int(round(HOP_S * fs))
    d1 = np.abs(np.diff(x, axis=1, prepend=x[:, :1])) * fs
    d2 = np.abs(np.diff(x, n=2, axis=1, prepend=x[:, :1], append=x[:, -1:])) * fs * fs
    w = _windows(x, win, hop)
    out = {
        "ptp": w.max(axis=2) - w.min(axis=2),
        "slope": _windows(d1, win, hop).max(axis=2),
        "sharp": _windows(d2, win, hop).max(axis=2),
        "ll": _windows(d1, win, hop).sum(axis=2) / fs,
    }
    dw = _windows(_bandpass(x, fs, 1.0, 4.0), win, hop)
    dptp = dw.max(axis=2) - dw.min(axis=2)
    nxt = np.empty_like(dptp)
    nxt[:, :-2] = dptp[:, 2:]  # window starting 0.5 s later
    nxt[:, -2:] = dptp[:, -1:]
    out["delta_next"] = nxt
    hw = _windows(_bandpass(x, fs, 20.0, 70.0), win, hop)
    out["hf_ratio"] = (hw.max(axis=2) - hw.min(axis=2)) / (out["ptp"] + 1e-6)
    return out


def _aggregate(m: np.ndarray) -> list[np.ndarray]:
    """(C, K) relative metric -> aggregates across channels, each (K,)."""
    s = np.sort(m, axis=0)[::-1]
    return [s[0], s[1] if len(s) > 1 else s[0], s[: min(3, len(s))].mean(axis=0),
            np.median(m, axis=0), (m > 3).sum(axis=0).astype(float), (m > 5).sum(axis=0).astype(float)]


def base_feature_names() -> list[str]:
    names = []
    for scope in ("bg", "loc"):
        for m in METRICS:
            names += [f"{m}_{scope}_{a}" for a in AGGS]
    return names + ["hf_ratio_at_max_slope", "hf_ratio_median", "fp_dominance",
                    "log_ptp_abs_max", "slope_ptp_ratio_at_max"]


def feature_names() -> list[str]:
    b = base_feature_names()
    return b + [f"prev_{n}" for n in b] + [f"next_{n}" for n in b]


def window_features(x: np.ndarray, fs: float, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix (K, F) and window start times (K,) for a (C, N) signal in µV."""
    idx = standard_channels(names)
    if len(idx) < 4:
        return np.zeros((0, len(feature_names()))), np.zeros(0)
    x = np.asarray(x[idx], dtype=np.float64)
    ch = [_ALIASES.get(names[i].upper(), names[i].upper()) for i in idx]
    raw = channel_metrics(x, fs)
    K = raw["ptp"].shape[1]
    cols: list[np.ndarray] = []
    for scope in ("bg", "loc"):
        for m in METRICS:
            v = raw[m]
            if scope == "bg":
                ref = np.median(v, axis=1, keepdims=True)
            else:
                ref = median_filter(v, size=(1, LOCAL_WINDOWS), mode="nearest")
            cols += _aggregate(v / (ref + 1e-6))
    slope_bg = raw["slope"] / (np.median(raw["slope"], axis=1, keepdims=True) + 1e-6)
    ptp_bg = raw["ptp"] / (np.median(raw["ptp"], axis=1, keepdims=True) + 1e-6)
    arg = np.argmax(slope_bg, axis=0)
    k = np.arange(K)
    cols.append(raw["hf_ratio"][arg, k])
    cols.append(np.median(raw["hf_ratio"], axis=0))
    fp = [i for i, c in enumerate(ch) if c in ("FP1", "FP2")]
    rest = [i for i in range(len(ch)) if i not in fp]
    if fp and rest:
        cols.append(ptp_bg[fp].max(axis=0) / (ptp_bg[rest].max(axis=0) + 1e-6))
    else:
        cols.append(np.ones(K))
    cols.append(np.log10(raw["ptp"].max(axis=0) + 1.0))
    cols.append(slope_bg[arg, k] / (ptp_bg[arg, k] + 1e-6))
    B = np.column_stack(cols)
    prev = np.vstack([B[:1], B[:-1]])
    nxt = np.vstack([B[1:], B[-1:]])
    X = np.hstack([B, prev, nxt])
    t0 = k * int(round(HOP_S * fs)) / fs
    return X, t0


def window_labels(t0: np.ndarray, positives: list[tuple[float, float]],
                  ignore_margin_s: float = HOP_S) -> np.ndarray:
    """1 = window overlaps an annotated discharge; 0 = clear of it; -1 = boundary
    window (within ``ignore_margin_s`` of a discharge without overlapping it)."""
    y = np.zeros(t0.size, dtype=np.int8)
    t1 = t0 + WIN_S
    for a, b in positives:
        hit = (t0 < b) & (t1 > a)
        near = (t0 < b + ignore_margin_s) & (t1 > a - ignore_margin_s) & ~hit
        y[near & (y == 0)] = -1
        y[hit] = 1
    return y


def probability_events(t0: np.ndarray, p: np.ndarray, threshold: float,
                       gap_windows: int = 1) -> list[tuple[float, float, float]]:
    """Runs of windows with p >= threshold -> (start_s, end_s, max_p)."""
    on = np.where(p >= threshold)[0]
    out: list[tuple[float, float, float]] = []
    if on.size == 0:
        return out
    start = prev = on[0]
    for i in on[1:]:
        if i - prev > gap_windows + 1:
            out.append((float(t0[start]), float(t0[prev] + WIN_S), float(p[start:prev + 1].max())))
            start = i
        prev = i
    out.append((float(t0[start]), float(t0[prev] + WIN_S), float(p[start:prev + 1].max())))
    return out


def annotate_candidates(model, sig, events, threshold: float, half_window_s: float = 0.5) -> int:
    """Attach the learned IED probability to heuristic spike candidates (in place).

    The probability is the maximum over windows overlapping the candidate +- half_window_s.
    Returns the number of candidates rated >= ``threshold``."""
    spikes = [e for e in events if e.code == "spike"]
    if not spikes:
        return 0
    X, t0 = window_features(sig.signal, sig.sampling_rate_hz, sig.channel_names)
    if X.shape[0] == 0:
        return 0
    p = model.predict_proba(X)
    high = 0
    for e in spikes:
        lo, hi = e.t_start - half_window_s, e.t_end + half_window_s
        near = (t0 < hi) & (t0 + WIN_S > lo)
        prob = float(p[near].max()) if near.any() else 0.0
        e.evidence.append(EventEvidence(feature="ied_probability_learned", value=round(prob, 3),
                                        reference=f">= {threshold}"))
        e.metadata["ied_probability_learned"] = round(prob, 3)
        high += prob >= threshold
    return int(high)


@dataclass
class IEDModelConfig:
    model_path: str
    threshold: float = 0.5
    max_events: int = 200


class MLSpikeDetector(Detector):
    """Pipeline detector using a frozen IED model (replaces the heuristic SpikeDetector
    when enabled in ``configs/ied.yaml``)."""

    code = "spike"
    group = "ied"

    def __init__(self, model, threshold: float, max_events: int = 200, name: str = "ied_gbm"):
        self.model, self.threshold, self.max_events, self.name = model, threshold, max_events, name

    @classmethod
    def from_path(cls, path: str | Path, threshold: float, max_events: int = 200):
        from .ml_ictal import load_model

        return cls(load_model(path), threshold, max_events, Path(path).stem)

    def detect(self, sig, features, thresholds, artifacts=None) -> list[Event]:
        fs = sig.sampling_rate_hz
        names = sig.channel_names
        X, t0 = window_features(sig.signal, fs, names)
        if X.shape[0] == 0:
            return []
        p = self.model.predict_proba(X)
        idx = standard_channels(names)
        x = np.asarray(sig.signal[idx], dtype=np.float64)
        ch = [names[i] for i in idx]
        events: list[Event] = []
        for a, b, conf in probability_events(t0, p, self.threshold):
            lo, hi = int(a * fs), max(int(b * fs), int(a * fs) + 1)
            seg = x[:, lo:hi]
            ptp = seg.max(axis=1) - seg.min(axis=1)
            bg = np.median(np.abs(x - np.median(x, axis=1, keepdims=True)), axis=1) * 6 + 1e-6
            rel = ptp / bg
            top = [ch[i] for i in np.argsort(rel)[::-1] if rel[i] >= 0.6 * rel.max()][:6]
            ru, uz = self.labels(self.code)
            events.append(Event(
                code=self.code, label_ru=ru, label_uz=uz, group=self.group,
                localization=Localization(channels=top, region=_region_of(top[0]) if top else None),
                t_start=round(a, 3), t_end=round(b, 3), confidence=round(float(conf), 3),
                is_artifact_hypothesis=False,
                evidence=[EventEvidence(feature="ied_probability", value=round(float(conf), 3),
                                        reference=f">= {self.threshold}"),
                          EventEvidence(feature="amplitude_uv", value=round(float(ptp.max()), 1),
                                        unit="uV", channels=top[:1])],
                metadata={"needs_artifact_check": True, "detector": self.name},
            ))
        events.sort(key=lambda e: e.confidence, reverse=True)
        return events[: self.max_events]
