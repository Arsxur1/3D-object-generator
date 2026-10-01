"""Visualization outputs (TZ §11.3, §11.4).

- Annotated montage curves (e.g. double banana) with event marks.
- DSA (spectrogram trend) + aEEG trend for long-record overview.

Topomaps and causal-graph visualization are deferred to the post-review
expansion (interfaces exist; see plan). Uses the Agg backend (headless).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..contracts.events import Event  # noqa: E402
from ..layer2_preprocess.montage_engine import MontagedSignal  # noqa: E402
from ..layer3_features.aeeg import AEEG  # noqa: E402
from ..layer3_features.dsa import DSA  # noqa: E402


def plot_montage_with_events(
    montaged: MontagedSignal,
    events: Iterable[Event],
    out_path: str | Path,
    window_s: Optional[tuple[float, float]] = None,
    max_seconds: float = 30.0,
) -> Path:
    fs = montaged.sampling_rate_hz
    n_samples = montaged.data.shape[1]
    total_s = n_samples / fs
    if window_s is None:
        window_s = (0.0, min(max_seconds, total_s))
    s0 = int(window_s[0] * fs)
    s1 = int(min(window_s[1], total_s) * fs)
    t = np.arange(s0, s1) / fs

    n = montaged.n_derivations
    # vertical offset per trace based on robust amplitude
    spread = np.percentile(np.abs(montaged.data[:, s0:s1]), 95) + 1e-6
    offset = 3.0 * spread

    fig, ax = plt.subplots(figsize=(12, max(4, 0.4 * n)))
    for i in range(n):
        ax.plot(t, montaged.data[i, s0:s1] + (n - i) * offset, lw=0.5, color="black")
    ax.set_yticks([(n - i) * offset for i in range(n)])
    ax.set_yticklabels(montaged.derivation_names, fontsize=7)
    ax.set_xlabel("Время, с / Vaqt, s")
    ax.set_title(f"Монтаж / Montaj: {montaged.montage_name}")

    colors = {"ictal": "red", "ied": "orange", "suppression": "purple",
              "background": "blue", "special": "green", "artifact": "gray"}
    for ev in events:
        if ev.t_end < window_s[0] or ev.t_start > window_s[1]:
            continue
        c = colors.get(ev.group, "blue")
        ax.axvspan(max(ev.t_start, window_s[0]), min(ev.t_end, window_s[1]),
                   color=c, alpha=0.12)
        ax.text(max(ev.t_start, window_s[0]), (n + 0.5) * offset, ev.code,
                color=c, fontsize=7, rotation=0)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_dsa_aeeg(dsa: DSA, aeeg: AEEG, out_path: str | Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    im = ax1.pcolormesh(dsa.times_s, dsa.freqs_hz, dsa.power_db, shading="auto", cmap="viridis")
    ax1.set_ylabel("Частота, Гц / Chastota")
    ax1.set_title(f"DSA (спектрограмма) — {dsa.channel}")
    fig.colorbar(im, ax=ax1, label="dB")

    ax2.plot(aeeg.times_s, aeeg.upper_uv, color="black", lw=0.8, label="upper")
    ax2.plot(aeeg.times_s, aeeg.lower_uv, color="gray", lw=0.8, label="lower")
    ax2.fill_between(aeeg.times_s, aeeg.lower_uv, aeeg.upper_uv, color="steelblue", alpha=0.3)
    ax2.set_ylabel("aEEG (сжатая шкала)")
    ax2.set_xlabel("Время, с / Vaqt, s")
    ax2.set_title(f"aEEG / CFM — {aeeg.channel}")
    ax2.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
