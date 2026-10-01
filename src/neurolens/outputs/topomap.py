"""Scalp topographic maps (TZ §11.5).

Renders the distribution of a per-channel scalar (band power, asymmetry, etc.)
as an interpolated scalp map with a head outline and electrode markers. Uses the
2-D 10-20 coordinates from ``configs/electrodes/coords_10_20.yaml`` (the optional
``mne`` backend provides source-grade geometry in v2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402

from ..layer3_features.feature_set import FeatureSet  # noqa: E402
from ..layer3_features.spectral import BANDS  # noqa: E402

_HEAD_R = 1.0  # head radius in the normalized coordinate space


def _draw_head(ax) -> None:
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(_HEAD_R * np.cos(theta), _HEAD_R * np.sin(theta), color="black", lw=1.5)
    # nose
    ax.plot([-0.12, 0, 0.12], [_HEAD_R * 0.99, _HEAD_R * 1.15, _HEAD_R * 0.99], color="black", lw=1.5)
    # ears
    for sx in (-1, 1):
        ax.plot([sx * _HEAD_R, sx * (_HEAD_R + 0.08), sx * _HEAD_R],
                [0.1, 0.0, -0.1], color="black", lw=1.5)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_topomap(
    ax,
    values: Mapping[str, float],
    coords: Mapping[str, tuple[float, float]],
    title: str = "",
    cmap: str = "RdBu_r",
    symmetric: bool = False,
) -> None:
    """Draw one interpolated topomap onto ``ax``."""
    pts, vals = [], []
    for ch, v in values.items():
        if ch in coords and abs(coords[ch][0]) <= _HEAD_R + 1e-6:
            pts.append(coords[ch])
            vals.append(v)
    if len(pts) < 3:
        ax.text(0.5, 0.5, "мало электродов", ha="center")
        _draw_head(ax)
        return
    pts = np.asarray(pts)
    vals = np.asarray(vals)

    grid_x, grid_y = np.mgrid[-_HEAD_R:_HEAD_R:120j, -_HEAD_R:_HEAD_R:120j]
    zi = griddata(pts, vals, (grid_x, grid_y), method="cubic")
    zi_lin = griddata(pts, vals, (grid_x, grid_y), method="linear")
    zi = np.where(np.isnan(zi), zi_lin, zi)
    mask = grid_x**2 + grid_y**2 > _HEAD_R**2
    zi = np.ma.array(zi, mask=mask)

    if symmetric:
        m = np.nanmax(np.abs(vals)) or 1.0
        vmin, vmax = -m, m
    else:
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))

    ax.contourf(grid_x, grid_y, zi, levels=14, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.scatter(pts[:, 0], pts[:, 1], c="black", s=8, zorder=3)
    _draw_head(ax)
    ax.set_title(title, fontsize=9)


def plot_band_topomaps(
    features: FeatureSet,
    coords: Mapping[str, tuple[float, float]],
    out_path: str | Path,
) -> Path:
    """Multi-panel: relative band-power maps + an anterior/posterior view."""
    bands = list(BANDS.keys())
    fig, axes = plt.subplots(1, len(bands), figsize=(3 * len(bands), 3.2))
    for ax, band in zip(np.atleast_1d(axes), bands):
        values = {
            ch: feats.get(f"rel_{band}", 0.0)
            for ch, feats in features.channel_summary.items()
        }
        plot_topomap(ax, values, coords, title=f"отн. мощность {band}", cmap="viridis")
    fig.suptitle("Топокарты относительной мощности по диапазонам", fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
