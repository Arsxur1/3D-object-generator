"""Operating curves of the learned ictal detector (leave-one-subject-out).

Two steps, so the figure is reproducible without recomputing everything:

    python examples/ml_operating_curves.py compute   # needs cached PhysioNet features
    python examples/ml_operating_curves.py plot      # docs/figures/operating_curves.{json,png}

``compute`` builds out-of-fold probabilities on the 20 training patients (or
reuses ``--offline-probs`` / ``--rt-probs`` / ``--rt-streams`` pickles), sweeps
the post-processing grid and keeps the Pareto frontier (max sensitivity for a
given false-alarm rate). ``plot`` draws two small multiples (offline, real-time)
on identical axes. Colours: the reference data-viz palette, slots 1-3, validated
for all pairs (CVD ΔE 9.2, normal-vision ΔE 24.0); the lowest-contrast series is
direct-labelled and every series has its own marker.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

OUT_JSON = Path("docs/figures/operating_curves.json")
OUT_PNG = Path("docs/figures/operating_curves.png")
SUBJ = ("chb01,chb03,chb05,chb08,siena:PN00,siena:PN11,siena:PN12,"
        "chb02,chb11,chb13,chb14,chb16,chb17,chb18,chb19,chb20,chb22,siena:PN05,siena:PN16,siena:PN17")
GRID = {"threshold": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99],
        "min_epochs": [5, 10, 15, 20, 30, 45]}


def _frontier(res) -> list[list[float]]:
    pts = sorted({(round(t.score.fa_per_hour, 4), round(t.score.sensitivity, 4)) for t in res.trials})
    out, best = [], -1.0
    for fa, sens in pts:  # ascending FA; keep points that raise sensitivity
        if sens > best:
            out.append([fa, sens])
            best = sens
    return out


def compute(args) -> None:
    from neurolens.cli import _items
    from neurolens.evaluation.ml import Dataset, loso_probabilities, postproc_search
    from neurolens.evaluation.runner import default_config, evaluate_offline, prepare_records

    cfg = default_config()
    recs = prepare_records(_items("chbmit", SUBJ, None), cfg, "data/physionet/.features", workers=3)
    ds = Dataset.build(recs)
    gbm = pickle.load(open(args.offline_probs, "rb")) if args.offline_probs else loso_probabilities(ds, kind="gbm")
    lr = loso_probabilities(ds, kind="lr")
    thr = evaluate_offline(recs, cfg.thresholds).total
    rds = Dataset.build(pickle.load(open(args.rt_streams, "rb")))
    rgbm = pickle.load(open(args.rt_probs, "rb")) if args.rt_probs else loso_probabilities(rds, kind="gbm")
    frozen = json.loads(Path("docs/preregistration_increment10_frozen.json").read_text())
    data = {
        "source": "leave-one-subject-out on 20 training patients (142 records, 151.6 h, 100 seizures)",
        "offline": {
            "gbm": _frontier(postproc_search(ds, gbm, grid=GRID)),
            "lr": _frontier(postproc_search(ds, lr, grid=GRID)),
            "threshold_default": [round(thr.fa_per_hour, 4), round(thr.sensitivity, 4)],
        },
        "realtime": {
            "gbm": _frontier(postproc_search(rds, rgbm, grid=GRID, causal=True)),
            "threshold_default": [frozen["realtime"]["threshold_monitor_default_fa_per_hour"], 0.81],
        },
    }
    for mode in ("offline", "realtime"):
        data[mode]["points"] = {op[0]: [round(frozen[mode][op]["loso"]["fa_per_hour"], 4),
                                        round(frozen[mode][op]["loso"]["sensitivity"], 4)]
                                for op in ("A_replacement", "B_quiet")}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"wrote {OUT_JSON}")


def plot(_args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    d = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"   # reference palette slots 1-3
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True, facecolor=surface)
    panels = [("offline", "Анализ записи (оффлайн)"), ("realtime", "Монитор (real-time)")]
    for ax, (mode, title) in zip(axes, panels):
        m = d[mode]
        ax.set_facecolor(surface)
        # (key, label, colour, marker, label anchor x, label offset): labels sit on an
        # interior point of their own curve, where the curves are apart
        series = [("gbm", "Градиентный бустинг", blue, "o", 0.5, (-10, 8), "right")]
        if "lr" in m:
            series.append(("lr", "Логистическая регрессия", aqua, "s", 0.6, (8, -18), "left"))
        for key, label, color, marker, lx, off, ha in series:
            xs, ys = zip(*[(max(fa, 0.05), s) for fa, s in m[key]])
            ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=5, label=label, zorder=3,
                    markeredgecolor=surface, markeredgewidth=1)
            i = min(range(len(xs)), key=lambda k: abs(np.log(xs[k]) - np.log(lx)))
            ax.annotate(label, (xs[i], ys[i]), xytext=off, textcoords="offset points",
                        ha=ha, fontsize=8.5, color=ink2)
        fa, s = m["threshold_default"]
        ax.plot([fa], [s], marker="D", ms=9, color=orange, ls="none", zorder=4,
                markeredgecolor=surface, markeredgewidth=2, label="Пороговый детектор (дефолт)")
        ax.annotate("пороговый\n(дефолт)", (fa, s), xytext=(0, -26), textcoords="offset points",
                    ha="center", fontsize=8.5, color=ink2)
        for name, (pfa, ps) in m["points"].items():
            ax.plot([pfa], [ps], marker="o", ms=11, mfc="none", mec=ink, mew=1.5, zorder=5)
            ax.annotate(f"{name}: {ps:.2f} при {pfa:.1f} FA/ч", (pfa, ps), xytext=(-8, 10),
                        textcoords="offset points", ha="right", fontsize=8.5, color=ink)
        ax.set_xscale("log")
        ax.set_xlim(0.05, 20)
        ax.set_ylim(0.3, 1.0)
        ax.set_xticks([0.1, 0.3, 1, 3, 10])
        ax.set_xticklabels(["0.1", "0.3", "1", "3", "10"])
        ax.set_title(title, color=ink, fontsize=11, loc="left")
        ax.set_xlabel("Ложные тревоги в час (лог. шкала; 0 — на левой границе)", color=ink2)
        ax.grid(True, color=grid, lw=0.8)
        ax.tick_params(colors=ink2, length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel("Чувствительность (доля найденных приступов)", color=ink2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9,
               labelcolor=ink2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Чувствительность против ложных тревог (LOSO: 20 пациентов, 100 приступов)",
                 color=ink, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=160, facecolor=surface)
    print(f"wrote {OUT_PNG}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--offline-probs")
    c.add_argument("--rt-probs")
    c.add_argument("--rt-streams", required=True)
    c.set_defaults(func=compute)
    sub.add_parser("plot").set_defaults(func=plot)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
