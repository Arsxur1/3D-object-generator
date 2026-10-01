"""Causal-graph visualization (TZ §11.6).

Draws the Layer-5 causal graph: nodes colored by physiology label (physiologic /
pathologic / artifact / uncertain), critical nodes outlined, edges labeled with
the rule that produced them. A layered left-to-right layout (events/context ->
mechanisms) keeps the cause->effect direction readable.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

from ..contracts.causal import CausalGraph  # noqa: E402
from ..layer5_reasoning.causal_graph import to_networkx  # noqa: E402

_COLORS = {
    "physiologic": "#2ca02c",
    "pathologic": "#d62728",
    "artifact": "#7f7f7f",
    "uncertain": "#ff7f0e",
}


def _layer(graph: CausalGraph, node_id: str) -> int:
    node = graph.node(node_id)
    if node is None:
        return 0
    return {"event": 0, "context": 0, "mechanism": 1}.get(node.kind.value, 1)


def plot_causal_graph(graph: CausalGraph, out_path: str | Path, title: str = "Каузальный граф") -> Path:
    g = to_networkx(graph)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Drop isolated artifact nodes (e.g. individual spikes resolved as ECG
    # artifact) — their story is told by the aggregated mechanism node.
    isolated_artifacts = [
        n for n in g.nodes
        if g.degree(n) == 0 and g.nodes[n].get("physiology") == "artifact"
    ]
    g.remove_nodes_from(isolated_artifacts)

    if g.number_of_nodes() == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Граф пуст", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        return out_path

    # layered layout: column by layer, spread vertically within a column
    layers: dict[int, list[str]] = {}
    for n in g.nodes:
        layers.setdefault(_layer(graph, n), []).append(n)
    pos = {}
    for col, nodes in layers.items():
        k = len(nodes)
        for i, n in enumerate(sorted(nodes)):
            pos[n] = (col * 2.2, (k - 1) / 2.0 - i)

    node_colors = [_COLORS.get(g.nodes[n]["physiology"], "#cccccc") for n in g.nodes]
    edge_colors = [_COLORS.get(g.edges[e]["physiology"], "#999999") for e in g.edges]
    linewidths = [2.6 if g.nodes[n].get("critical") else 0.8 for n in g.nodes]

    height = max(4.0, 0.9 * max(len(v) for v in layers.values()))
    fig, ax = plt.subplots(figsize=(12, height))
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=edge_colors, width=1.4,
                           arrows=True, arrowsize=16, min_target_margin=22, node_size=2600)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=node_colors, node_size=2600,
                           edgecolors="black", linewidths=linewidths)
    labels = {n: textwrap.fill(g.nodes[n]["label_ru"], 22)[:80] for n in g.nodes}
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=6.5)
    edge_labels = {e: (g.edges[e].get("rule_id") or "") for e in g.edges}
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, ax=ax, font_size=5.5,
                                 rotate=False, bbox=dict(fc="white", ec="none", alpha=0.6))

    # legend
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                          markersize=10, label=lbl)
               for lbl, c in [("физиологично", _COLORS["physiologic"]),
                              ("патологично", _COLORS["pathologic"]),
                              ("артефакт", _COLORS["artifact"]),
                              ("неопределённо", _COLORS["uncertain"])]]
    ax.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, frameon=False)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
