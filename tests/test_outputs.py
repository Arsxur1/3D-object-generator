"""Output-format tests (TZ §11): topomaps, causal graph, PDF."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from neurolens.contracts.causal import (  # noqa: E402
    CausalEdge,
    CausalGraph,
    CausalNode,
    NodeKind,
    PhysiologyLabel,
)
from neurolens.contracts.events import DetectionResult  # noqa: E402
from neurolens.contracts.report import (  # noqa: E402
    Bilingual,
    Confidence,
    LLMReport,
    OperatingMode,
)
from neurolens.contracts.signal import UnifiedSignal  # noqa: E402
from neurolens.outputs.causal_plot import plot_causal_graph  # noqa: E402
from neurolens.outputs.pdf_report import build_pdf_report  # noqa: E402
from neurolens.outputs.topomap import plot_topomap  # noqa: E402

import numpy as np  # noqa: E402

PNG_MAGIC = b"\x89PNG"


def test_topomap_renders_png(configs, tmp_path):
    values = {ch: float(i) for i, ch in enumerate(configs.electrode_coords)}
    fig, ax = plt.subplots()
    plot_topomap(ax, values, configs.electrode_coords, title="test", cmap="viridis")
    out = tmp_path / "topo.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists() and out.read_bytes()[:4] == PNG_MAGIC


def test_causal_graph_png(tmp_path):
    g = CausalGraph(
        nodes=[
            CausalNode(id="evt:diffuse_slowing:0", kind=NodeKind.EVENT,
                       label_ru="Диффузное замедление", label_uz="x",
                       physiology=PhysiologyLabel.PATHOLOGIC, confidence=0.7,
                       event_code="diffuse_slowing"),
            CausalNode(id="mech:enc", kind=NodeKind.MECHANISM,
                       label_ru="Энцефалопатия", label_uz="x",
                       physiology=PhysiologyLabel.PATHOLOGIC, confidence=0.7, critical=True),
        ],
        edges=[CausalEdge(source="evt:diffuse_slowing:0", target="mech:enc",
                          rule_id="diffuse_slowing_encephalopathy",
                          physiology=PhysiologyLabel.PATHOLOGIC, confidence=0.7)],
    )
    out = plot_causal_graph(g, tmp_path / "graph.png")
    assert out.exists() and out.read_bytes()[:4] == PNG_MAGIC


def test_causal_graph_drops_isolated_artifacts(tmp_path):
    # an isolated artifact node should not crash and should be dropped
    g = CausalGraph(nodes=[
        CausalNode(id="evt:spike:0", kind=NodeKind.EVENT, label_ru="спайк", label_uz="x",
                   physiology=PhysiologyLabel.ARTIFACT, confidence=0.5, event_code="spike"),
    ])
    out = plot_causal_graph(g, tmp_path / "g2.png")
    assert out.exists()


def test_pdf_report_valid(tmp_path):
    report = LLMReport(
        mode=OperatingMode.A_DECISION_SUPPORT,
        impression=Bilingual(ru="Заключение: тест кириллицы", uz="Xulosa: test"),
        overall_confidence=Confidence(value=0.6, label_ru="средняя", label_uz="o‘rtacha"),
        reasoning_trace=["шаг 1", "шаг 2"],
    )
    sig = UnifiedSignal(signal=np.zeros((2, 256), dtype=np.float32),
                        sampling_rate_hz=256, channel_names=["C3", "C4"])
    out = build_pdf_report(report, DetectionResult(), sig, "double_banana", {}, tmp_path / "r.pdf")
    data = out.read_bytes()
    assert data[:5] == b"%PDF-"
    assert len(data) > 1000
