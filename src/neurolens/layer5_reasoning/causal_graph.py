"""Causal-graph helpers (TZ §8). networkx view for analysis and visualization."""

from __future__ import annotations

import networkx as nx

from ..contracts.causal import CausalGraph


def to_networkx(graph: CausalGraph) -> nx.DiGraph:
    """Build a networkx DiGraph from the serializable CausalGraph contract."""
    g = nx.DiGraph()
    for n in graph.nodes:
        g.add_node(
            n.id,
            kind=n.kind.value,
            label_ru=n.label_ru,
            label_uz=n.label_uz,
            physiology=n.physiology.value,
            confidence=n.confidence,
            critical=n.critical,
            implausible=n.implausible,
            event_code=n.event_code,
        )
    for e in graph.edges:
        g.add_edge(
            e.source,
            e.target,
            rule_id=e.rule_id,
            relation_ru=e.relation_ru,
            physiology=e.physiology.value,
            confidence=e.confidence,
        )
    return g
