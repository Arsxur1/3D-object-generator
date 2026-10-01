"""Causal graph contract — output of Layer 5 (TZ §8).

The graph is the core interpretation artifact. Nodes are events or mechanisms;
edges are cause -> effect relations. Nodes/edges carry a physiology label
(physiologic vs pathologic), a confidence, and safety/plausibility flags.

Layer 6 narrates *only* what is present in this graph (grounding, TZ §9).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class NodeKind(str, Enum):
    EVENT = "event"  # a detected finding (references a Layer-4 event code)
    MECHANISM = "mechanism"  # an inferred physiological/pathological mechanism
    CONTEXT = "context"  # a clinical-context fact used in reasoning


class PhysiologyLabel(str, Enum):
    PHYSIOLOGIC = "physiologic"  # normal / expected for age / activation response
    PATHOLOGIC = "pathologic"
    UNCERTAIN = "uncertain"
    ARTIFACT = "artifact"  # explained by a non-cerebral source


class CausalNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: NodeKind
    label_ru: str
    label_uz: str
    physiology: PhysiologyLabel = PhysiologyLabel.UNCERTAIN
    confidence: float = Field(ge=0, le=1)
    event_code: Optional[str] = Field(
        default=None, description="Links an EVENT node back to a Layer-4 Event.code."
    )
    critical: bool = Field(default=False, description="Hard-safety 'must not miss' finding.")
    implausible: bool = Field(
        default=False, description="Nothing explains this node -> check artifact/electrode."
    )
    evidence_refs: list[str] = Field(
        default_factory=list,
        description="Feature names / event codes that ground this node.",
    )


class CausalEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str  # node id (cause)
    target: str  # node id (effect)
    relation_ru: str = "приводит к"
    relation_uz: str = "olib keladi"
    rule_id: Optional[str] = Field(default=None, description="Rule from the YAML rule base.")
    physiology: PhysiologyLabel = PhysiologyLabel.UNCERTAIN
    confidence: float = Field(ge=0, le=1)


class CausalGraph(BaseModel):
    """Causal graph + critical/plausibility flags (TZ §8 output)."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[CausalNode] = Field(default_factory=list)
    edges: list[CausalEdge] = Field(default_factory=list)
    critical_flags: list[str] = Field(default_factory=list)
    implausible_flags: list[str] = Field(default_factory=list)
    rule_base_version: Optional[str] = None
    notes: list[str] = Field(default_factory=list)

    # -- helpers used by Layer 6 / grounding -----------------------------
    def node(self, node_id: str) -> Optional[CausalNode]:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def grounding_keys(self) -> set[str]:
        """All identifiers a downstream statement may legitimately cite."""
        keys: set[str] = set()
        for n in self.nodes:
            keys.add(n.id)
            if n.event_code:
                keys.add(n.event_code)
            keys.update(n.evidence_refs)
        for e in self.edges:
            keys.add(f"{e.source}->{e.target}")
            if e.rule_id:
                keys.add(e.rule_id)
        return keys

    def has_critical(self) -> bool:
        return bool(self.critical_flags) or any(n.critical for n in self.nodes)
