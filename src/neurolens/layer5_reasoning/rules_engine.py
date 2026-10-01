"""Rules engine — builds the causal graph from events + rule base + context (TZ §8).

The rule base (``configs/rules/neuro_rules.yaml``) is the configurable knowledge
graph. This engine matches each chain's condition against the detected events
and clinical context, then emits cause->effect edges with physiology labels and
confidences. It also performs the ECG-artifact-vs-IED resolution and applies
context splits (e.g. burst-suppression: sedation/hypothermia vs severe injury).
"""

from __future__ import annotations

from ..contracts.causal import (
    CausalEdge,
    CausalGraph,
    CausalNode,
    NodeKind,
    PhysiologyLabel,
)
from ..contracts.config import RuleBase, RuleChain
from ..contracts.events import DetectionResult, Event
from ..contracts.signal import ClinicalContext, PatientInfo
from ..layer2_preprocess.artifacts import ArtifactReport
from .physiology import PhysiologyEngine
from .plausibility import check_plausibility

_PHYS = {
    "physiologic": PhysiologyLabel.PHYSIOLOGIC,
    "pathologic": PhysiologyLabel.PATHOLOGIC,
    "uncertain": PhysiologyLabel.UNCERTAIN,
    "artifact": PhysiologyLabel.ARTIFACT,
}

_CRITICAL_GROUPS = {"ictal", "suppression"}


class RulesEngine:
    def __init__(self, rule_base: RuleBase, physiology: PhysiologyEngine):
        self.rule_base = rule_base
        self.physiology = physiology

    # -- public ----------------------------------------------------------
    def build(
        self,
        detection: DetectionResult,
        context: ClinicalContext,
        patient: PatientInfo,
        artifacts: ArtifactReport | None = None,
    ) -> CausalGraph:
        plaus = check_plausibility(detection, artifacts)
        resolutions = plaus.get("spike_resolutions", {})

        nodes: list[CausalNode] = []
        edges: list[CausalEdge] = []
        critical_flags: list[str] = []
        implausible_flags: list[str] = []

        # 1) event nodes
        event_node_ids: dict[int, str] = {}
        code_to_nodes: dict[str, list[str]] = {}
        for i, ev in enumerate(detection.events):
            nid = f"evt:{ev.code}:{i}"
            resolved = resolutions.get(id(ev))
            if resolved and resolved.get("kind") == "artifact":
                phys = PhysiologyLabel.ARTIFACT
            elif resolved and resolved.get("kind") == "variant":
                phys = PhysiologyLabel.PHYSIOLOGIC
            else:
                phys = self.physiology.label_event(ev, patient)
            critical = ev.group in _CRITICAL_GROUPS or ev.code == "lpds"
            node = CausalNode(
                id=nid,
                kind=NodeKind.EVENT,
                label_ru=ev.label_ru,
                label_uz=ev.label_uz,
                physiology=phys,
                confidence=ev.confidence,
                event_code=ev.code,
                critical=critical,
                evidence_refs=[e.feature for e in ev.evidence] + [ev.code],
            )
            nodes.append(node)
            event_node_ids[id(ev)] = nid
            code_to_nodes.setdefault(ev.code, []).append(nid)
            if critical:
                critical_flags.append(f"{ev.code}: {ev.label_ru}")

        present = detection.codes()

        # 2) apply rule chains
        for rule in self.rule_base.chains:
            if not self._matches(rule, present, context):
                continue
            if rule.resolves_artifact_for:
                self._apply_artifact_resolution(
                    rule, detection, resolutions, event_node_ids, code_to_nodes,
                    nodes, edges, implausible_flags,
                )
                continue
            self._apply_causal_chain(
                rule, detection, context, code_to_nodes, nodes, edges, critical_flags
            )

        # 2b) benign-variant (wicket) resolution of spike candidates
        self._apply_variant_resolution(
            detection, resolutions, event_node_ids, code_to_nodes, nodes, edges
        )

        # 3) unresolved spikes with low support -> implausible (needs review)
        for ev in detection.by_group("ied"):
            nid = event_node_ids[id(ev)]
            if id(ev) not in resolutions and ev.confidence < 0.6:
                node = next(n for n in nodes if n.id == nid)
                node.implausible = True
                implausible_flags.append(
                    f"{ev.code}@{ev.t_start:.1f}s: не подтверждён — проверить артефакт/вариант"
                )

        return CausalGraph(
            nodes=nodes,
            edges=edges,
            critical_flags=critical_flags,
            implausible_flags=implausible_flags,
            rule_base_version=self.rule_base.version,
        )

    # -- helpers ---------------------------------------------------------
    def _matches(self, rule: RuleChain, present: set[str], context: ClinicalContext) -> bool:
        w = rule.when
        if w.any_event and not (set(w.any_event) & present):
            return False
        if w.all_events and not set(w.all_events).issubset(present):
            return False
        for token in w.context_present:
            if not _context_has(context, token):
                return False
        for token in w.context_absent:
            if _context_has(context, token):
                return False
        return True

    def _apply_causal_chain(
        self, rule, detection, context, code_to_nodes, nodes, edges, critical_flags
    ) -> None:
        # anchor cause on the first present triggering event node
        trigger_codes = list(rule.when.any_event) + list(rule.when.all_events)
        cause_id = next(
            (code_to_nodes[c][0] for c in trigger_codes if c in code_to_nodes), None
        )
        if cause_id is None:
            return

        mech_id = f"mech:{rule.id}"
        phys = _PHYS.get(rule.physiology, PhysiologyLabel.UNCERTAIN)
        effect_ru, effect_uz = rule.effect_ru, rule.effect_uz
        note = None

        # context split for burst-suppression
        if rule.id == "burst_suppression_context":
            if context.sedatives or (context.temperature_c is not None and context.temperature_c < 34):
                note = "контекст: седация/гипотермия — вероятна обратимая причина"
            else:
                note = "контекст: седативы/гипотермия не указаны — приоритет тяжёлого повреждения"

        if not any(n.id == mech_id for n in nodes):
            nodes.append(
                CausalNode(
                    id=mech_id,
                    kind=NodeKind.MECHANISM,
                    label_ru=effect_ru + (f" [{note}]" if note else ""),
                    label_uz=effect_uz,
                    physiology=phys,
                    confidence=rule.confidence,
                    critical=rule.critical,
                    evidence_refs=[rule.id],
                )
            )
        edges.append(
            CausalEdge(
                source=cause_id,
                target=mech_id,
                rule_id=rule.id,
                physiology=phys,
                confidence=rule.confidence,
            )
        )
        if rule.critical:
            critical_flags.append(f"{rule.id}: {effect_ru}")

    def _apply_artifact_resolution(
        self, rule, detection, resolutions, event_node_ids, code_to_nodes,
        nodes, edges, implausible_flags,
    ) -> None:
        ecg_ids = code_to_nodes.get("ecg_artifact", [])
        if not ecg_ids:
            return
        ecg_id = ecg_ids[0]

        resolved = [
            ev for ev in detection.by_group("ied")
            if resolutions.get(id(ev), {}).get("kind") == "artifact"
        ]
        if not resolved:
            return

        # mark each resolved spike node as artifact (no per-spike edge flood)
        reasons = set()
        for ev in resolved:
            node = next(n for n in nodes if n.id == event_node_ids[id(ev)])
            node.physiology = PhysiologyLabel.ARTIFACT
            reasons.add(resolutions[id(ev)]["reason"])

        # single aggregated pseudo-IED mechanism node + one edge
        mech_id = "mech:pseudo_ied"
        if not any(n.id == mech_id for n in nodes):
            nodes.append(
                CausalNode(
                    id=mech_id,
                    kind=NodeKind.MECHANISM,
                    label_ru=(
                        f"Псевдо-эпилептиформные транзиенты (n={len(resolved)}) объяснены "
                        f"ЭКГ-артефактом ({'; '.join(sorted(reasons))}) — не истинные IED"
                    ),
                    label_uz=(
                        f"Soxta-epileptiform tranzientlar (n={len(resolved)}) EKG artefakti "
                        f"bilan izohlandi — haqiqiy IED emas"
                    ),
                    physiology=PhysiologyLabel.ARTIFACT,
                    confidence=rule.confidence,
                    evidence_refs=[rule.id, "spike", "ecg_artifact"],
                )
            )
        edges.append(
            CausalEdge(
                source=ecg_id,
                target=mech_id,
                rule_id=rule.id,
                relation_ru="объясняет как артефакт",
                relation_uz="artefakt sifatida izohlaydi",
                physiology=PhysiologyLabel.ARTIFACT,
                confidence=rule.confidence,
            )
        )

    def _apply_variant_resolution(
        self, detection, resolutions, event_node_ids, code_to_nodes, nodes, edges
    ) -> None:
        """Resolve spikes coinciding with a benign variant (wicket) as physiologic."""
        resolved = [
            ev for ev in detection.by_group("ied")
            if resolutions.get(id(ev), {}).get("kind") == "variant"
        ]
        if not resolved:
            return
        wicket_ids = code_to_nodes.get("wicket", [])
        for ev in resolved:
            node = next(n for n in nodes if n.id == event_node_ids[id(ev)])
            node.physiology = PhysiologyLabel.PHYSIOLOGIC
            node.label_ru += " → доброкачественный вариант (wicket), не IED"

        mech_id = "mech:benign_variant"
        if not any(n.id == mech_id for n in nodes):
            nodes.append(
                CausalNode(
                    id=mech_id,
                    kind=NodeKind.MECHANISM,
                    label_ru=(
                        f"Острые транзиенты (n={len(resolved)}) соответствуют "
                        f"доброкачественному варианту (wicket) — физиологично, не эпилептиформно"
                    ),
                    label_uz=(
                        f"O‘tkir tranzientlar (n={len(resolved)}) xavfsiz variantga (wicket) "
                        f"mos — fiziologik, epileptiform emas"
                    ),
                    physiology=PhysiologyLabel.PHYSIOLOGIC,
                    confidence=0.6,
                    evidence_refs=["wicket", "spike"],
                )
            )
        if wicket_ids:
            edges.append(
                CausalEdge(
                    source=wicket_ids[0],
                    target=mech_id,
                    rule_id="wicket_benign_variant",
                    relation_ru="объясняет как вариант",
                    relation_uz="variant sifatida izohlaydi",
                    physiology=PhysiologyLabel.PHYSIOLOGIC,
                    confidence=0.6,
                )
            )


def _context_has(context: ClinicalContext, token: str) -> bool:
    token = token.lower()
    if token in ("sedatives", "sedation"):
        return bool(context.sedatives)
    if token in ("antiseizure_meds", "asm", "aed"):
        return bool(context.antiseizure_meds)
    if token in ("hypothermia",):
        return context.temperature_c is not None and context.temperature_c < 34.0
    if token in ("fever", "hyperthermia"):
        return context.temperature_c is not None and context.temperature_c > 38.0
    if token in ("hepatic_failure", "renal_failure", "metabolic"):
        return any(token.split("_")[0] in s.lower() for s in context.symptoms)
    # generic: match against symptoms text
    return any(token in s.lower() for s in context.symptoms)


def build_causal_graph(
    detection: DetectionResult,
    context: ClinicalContext,
    patient: PatientInfo,
    rule_base: RuleBase,
    physiology: PhysiologyEngine,
    artifacts: ArtifactReport | None = None,
) -> CausalGraph:
    """Convenience wrapper used by the pipeline."""
    return RulesEngine(rule_base, physiology).build(detection, context, patient, artifacts)
