"""Deterministic, fully-grounded Layer-6 provider (TZ §9).

Builds the RU/UZ protocol directly from the causal graph and features. It
cannot hallucinate: every sentence is templated from a specific node/edge/
feature and cites its grounding. This is the default provider and the one used
in tests (no network, reproducible). The Anthropic provider produces richer
prose but is validated against the same grounding rules.
"""

from __future__ import annotations

from ..contracts.report import (
    Bilingual,
    Confidence,
    LLMReport,
    OperatingMode,
    ReportSection,
)
from .llm_base import LLMInput, LLMProvider


def _confidence(value: float) -> Confidence:
    if value >= 0.75:
        ru, uz = "высокая", "yuqori"
    elif value >= 0.5:
        ru, uz = "средняя", "o‘rtacha"
    else:
        ru, uz = "низкая", "past"
    return Confidence(value=round(value, 2), label_ru=ru, label_uz=uz)


class DeterministicProvider(LLMProvider):
    name = "deterministic"

    def generate(self, data: LLMInput) -> LLMReport:
        sections: list[ReportSection] = []
        trace: list[str] = []

        # --- background ---
        bg = data.background
        sections.append(
            ReportSection(
                key="background",
                text=Bilingual(
                    ru=(
                        f"Фон: задний доминирующий ритм ~{bg['pdr_hz']} Гц; "
                        f"относительная мощность дельта {bg['global_rel_delta']}, "
                        f"тета {bg['global_rel_theta']}, альфа {bg['global_rel_alpha']}; "
                        f"межполушарная асимметрия {bg['global_asymmetry']}; "
                        f"непрерывность {bg['continuity']}."
                    ),
                    uz=(
                        f"Fon: orqa dominant ritm ~{bg['pdr_hz']} Hz; nisbiy quvvat "
                        f"delta {bg['global_rel_delta']}, teta {bg['global_rel_theta']}, "
                        f"alfa {bg['global_rel_alpha']}; assimetriya {bg['global_asymmetry']}; "
                        f"uzluksizlik {bg['continuity']}."
                    ),
                ),
                grounding_refs=[
                    "pdr_hz", "global_rel_delta", "global_rel_theta",
                    "global_rel_alpha", "global_asymmetry", "continuity",
                ],
            )
        )
        trace.append("Фон охарактеризован по количественным признакам Слоя 3.")

        # --- epileptiform (IED), including artifact resolution ---
        ied = [e for e in data.events if e["group"] == "ied"]
        if ied:
            resolved = [
                n for n in data.causal_nodes
                if n["event_code"] == "spike" and n["physiology"] == "artifact"
            ]
            if resolved and len(resolved) == len(ied):
                txt = Bilingual(
                    ru="Острые транзиенты объяснены ЭКГ-артефактом, а не истинными эпилептиформными разрядами.",
                    uz="O‘tkir tranzientlar haqiqiy epileptiform razryad emas, EKG artefakti bilan izohlandi.",
                )
            else:
                regions = sorted({e["localization"].get("region") or "?" for e in ied})
                txt = Bilingual(
                    ru=f"Эпилептиформные разряды (спайки/острые волны): {len(ied)}; локализация: {', '.join(regions)}.",
                    uz=f"Epileptiform razryadlar (spike/o‘tkir to‘lqin): {len(ied)}; lokalizatsiya: {', '.join(regions)}.",
                )
            sections.append(
                ReportSection(key="epileptiform", text=txt, grounding_refs=["spike", "ecg_artifact"])
            )
            trace.append("Эпилептиформные кандидаты сверены с гипотезой ЭКГ-артефакта (Слой 5).")

        # --- ictal / ACNS ---
        ictal = [e for e in data.events if e["group"] in ("ictal", "periodic")]
        if ictal:
            e = ictal[0]
            dur = e["t_end"] - e["t_start"]
            lat = e["localization"].get("lateralization") or "?"
            sections.append(
                ReportSection(
                    key="ictal_acns",
                    text=Bilingual(
                        ru=f"Иктальный ритмический паттерн ~{dur:.0f} с, латерализация: {lat}.",
                        uz=f"Iktal ritmik pattern ~{dur:.0f} s, lateralizatsiya: {lat}.",
                    ),
                    grounding_refs=[e["code"]],
                    confidence=_confidence(e["confidence"]),
                )
            )
            trace.append("Иктальное событие зарегистрировано детектором ритмической активности.")

        # --- causal chain ---
        seen_edges: set[tuple] = set()
        for edge in data.causal_edges:
            sig = (edge["source"], edge["target"], edge["rule_id"])
            if sig in seen_edges:
                continue
            seen_edges.add(sig)
            src = _node(data, edge["source"])
            tgt = _node(data, edge["target"])
            if not src or not tgt:
                continue
            phys_ru = _phys_ru(edge["physiology"])
            sections.append(
                ReportSection(
                    key="causal",
                    text=Bilingual(
                        ru=f"{src['label_ru']} → {tgt['label_ru']} ({phys_ru}).",
                        uz=f"{src['label_uz']} → {tgt['label_uz']}.",
                    ),
                    grounding_refs=[f"{edge['source']}->{edge['target']}", edge["rule_id"] or ""],
                    confidence=_confidence(edge["confidence"]),
                )
            )
            trace.append(
                f"Правило {edge['rule_id']}: {src['label_ru']} → {tgt['label_ru']}."
            )

        # --- impression + mode handling ---
        patho = [n for n in data.causal_nodes if n["physiology"] == "pathologic"]
        conf_val = max((n["confidence"] for n in patho), default=0.5) if patho else 0.5

        if patho:
            impression = Bilingual(
                ru="Заключение: картина отклонений — " + "; ".join(sorted({n["label_ru"] for n in patho}))[:400] + ".",
                uz="Xulosa: og‘ishlar manzarasi — " + "; ".join(sorted({n["label_uz"] for n in patho}))[:400] + ".",
            )
        else:
            impression = Bilingual(
                ru="Заключение: явных патологических паттернов не выделено (в пределах анализа).",
                uz="Xulosa: aniq patologik patternlar ajratilmadi (tahlil doirasida).",
            )

        mode_gate_passed = False
        escalation = None
        if data.mode == OperatingMode.B_AUTONOMOUS:
            if data.gate.passed:
                mode_gate_passed = True
                trace.append("Режим B: гейты пройдены — сформировано автономное заключение.")
            else:
                escalation = Bilingual(
                    ru="Требуется просмотр врача: " + "; ".join(data.gate.reasons_ru),
                    uz="Shifokor ko‘rigi zarur: " + "; ".join(data.gate.reasons_uz),
                )
                trace.append("Режим B: гейты не пройдены — эскалация на просмотр врача.")

        sections.append(
            ReportSection(
                key="next_step",
                text=Bilingual(
                    ru="Следующий шаг: клиническая корреляция; при необходимости — расширенный/непрерывный мониторинг.",
                    uz="Keyingi qadam: klinik korrelyatsiya; zarur bo‘lsa — kengaytirilgan/uzluksiz monitoring.",
                ),
                grounding_refs=[],
            )
        )

        # dedupe reasoning trace, preserving order
        seen: set[str] = set()
        trace = [s for s in trace if not (s in seen or seen.add(s))]

        return LLMReport(
            mode=data.mode,
            mode_gate_passed=mode_gate_passed,
            escalation_reason=escalation,
            sections=sections,
            impression=impression,
            overall_confidence=_confidence(conf_val),
            reasoning_trace=trace,
            provider=self.name,
        )


def _node(data: LLMInput, node_id: str):
    return next((n for n in data.causal_nodes if n["id"] == node_id), None)


def _phys_ru(p: str) -> str:
    return {
        "physiologic": "физиологично",
        "pathologic": "патологично",
        "artifact": "артефакт",
        "uncertain": "неопределённо",
    }.get(p, p)
