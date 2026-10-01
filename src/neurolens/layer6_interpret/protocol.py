"""Render an :class:`LLMReport` to a human-readable RU+UZ protocol (TZ §9, §11.2)."""

from __future__ import annotations

from ..contracts.report import LLMReport

_SECTION_TITLES = {
    "background": ("Фон", "Fon"),
    "epileptiform": ("Эпилептиформные изменения", "Epileptiform o‘zgarishlar"),
    "ictal_acns": ("Судороги / паттерны ACNS / ИИК", "Tutqanoq / ACNS patternlari / IIK"),
    "causal": ("Причинно-следственный анализ", "Sabab-oqibat tahlili"),
    "next_step": ("Следующий шаг", "Keyingi qadam"),
}


def render_protocol(report: LLMReport) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("ПРОТОКОЛ ЭЭГ / EEG BAYONNOMASI — NeuroLens")
    lines.append(f"Режим / Rejim: {report.mode.value}   Провайдер / Provayder: {report.provider}")
    lines.append("=" * 72)

    if report.escalation_reason:
        lines.append("")
        lines.append("⚠ ТРЕБУЕТСЯ ПРОСМОТР ВРАЧА / SHIFOKOR KO‘RIGI ZARUR")
        lines.append(f"   RU: {report.escalation_reason.ru}")
        lines.append(f"   UZ: {report.escalation_reason.uz}")

    # critical findings first (hard safety)
    if report.critical_findings:
        lines.append("")
        lines.append("‼ КРИТИЧЕСКИЕ НАХОДКИ / KRITIK TOPILMALAR:")
        for cf in report.critical_findings:
            lines.append(f"  • [{cf.code}]")
            lines.append(f"    RU: {cf.text.ru}")
            lines.append(f"    UZ: {cf.text.uz}")

    # grouped sections in canonical order
    order = ["background", "epileptiform", "ictal_acns", "causal", "next_step"]
    for key in order:
        secs = [s for s in report.sections if s.key == key]
        if not secs:
            continue
        title_ru, title_uz = _SECTION_TITLES.get(key, (key, key))
        lines.append("")
        lines.append(f"— {title_ru} / {title_uz} —")
        for s in secs:
            lines.append(f"  RU: {s.text.ru}")
            lines.append(f"  UZ: {s.text.uz}")
            if s.grounding_refs:
                lines.append(f"     [обоснование/grounding: {', '.join(s.grounding_refs)}]")

    lines.append("")
    lines.append("— Заключение / Xulosa —")
    lines.append(f"  RU: {report.impression.ru}")
    lines.append(f"  UZ: {report.impression.uz}")
    c = report.overall_confidence
    lines.append(f"  Уверенность / Ishonch: {c.value} ({c.label_ru} / {c.label_uz})")

    if report.reasoning_trace:
        lines.append("")
        lines.append("— Цепочка рассуждений / Fikrlash zanjiri —")
        for i, step in enumerate(report.reasoning_trace, 1):
            lines.append(f"  {i}. {step}")

    lines.append("")
    lines.append("-" * 72)
    lines.append(f"⚕ {report.disclaimer_ru}")
    lines.append(f"⚕ {report.disclaimer_uz}")
    lines.append("-" * 72)
    return "\n".join(lines)
