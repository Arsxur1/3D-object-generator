"""PDF report assembly (TZ §11.7).

Combines the summary, key curves (montage / DSA / aEEG), topomaps, the causal
graph, and the RU/UZ conclusion + disclaimer into one PDF. A Unicode font
(DejaVu Sans) is registered so Cyrillic renders correctly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ..contracts.events import DetectionResult
from ..contracts.report import LLMReport
from ..contracts.signal import UnifiedSignal

_FONT = "NLSans"
_FONT_B = "NLSans-Bold"


def _register_fonts() -> None:
    if _FONT in pdfmetrics.getRegisteredFontNames():
        return
    from matplotlib import font_manager as fm

    reg = fm.findfont("DejaVu Sans")
    bold = fm.findfont("DejaVu Sans:bold")
    pdfmetrics.registerFont(TTFont(_FONT, reg))
    pdfmetrics.registerFont(TTFont(_FONT_B, bold))


def _styles():
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["Normal"], fontName=_FONT, fontSize=8.5, leading=11)
    h = ParagraphStyle("h", parent=ss["Heading2"], fontName=_FONT_B, fontSize=12, spaceAfter=4)
    title = ParagraphStyle("title", parent=ss["Title"], fontName=_FONT_B, fontSize=16)
    crit = ParagraphStyle("crit", parent=body, textColor=colors.HexColor("#b00020"), fontName=_FONT_B)
    small = ParagraphStyle("small", parent=body, fontSize=7, textColor=colors.grey)
    return {"body": body, "h": h, "title": title, "crit": crit, "small": small}


def _img(path: str | Path, max_w: float = 17 * cm):
    path = Path(path)
    if not path.exists():
        return None
    w, hgt = ImageReader(str(path)).getSize()
    scale = min(max_w / w, 1.0)
    return Image(str(path), width=w * scale, height=hgt * scale)


def build_pdf_report(
    report: LLMReport,
    detection: DetectionResult,
    signal: UnifiedSignal,
    montage_name: str,
    image_paths: Mapping[str, Path],
    out_path: str | Path,
    sleep: dict | None = None,
) -> Path:
    _register_fonts()
    st = _styles()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                            leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.4 * cm, bottomMargin=1.4 * cm)
    story = []

    story.append(Paragraph("Протокол ЭЭГ — NeuroLens", st["title"]))
    story.append(Paragraph(
        f"Режим: {report.mode.value} &nbsp;&nbsp; Провайдер: {report.provider} &nbsp;&nbsp; "
        f"Монтаж: {montage_name} &nbsp;&nbsp; Референс: {signal.reference}", st["small"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("⚕ " + report.disclaimer_ru, st["small"]))
    story.append(Spacer(1, 8))

    # patient / context
    p = signal.patient
    c = signal.context
    ctx_rows = [
        ["Возраст", _fmt(p.age_years), "ПМВ (нед)", _fmt(p.postmenstrual_age_weeks)],
        ["Седация", ", ".join(c.sedatives) or "—", "ПЭП", ", ".join(c.antiseizure_meds) or "—"],
        ["Температура", _fmt(c.temperature_c), "Клин. вопрос", (c.clinical_question or "—")[:60]],
    ]
    story.append(_table(ctx_rows))
    story.append(Spacer(1, 8))

    # critical findings
    if report.critical_findings:
        story.append(Paragraph("Критические находки", st["h"]))
        for cf in report.critical_findings:
            story.append(Paragraph(f"‼ [{cf.code}] {cf.text.ru}", st["crit"]))
        story.append(Spacer(1, 6))

    # protocol sections
    section_titles = {
        "background": "Фон", "epileptiform": "Эпилептиформные изменения",
        "ictal_acns": "Судороги / ACNS / ИИК", "causal": "Причинно-следственный анализ",
        "next_step": "Следующий шаг",
    }
    for key, title in section_titles.items():
        secs = [s for s in report.sections if s.key == key]
        if not secs:
            continue
        story.append(Paragraph(title, st["h"]))
        for s in secs:
            story.append(Paragraph(f"• {s.text.ru}", st["body"]))
            story.append(Paragraph(f"<i>UZ:</i> {s.text.uz}", st["small"]))
        story.append(Spacer(1, 4))

    # events table
    if detection.events:
        story.append(Paragraph("События", st["h"]))
        rows = [["Код", "Локализация", "Время, с", "Увер.", "ACNS"]]
        for e in detection.events[:24]:
            loc = e.localization.lateralization or (e.localization.region or "—")
            acns = ""
            if e.acns:
                bits = []
                if e.acns.frequency_hz is not None:
                    bits.append(f"{e.acns.frequency_hz}Гц")
                if e.acns.plus_features:
                    bits.append("".join(e.acns.plus_features))
                if e.acns.iic:
                    bits.append("ИИК")
                acns = " ".join(bits)
            rows.append([e.code, loc, f"{e.t_start:.0f}–{e.t_end:.0f}", f"{e.confidence:.2f}", acns])
        story.append(_table(rows, header=True))
        story.append(Spacer(1, 6))

    # sleep staging (hypnogram summary)
    if sleep:
        from .sleep_section import sleep_text

        s = sleep["summary"]
        pct = s["stage_percent_of_sleep"]
        eff = s.get("sleep_efficiency")
        story.append(Paragraph("Сон (автоматическое стадирование AASM)", st["h"]))
        story.append(_table([
            ["Время записи, мин", _fmt(s["time_in_bed_min"]), "Сон, мин", _fmt(s["total_sleep_min"])],
            ["Эффективность", "—" if eff is None else f"{100 * float(eff):.0f}%",
             "Латентность сна / REM, мин", f"{_fmt(s['sleep_latency_min'])} / {_fmt(s['rem_latency_min'])}"],
            ["N1 / N2 / N3 / REM, % сна", " / ".join(f"{float(pct.get(k, 0)):.0f}" for k in ("N1", "N2", "N3", "REM")),
             "Производные", sleep.get("derivations", "—")],
        ]))
        txt = sleep_text(sleep)
        story.append(Paragraph(txt.ru, st["small"]))
        story.append(Paragraph(f"<i>UZ:</i> {txt.uz}", st["small"]))
        img = _img(image_paths.get("hypnogram", "")) if image_paths.get("hypnogram") else None
        if img is not None:
            story.append(img)
        story.append(Spacer(1, 6))

    # impression + confidence
    story.append(Paragraph("Заключение", st["h"]))
    story.append(Paragraph(report.impression.ru, st["body"]))
    story.append(Paragraph(f"<i>UZ:</i> {report.impression.uz}", st["small"]))
    conf = report.overall_confidence
    story.append(Paragraph(
        f"Уверенность: {conf.value} ({conf.label_ru})", st["body"]))
    if report.escalation_reason:
        story.append(Paragraph("⚠ Требуется просмотр врача: " + report.escalation_reason.ru, st["crit"]))

    # figures
    for name, cap in [("curves", "Размеченные кривые"), ("dsa_aeeg", "DSA + aEEG"),
                      ("topomap", "Топокарты"), ("causal_graph", "Каузальный граф")]:
        img = _img(image_paths.get(name, "")) if image_paths.get(name) else None
        if img is not None:
            story.append(PageBreak())
            story.append(Paragraph(cap, st["h"]))
            story.append(img)

    # reasoning trace
    if report.reasoning_trace:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Цепочка рассуждений", st["h"]))
        for i, step in enumerate(report.reasoning_trace, 1):
            story.append(Paragraph(f"{i}. {step}", st["small"]))

    story.append(Spacer(1, 10))
    story.append(Paragraph("⚕ " + report.disclaimer_ru, st["small"]))
    story.append(Paragraph("⚕ " + report.disclaimer_uz, st["small"]))

    doc.build(story)
    return out_path


def _fmt(v) -> str:
    return "—" if v is None else str(v)


def _table(rows, header: bool = False) -> Table:
    t = Table(rows, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), _FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f4f4f4")]),
    ]
    if header:
        style += [("FONTNAME", (0, 0), (-1, 0), _FONT_B),
                  ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e0e0"))]
    t.setStyle(TableStyle(style))
    return t
