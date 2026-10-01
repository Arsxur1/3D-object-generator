"""Critical 'must not miss' findings (TZ §10).

These are surfaced always, in every report, regardless of operating mode or
what the LLM says. Sensitivity is prioritized; each finding carries the
grounding (event codes) that triggered it. Electrocerebral silence is NOT
auto-asserted (brain-death context is out of scope) — severe suppression is
flagged as a finding requiring technical review instead.
"""

from __future__ import annotations

from ..contracts.events import DetectionResult, Event
from ..contracts.report import Bilingual, CriticalFinding

# Seconds of continuous ictal activity that should prompt status-epilepticus review.
STATUS_DURATION_S = 30.0


def scan_critical_findings(detection: DetectionResult) -> list[CriticalFinding]:
    findings: list[CriticalFinding] = []

    for ev in detection.by_group("ictal"):
        dur = ev.duration_s
        if dur >= STATUS_DURATION_S:
            findings.append(
                CriticalFinding(
                    code="status_epilepticus_suspected",
                    text=Bilingual(
                        ru=(
                            f"Продолжающаяся иктальная активность ~{dur:.0f} с — "
                            "рассмотреть эпилептический статус (в т.ч. бессудорожный). "
                            "Требуется немедленная верификация врачом."
                        ),
                        uz=(
                            f"~{dur:.0f} s davom etayotgan iktal faollik — epileptik "
                            "status (jumladan, tutqanoqsiz) ko‘rib chiqilsin. "
                            "Zudlik bilan shifokor tekshiruvi zarur."
                        ),
                    ),
                    grounding_refs=[ev.code],
                )
            )
        else:
            findings.append(
                CriticalFinding(
                    code="ongoing_seizure",
                    text=Bilingual(
                        ru=f"Зарегистрирована судорожная активность (~{dur:.0f} с). Клиническая корреляция.",
                        uz=f"Tutqanoq faolligi qayd etildi (~{dur:.0f} s). Klinik korrelyatsiya.",
                    ),
                    grounding_refs=[ev.code],
                )
            )

    for ev in detection.by_group("suppression"):
        findings.append(
            CriticalFinding(
                code="burst_suppression",
                text=Bilingual(
                    ru=(
                        "Паттерн «вспышка-подавление»/выраженная супрессия — тяжёлое "
                        "повреждение против глубокой седации/гипотермии; требует "
                        "срочной клинико-контекстной оценки."
                    ),
                    uz=(
                        "“Portlash-bostirish”/ifodalangan supressiya — og‘ir shikast "
                        "yoki chuqur sedatsiya/gipotermiya; shoshilinch klinik-kontekst "
                        "baholash zarur."
                    ),
                ),
                grounding_refs=[ev.code],
            )
        )

    # IIC / periodic patterns requiring attention (TZ §10) — flagged critical only
    # when on the ictal-interictal continuum (frequency 1.5-2.5 Hz or +F).
    for code in ("lpds", "gpds", "bipds", "lrda"):
        for ev in (e for e in detection.events if e.code == code):
            iic = bool(ev.acns and ev.acns.iic)
            freq = ev.acns.frequency_hz if ev.acns else None
            plus = ",".join(ev.acns.plus_features) if ev.acns else ""
            modifiers = []
            if freq is not None:
                modifiers.append(f"{freq} Гц")
            if plus:
                modifiers.append(plus)
            mod_str = f" ({'; '.join(modifiers)})" if modifiers else ""
            if not iic:
                continue
            findings.append(
                CriticalFinding(
                    code=f"iic_{code}",
                    text=Bilingual(
                        ru=(f"Паттерн {code.upper()}{mod_str} на иктально-интериктальном "
                            "континууме — требует внимания и клинической корреляции."),
                        uz=(f"{code.upper()}{mod_str} patterni iktal-interiktal kontinuumda — "
                            "e’tibor va klinik korrelyatsiya talab qiladi."),
                    ),
                    grounding_refs=[ev.code],
                )
            )

    # High seizure burden (TZ §7.3) — a strong cEEG signal even below status.
    sb = detection.seizure_burden
    if sb and sb.n_seizures >= 2 and sb.seizure_fraction >= 0.1 and not sb.status_epilepticus_suspected:
        findings.append(
            CriticalFinding(
                code="high_seizure_burden",
                text=Bilingual(
                    ru=(f"Высокая судорожная нагрузка: {sb.n_seizures} эпизодов, "
                        f"{sb.seizure_fraction*100:.0f}% записи, {sb.seizures_per_hour:.1f}/час."),
                    uz=(f"Yuqori tutqanoq yuki: {sb.n_seizures} epizod, "
                        f"yozuvning {sb.seizure_fraction*100:.0f}%, {sb.seizures_per_hour:.1f}/soat."),
                ),
                grounding_refs=["ictal_rhythm"],
            )
        )

    return findings
