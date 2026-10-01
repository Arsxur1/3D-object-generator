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

    # IIC / periodic patterns requiring attention (codes appear from v2 detectors).
    for code in ("lpds", "gpds", "bipds", "lrda"):
        evs = [e for e in detection.events if e.code == code]
        for ev in evs:
            findings.append(
                CriticalFinding(
                    code=f"iic_{code}",
                    text=Bilingual(
                        ru=f"Паттерн {code.upper()} на иктально-интериктальном континууме — требует внимания.",
                        uz=f"{code.upper()} patterni iktal-interiktal kontinuumda — e’tibor talab qiladi.",
                    ),
                    grounding_refs=[ev.code],
                )
            )

    return findings
