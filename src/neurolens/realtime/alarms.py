"""Alarm gating for real-time cEEG (TZ §2, §10, §16).

Turns per-window detection candidates into alarms while controlling false
alarms/hour. Three gates: persistence (finding must recur over N consecutive
windows), refractory (no repeat of a type within a cool-off), and a per-hour
cap. Critical types (TZ §10) bypass the per-hour cap but still respect
persistence + refractory, so they surface without spamming every window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..contracts.alarms import Alarm, AlarmSeverity, AlarmType
from ..contracts.config import RealtimeConfig
from ..contracts.report import Bilingual


@dataclass
class AlarmCandidate:
    """A per-window finding proposed for alarming."""

    type: AlarmType
    severity: AlarmSeverity
    message: Bilingual
    confidence: float
    channels: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    is_critical: bool = True


class AlarmManager:
    def __init__(self, cfg: RealtimeConfig):
        self.cfg = cfg
        self._streak: dict[str, int] = {}
        self._last_fire: dict[str, float] = {}
        self._fire_times: dict[str, list[float]] = {}
        self.n_raised = 0
        self.n_suppressed = 0
        self._counter = 0

    def process(
        self,
        candidates: list[AlarmCandidate],
        window_index: int,
        t0: float,
        t1: float,
        latency_ms: float,
    ) -> list[Alarm]:
        # candidates meeting their type's min confidence (best per type)
        ok: dict[str, AlarmCandidate] = {}
        for c in candidates:
            rule = self.cfg.rule_for(c.type.value)
            if c.confidence >= rule.min_confidence:
                prev = ok.get(c.type.value)
                if prev is None or c.confidence > prev.confidence:
                    ok[c.type.value] = c

        # update persistence streaks (present -> +1, absent -> reset)
        for tval in set(self._streak) | set(ok):
            self._streak[tval] = self._streak.get(tval, 0) + 1 if tval in ok else 0

        fired: list[Alarm] = []
        for tval, c in ok.items():
            rule = self.cfg.rule_for(tval)
            if self._streak[tval] < rule.persistence_windows:
                continue
            last = self._last_fire.get(tval)
            if last is not None and (t0 - last) < rule.refractory_s:
                continue
            recent = [ft for ft in self._fire_times.get(tval, []) if t0 - ft <= 3600.0]
            self._fire_times[tval] = recent
            if len(recent) >= rule.max_per_hour and not c.is_critical:
                self.n_suppressed += 1
                continue

            alarm = Alarm(
                id=f"alarm-{self._counter:04d}",
                type=c.type,
                severity=c.severity,
                message=c.message,
                t_start=round(t0, 2),
                t_end=round(t1, 2),
                channels=c.channels,
                confidence=round(c.confidence, 3),
                evidence=c.evidence,
                window_index=window_index,
                latency_ms=round(latency_ms, 1),
                is_critical_finding=c.is_critical,
            )
            fired.append(alarm)
            self._counter += 1
            self.n_raised += 1
            self._last_fire[tval] = t0
            self._fire_times.setdefault(tval, []).append(t0)
        return fired
