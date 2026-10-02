"""Learned ictal detector inside the streaming monitor (increment 10).

Each analysis window contributes only its *new* epochs (those after the last
epoch already seen) — the same epoch stream the real-time model was trained
and evaluated on. Features come from :class:`StreamingFeaturizer` (identical to
offline ``featurize``), probabilities are smoothed causally, and while a run of
``min_epochs`` epochs above ``threshold`` is ongoing an ictal event is emitted
for the current window (the AlarmManager then applies refractory/caps).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..contracts.events import Event, EventEvidence, Localization
from ..layer4_detect.base import LABELS
from ..layer4_detect.ml_ictal import SMOOTH_EPOCHS, StreamingFeaturizer


class StreamingIctalML:
    code = "ictal_ml"

    def __init__(self, model, threshold: float, min_epochs: int, smooth: int = SMOOTH_EPOCHS):
        self.model = model
        self.threshold = float(threshold)
        self.min_epochs = int(min_epochs)
        self._feat = StreamingFeaturizer()
        self._pending_times: deque[float] = deque()
        self._p = deque(maxlen=smooth)
        self._last_t = -np.inf
        self._run_start: float | None = None
        self._run_len = 0
        self._run_p: list[float] = []
        self._last_epoch_t = 0.0

    def update(self, t0: float, feats) -> list[Event]:
        """Consume the new epochs of a window starting at stream time ``t0``;
        return an ictal event (times relative to ``t0``) while a qualifying run
        is active at the newest epoch."""
        et = t0 + np.asarray(feats.epoch_times, float)
        new = et > self._last_t + 1e-6
        if new.any():
            self._last_t = float(et[new][-1])
            self._pending_times.extend(et[new].tolist())
            X = self._feat.push(
                np.asarray(feats.epoch_rms)[new], np.asarray(feats.epoch_band_conc)[new],
                np.asarray(feats.epoch_domfreq)[new],
                {k: np.asarray(v)[new] for k, v in feats.epoch_relpow.items()},
            )
            if len(X):
                for t, p in zip([self._pending_times.popleft() for _ in range(len(X))],
                                self.model.predict_proba(X)):
                    self._p.append(float(p))
                    self._last_epoch_t = t
                    if float(np.mean(self._p)) >= self.threshold:
                        if self._run_start is None:
                            self._run_start, self._run_p = t, []
                        self._run_len += 1
                        self._run_p.append(float(p))
                    else:
                        self._run_start, self._run_len, self._run_p = None, 0, []
        if self._run_start is None or self._run_len < self.min_epochs:
            return []
        conf = float(np.clip(np.mean(self._run_p), 0.5, 0.99))
        ru, uz = LABELS.get(self.code, (self.code, self.code))
        return [Event(
            code=self.code, label_ru=ru, label_uz=uz, group="ictal",
            localization=Localization(channels=list(getattr(feats, "eeg_channels", []))),
            t_start=max(0.0, self._run_start - t0), t_end=max(0.0, self._last_epoch_t - t0),
            confidence=conf,
            evidence=[EventEvidence(feature="ml_probability", value=round(conf, 3),
                                    reference=f">= {self.threshold} for >= {self.min_epochs} epochs")],
            metadata={"run_epochs": self._run_len, "detector": "learned (streaming)"},
        )]
