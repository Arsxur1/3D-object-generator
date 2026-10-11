"""Sleep staging in the clinician-facing outputs (TZ §11): bilingual RU/UZ summary
text and a hypnogram figure, built from ``result_json["sleep_staging"]``.

The text states the validation basis, so the reader knows how far the numbers can
be trusted: which derivations and model were used, and the held-out agreement with
experts (Cohen's kappa) measured for that model in the pre-registered checks.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..contracts.report import Bilingual  # noqa: E402

# held-out agreement with experts per model (docs/increment13_report.md, docs/increment14_report.md)
_VALIDATION = {
    "midline": Bilingual(
        ru="модель проверена на ПСГ здоровых взрослых (каппа 0.71 с экспертами)",
        uz="model sog‘lom kattalar PSG’sida tekshirilgan (ekspertlar bilan kappa 0.71)"),
    "parasagittal": Bilingual(
        ru="модель для клинических производных проверена у пациентов с нарушениями сна (каппа 0.71); "
           "при нарушении поведения в фазе REM точность ниже",
        uz="klinik hosilalar modeli uyqu buzilishi bo‘lgan bemorlarda tekshirilgan (kappa 0.71); "
           "REM fazasidagi xulq buzilishida aniqlik pastroq"),
}
_ORDER = ("W", "REM", "N1", "N2", "N3")  # hypnogram rows, top to bottom
_LEVEL = {s: i for i, s in enumerate(_ORDER)}


def _fmt(v, unit: str = "") -> str:
    return "—" if v is None else f"{float(v):.0f}{unit}"


def sleep_text(sleep: dict) -> Bilingual:
    """One-paragraph bilingual summary of the hypnogram."""
    s = sleep["summary"]
    pct = s["stage_percent_of_sleep"]
    eff = s.get("sleep_efficiency")
    eff_s = "—" if eff is None else f"{100 * float(eff):.0f}%"
    stages_ru = ", ".join(f"{k} {float(pct.get(k, 0)):.0f}%" for k in ("N1", "N2", "N3", "REM"))
    val = _VALIDATION.get(sleep.get("derivations", "midline"), _VALIDATION["midline"])
    ru = (f"Автоматическое стадирование сна (AASM, эпохи {int(sleep['epoch_s'])} с): время в записи "
          f"{_fmt(s['time_in_bed_min'], ' мин')}, сон {_fmt(s['total_sleep_min'], ' мин')}, эффективность {eff_s}, "
          f"латентность сна {_fmt(s['sleep_latency_min'], ' мин')}, латентность REM {_fmt(s['rem_latency_min'], ' мин')} (до первого REM ≥ 1.5 мин); "
          f"доли стадий: {stages_ru}. Производные — {sleep.get('derivations', 'midline')}; {val.ru}. "
          f"Стадия N1 определяется ненадёжно. Требует проверки специалистом.")
    uz = (f"Uyquni avtomatik bosqichlash (AASM, {int(sleep['epoch_s'])} s davrlar): yozuv vaqti "
          f"{_fmt(s['time_in_bed_min'], ' daq')}, uyqu {_fmt(s['total_sleep_min'], ' daq')}, samaradorlik {eff_s}, "
          f"uyqu latentligi {_fmt(s['sleep_latency_min'], ' daq')}, REM latentligi {_fmt(s['rem_latency_min'], ' daq')} (≥ 1.5 daq birinchi REM gacha); "
          f"bosqichlar ulushi: {stages_ru}. Hosilalar — {sleep.get('derivations', 'midline')}; {val.uz}. "
          f"N1 bosqichi ishonchsiz aniqlanadi. Mutaxassis tekshiruvi zarur.")
    return Bilingual(ru=ru, uz=uz)


def plot_hypnogram(sleep: dict, out_path: str | Path) -> Path:
    """Step plot of stages over time (hours from recording start)."""
    stages = sleep["hypnogram"]
    ep = float(sleep["epoch_s"])
    y = np.array([_LEVEL.get(s, 0) for s in stages], dtype=float)
    t = np.arange(len(y) + 1) * ep / 3600.0
    fig, ax = plt.subplots(figsize=(10, 2.6), dpi=110)
    ax.step(t, np.append(y, y[-1] if len(y) else 0), where="post", color="#2b4c7e", linewidth=1.2)
    rem = y == _LEVEL["REM"]
    for i in np.where(rem)[0]:
        ax.plot([t[i], t[i + 1]], [y[i], y[i]], color="#c0392b", linewidth=3)
    ax.set_yticks(range(len(_ORDER)))
    ax.set_yticklabels(_ORDER)
    ax.invert_yaxis()
    ax.set_xlabel("часы от начала записи / yozuv boshidan soatlar")
    ax.set_xlim(0, t[-1] if len(t) else 1)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Гипнограмма / Gipnogramma", fontsize=10)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
