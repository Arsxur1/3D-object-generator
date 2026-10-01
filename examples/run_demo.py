"""End-to-end demo (TZ §18.4).

Generates the deterministic synthetic EDF (if missing), runs the full pipeline
in mode A with the double-banana montage, prints the RU/UZ protocol, and writes
JSON + protocol + plots to ``out_demo/``.

    python examples/run_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "data" / "synthetic"))

from make_synthetic_edf import make_synthetic_edf  # noqa: E402

from neurolens.contracts.report import OperatingMode  # noqa: E402
from neurolens.contracts.signal import ClinicalContext, PatientInfo  # noqa: E402
from neurolens.pipeline.pipeline import Pipeline  # noqa: E402


def main() -> int:
    out_dir = REPO / "out_demo"
    edf = out_dir / "demo.edf"
    if not edf.exists():
        make_synthetic_edf(edf)
        print(f"[demo] wrote synthetic EDF: {edf}")

    patient = PatientInfo(age_years=55)
    context = ClinicalContext(
        sedatives=["propofol"],
        antiseizure_meds=[],
        temperature_c=36.5,
        clinical_question="Кома в ОРИТ: оценка энцефалопатии и судорог",
        symptoms=["altered_mental_status"],
    )

    pipe = Pipeline(provider_pref="auto")  # Anthropic if key present, else deterministic
    out = pipe.analyze_file(
        edf, mode=OperatingMode.A_DECISION_SUPPORT,
        montage_name="double_banana", patient=patient, context=context,
    )

    print(out.protocol_text)
    print("\n--- События / Hodisalar ---")
    for e in out.detection.events:
        print(f"  [{e.code}] {e.label_ru}  t={e.t_start:.1f}-{e.t_end:.1f}s  conf={e.confidence:.2f}")
    print("\n--- Каузальные рёбра / Sabab qirralari ---")
    for ed in out.graph.edges:
        print(f"  {ed.source} -> {ed.target}  [{ed.rule_id}] ({ed.physiology.value})")

    paths = pipe.save_outputs(out, out_dir)
    print("\n--- Файлы вывода / Chiqish fayllari ---")
    for k, p in paths.items():
        print(f"  {k}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
