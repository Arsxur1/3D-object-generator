"""NeuroLens command-line interface.

    neurolens analyze <file.edf> [--mode A|B] [--montage ...] [--out DIR] [--provider ...]
    neurolens demo   [--out DIR]          # generate synthetic EDF and analyze it
    neurolens export-schemas [DIR]        # write JSON Schemas from the contracts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .contracts.report import OperatingMode
from .contracts.signal import ClinicalContext, PatientInfo


def _analyze(args) -> int:
    from .pipeline.pipeline import Pipeline

    mode = OperatingMode.B_AUTONOMOUS if args.mode.upper() == "B" else OperatingMode.A_DECISION_SUPPORT
    patient = PatientInfo(age_years=args.age) if args.age is not None else None
    context = ClinicalContext(
        sedatives=args.sedative or [],
        clinical_question=args.question,
    )
    pipe = Pipeline(provider_pref=args.provider)
    out = pipe.analyze_file(
        args.file, mode=mode, montage_name=args.montage, patient=patient, context=context
    )
    print(out.protocol_text)
    if args.out:
        paths = pipe.save_outputs(out, args.out)
        print("\nСохранено / Saqlangan:")
        for k, p in paths.items():
            print(f"  {k}: {p}")
    return 0


def _demo(args) -> int:
    edf = _generate_demo_edf(Path(args.out) / "demo.edf")
    ns = argparse.Namespace(
        file=str(edf), mode="A", montage="double_banana", out=args.out,
        provider=args.provider, age=55, sedative=["propofol"], question="кома, оценка энцефалопатии",
    )
    return _analyze(ns)


def _generate_demo_edf(out_path: Path) -> Path:
    """Locate and run the synthetic EDF generator (data/synthetic)."""
    import importlib.util

    gen_path = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "make_synthetic_edf.py"
    spec = importlib.util.spec_from_file_location("make_synthetic_edf", gen_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod.make_synthetic_edf(out_path)


def _export_schemas(args) -> int:
    from .contracts.export_schemas import main as export_main

    return export_main([args.dir] if args.dir else [])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="neurolens", description="EEG interpretation (NeuroLens).")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="Analyze an EDF file.")
    a.add_argument("file")
    a.add_argument("--mode", default="A", choices=["A", "B", "a", "b"])
    a.add_argument("--montage", default="double_banana")
    a.add_argument("--out", default=None, help="Output directory for JSON/protocol/plots.")
    a.add_argument("--provider", default="auto", choices=["auto", "anthropic", "deterministic"])
    a.add_argument("--age", type=float, default=None)
    a.add_argument("--sedative", action="append", default=None)
    a.add_argument("--question", default=None)
    a.set_defaults(func=_analyze)

    d = sub.add_parser("demo", help="Generate a synthetic EDF and analyze it.")
    d.add_argument("--out", default="out_demo")
    d.add_argument("--provider", default="auto", choices=["auto", "anthropic", "deterministic"])
    d.set_defaults(func=_demo)

    s = sub.add_parser("export-schemas", help="Export JSON Schemas from contracts.")
    s.add_argument("dir", nargs="?", default=None)
    s.set_defaults(func=_export_schemas)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
