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
    pipe = Pipeline(provider_pref=args.provider, calibration_file=getattr(args, "calibration", None))
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


def _monitor(args) -> int:
    from .outputs.json_out import save_json
    from .pipeline.config_loader import load_configs
    from .realtime.monitor import RealtimeMonitor
    from .realtime.stream import EdfReplaySource

    cfg = load_configs()
    if args.window is not None:
        cfg.realtime.window_s = args.window
    if args.step is not None:
        cfg.realtime.step_s = args.step
    source = EdfReplaySource(args.file, chunk_s=cfg.realtime.step_s, realtime=args.realtime)
    monitor = RealtimeMonitor(config=cfg, montage_name=args.montage, out_dir=args.out)
    summary = monitor.run(source)

    print(f"=== cEEG мониторинг / cEEG monitoring: {summary.source} ===")
    print(f"Окон/Oynalar: {summary.n_windows}  окно/step: {summary.window_s}/{summary.step_s}с")
    print(f"Тревоги/Alarms: {summary.n_alarms_raised} (подавлено/suppressed: {summary.n_alarms_suppressed})"
          f"  {summary.alarms_per_hour}/час (бюджет {summary.false_alarm_budget_per_hour})"
          f"  в пределах бюджета: {summary.within_budget}")
    print(f"Латентность/окно: mean {summary.latency_ms_mean} мс, p95 {summary.latency_ms_p95} мс"
          f"  real-time осуществимо: {summary.realtime_feasible}")
    if summary.seizure_burden:
        sb = summary.seizure_burden
        print(f"Судорожная нагрузка: {sb.n_seizures} эп., {sb.total_seizure_time_s}с "
              f"({sb.seizure_fraction*100:.0f}%), статус: {sb.status_epilepticus_suspected}")
    for al in summary.alarms:
        frag = f"  фрагмент: {al.fragment_path}" if al.fragment_path else ""
        print(f"  ⚠ [{al.type.value}] t={al.t_start:.0f}–{al.t_end:.0f}с "
              f"conf={al.confidence} — {al.message.ru}{frag}")

    if args.out:
        path = save_json(summary.model_dump(mode="json"), Path(args.out) / "monitor_summary.json")
        print(f"\nСводка/Xulosa: {path}")
    return 0


def _calibrate(args) -> int:
    from .calibration.from_feedback import build_calibration

    cal = build_calibration(args.feedback, out_path=args.out, min_samples=args.min_samples)
    print(f"Калибровка/Kalibrovka: default T={cal.default}, "
          f"per-code={len(cal.temperatures)} кодов")
    if cal.metrics:
        m = cal.metrics
        print(f"  ECE до/после: {m.get('ece_before')} -> {m.get('ece_after')}  (n={m.get('n_samples')})")
    if args.out:
        print(f"  Сохранено/Saqlangan: {args.out}")
    return 0


def _serve(args) -> int:
    try:
        import uvicorn
    except Exception:
        print("FastAPI/uvicorn not installed. Install with: pip install -e '.[api]'")
        return 1
    from .api.app import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


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
    a.add_argument("--calibration", default=None, help="Confidence calibration JSON file.")
    a.set_defaults(func=_analyze)

    d = sub.add_parser("demo", help="Generate a synthetic EDF and analyze it.")
    d.add_argument("--out", default="out_demo")
    d.add_argument("--provider", default="auto", choices=["auto", "anthropic", "deterministic"])
    d.set_defaults(func=_demo)

    m = sub.add_parser("monitor", help="Stream an EDF as real-time cEEG with alarms.")
    m.add_argument("file")
    m.add_argument("--montage", default="double_banana")
    m.add_argument("--window", type=float, default=None, help="Analysis window (s).")
    m.add_argument("--step", type=float, default=None, help="Advance between analyses (s).")
    m.add_argument("--out", default=None, help="Output dir for alarm fragments + summary.")
    m.add_argument("--realtime", action="store_true", help="Pace replay to wall-clock.")
    m.set_defaults(func=_monitor)

    cal = sub.add_parser("calibrate", help="Fit confidence calibration from a feedback log.")
    cal.add_argument("--feedback", required=True, help="Path to the feedback JSONL log.")
    cal.add_argument("--out", default="configs/calibration.json")
    cal.add_argument("--min-samples", type=int, default=10, dest="min_samples")
    cal.set_defaults(func=_calibrate)

    sv = sub.add_parser("serve", help="Run the REST API (requires the 'api' extra).")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=_serve)

    s = sub.add_parser("export-schemas", help="Export JSON Schemas from contracts.")
    s.add_argument("dir", nargs="?", default=None)
    s.set_defaults(func=_export_schemas)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
