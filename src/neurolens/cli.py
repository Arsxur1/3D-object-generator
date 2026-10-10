"""NeuroLens command-line interface.

    neurolens analyze <file.edf> [--mode A|B] [--montage ...] [--out DIR] [--provider ...]
    neurolens demo   [--out DIR]          # generate synthetic EDF and analyze it
    neurolens export-schemas [DIR]        # write JSON Schemas from the contracts
    neurolens physionet list|fetch ...    # open annotated EEG (CHB-MIT, Siena)
    neurolens evaluate --subjects ...     # sensitivity / FA/h / latency vs experts
    neurolens tune --train ... --test ... # tune thresholds on train, report held-out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .contracts.report import OperatingMode
from .contracts.signal import ClinicalContext, PatientInfo


def _load_cfg(overrides: str | None = None):
    from .pipeline.config_loader import apply_overrides, load_configs

    cfg = load_configs()
    return apply_overrides(cfg, overrides) if overrides else cfg


def _analyze(args) -> int:
    from .pipeline.pipeline import Pipeline

    mode = OperatingMode.B_AUTONOMOUS if args.mode.upper() == "B" else OperatingMode.A_DECISION_SUPPORT
    pma = getattr(args, "pma", None)
    patient = (
        PatientInfo(age_years=args.age, postmenstrual_age_weeks=pma)
        if (args.age is not None or pma is not None) else None
    )
    context = ClinicalContext(
        sedatives=args.sedative or [],
        clinical_question=args.question,
    )
    pipe = Pipeline(
        config=_load_cfg(getattr(args, "overrides", None)),
        provider_pref=args.provider, calibration_file=getattr(args, "calibration", None),
    )
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

    cfg = _load_cfg(args.overrides)
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


def _subjects(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _items(default_db: str, text: str, cache):
    """Subjects may name their database ("siena:PN00"), so one evaluation can
    pool populations (paediatric CHB-MIT + adult Siena)."""
    from .evaluation.runner import load_items

    by_db: dict[str, list[str]] = {}
    for tok in _subjects(text):
        db, _, sub = tok.rpartition(":")
        by_db.setdefault(db or default_db, []).append(sub)
    items = []
    for db, subs in by_db.items():
        items += load_items(db, subs, cache)
    return items


def _physionet(args) -> int:
    from .datasets.physionet import client_for

    client = client_for(args.db, args.cache)
    if args.subjects in ("dev", "test") and args.db == "helsinki":
        args.subjects = ",".join(client.split()[args.subjects])  # rule-based split (pre-registered)
    if args.action == "list":
        subs = _subjects(args.subjects) if args.subjects else client.subjects()
        for sub in subs:
            anns = client.annotations(sub) if args.subjects else []
            if not anns:
                print(sub)
                continue
            for a in anns:
                mark = "cached" if client.is_cached(a.file) else ""
                sz = ", ".join(f"{s.onset_s:.0f}-{s.offset_s:.0f}s" for s in a.seizures)
                print(f"{a.file:28s} seizures={len(a.seizures)} {sz:30s} {mark}")
        if args.db == "helsinki":
            from .datasets.helsinki import CITATION, LICENSE
        else:
            from .datasets.physionet import DATABASES

            CITATION, LICENSE = DATABASES[args.db].citation, DATABASES[args.db].license
        print(f"\n{CITATION}\nLicense: {LICENSE}")
        return 0

    files: list[str] = []
    for sub in _subjects(args.subjects):
        anns = client.annotations(sub)
        files += [a.file for a in anns if a.has_seizure]
        # one recording per neonate in the Helsinki set: seizure-free neonates are part of the cohort
        files += [a.file for a in anns if not a.has_seizure][: None if args.db == "helsinki" else args.n_free]
    todo = [f for f in files if not client.is_cached(f)]
    est_gb = sum((client.remote_size(f) or 42_000_000) for f in todo) / 1e9
    print(f"{len(files)} files ({est_gb:.1f} GB to download) -> {client.cache}")
    if est_gb > args.max_gb:
        print(f"Refusing: exceeds --max-gb {args.max_gb}")
        return 2
    res = client.fetch_many(files, workers=args.workers,
                            on_done=lambda r, e: print(("ok  " if e is None else f"ERR {e} ") + r, flush=True))
    return 0 if all(not isinstance(v, Exception) for v in res.values()) else 1


def _evaluate(args) -> int:
    from .evaluation.report import save_results
    from .evaluation.runner import evaluate_offline, evaluate_realtime, load_items, prepare_records

    cfg = _load_cfg(args.overrides)
    items = _items(args.db, args.subjects, args.cache)
    if not items:
        print("No cached records — run `neurolens physionet fetch` first.")
        return 2
    recs = prepare_records(items, cfg, args.feature_cache, progress=print, workers=args.workers)
    results = [evaluate_offline(recs, cfg.thresholds)]
    if args.realtime:
        results.append(evaluate_realtime(items, cfg, progress=print))
    for r in results:
        t = r.total
        print(f"[{r.mode}] {t.n_records} rec, {t.hours:.1f} h: sensitivity {t.tp}/{t.n_seizures}"
              f" = {t.sensitivity:.2f}, FA/h {t.fa_per_hour:.2f}, latency median "
              f"{t.latency_median_s}s")
    if args.out:
        paths = save_results(results, args.out, title=f"{args.db} {args.subjects}",
                             extra={"overrides": args.overrides})
        print(f"Saved: {paths['json']}, {paths['markdown']}")
    return 0


def _fmt_score(name: str, r) -> str:
    t = r.total
    return (f"{name:26s}: sens {t.tp}/{t.n_seizures} = {t.sensitivity:.2f}, "
            f"FA/h {t.fa_per_hour:.2f}, latency median {t.latency_median_s}s")


def _tune(args) -> int:
    from .evaluation.report import save_results
    from .evaluation.runner import evaluate_offline, prepare_records
    from .evaluation.tune import SELECTORS, fit_ictal_calibration, grid_search, write_overrides

    selectors = _subjects(args.selectors)
    bad = [x for x in selectors if x not in SELECTORS]
    if not selectors or bad:
        print(f"--selectors must be from {SELECTORS}; got {args.selectors!r}")
        return 2
    primary = selectors[0]

    cfg = _load_cfg()
    train_items = _items(args.db, args.train, args.cache)
    test_items = _items(args.db, args.test, args.cache)
    if not train_items or not test_items:
        print("Need cached train and test records — run `neurolens physionet fetch` first.")
        return 2

    # --- offline (whole-record) operating point ---
    train = prepare_records(train_items, cfg, args.feature_cache, progress=print, workers=args.workers)
    test = prepare_records(test_items, cfg, args.feature_cache, progress=print, workers=args.workers)
    res = grid_search(train, cfg.thresholds, fa_target=args.fa_target)
    chosen = {sel: res.select(sel) for sel in selectors}
    test_default = evaluate_offline(test, cfg.thresholds)
    test_default.mode = "offline-default"
    results = [test_default]
    provenance = {
        "train_subjects": _subjects(args.train), "test_subjects": _subjects(args.test),
        "default_database": args.db, "fa_target_per_hour": args.fa_target,
        "selectors": selectors, "primary_selector": primary,
        "offline": {"n_trials": len(res.trials), "test_default": test_default.total.as_dict()},
    }
    for sel, trial in chosen.items():
        r = evaluate_offline(test, cfg.thresholds.model_copy(update=trial.params))
        r.mode = f"offline-{sel}"
        results.append(r)
        provenance["offline"][sel] = {"params": trial.params, "train": trial.score.as_dict(),
                                      "test": r.total.as_dict()}
        print(f"offline [{sel}] train {trial.params}: sens {trial.score.sensitivity:.2f} "
              f"FA/h {trial.score.fa_per_hour:.2f} ({len(res.trials)} trials)")
    best = chosen[primary]
    tuned = cfg.thresholds.model_copy(update=best.params)
    cal, cal_metrics = fit_ictal_calibration(train, tuned)
    provenance["calibration"] = cal_metrics
    print(f"calibration ({primary}): {cal_metrics}")
    rt_params = None
    top = {"offline": [{"params": t.params, **t.score.as_dict()} for t in res.top(15)]}

    # --- streaming operating point (tuned separately on replayed monitor traces) ---
    if args.realtime:
        from .evaluation.realtime_replay import build_traces, evaluate_replay, with_realtime_params
        from .evaluation.tune import grid_search_realtime

        tr_train = build_traces(train_items, cfg, args.feature_cache, progress=print, workers=args.workers)
        tr_test = build_traces(test_items, cfg, args.feature_cache, progress=print, workers=args.workers)
        rres = grid_search_realtime(tr_train, cfg.thresholds, cfg.realtime, fa_target=args.fa_target)
        rt_default = evaluate_replay(tr_test, cfg.thresholds, cfg.realtime)
        rt_offline_th = evaluate_replay(tr_test, tuned, cfg.realtime)
        rt_default.mode, rt_offline_th.mode = "realtime-default", f"realtime-offline-{primary}-thresholds"
        results += [rt_default, rt_offline_th]
        provenance["realtime"] = {"n_trials": len(rres.trials), "test_default": rt_default.total.as_dict(),
                                  "test_offline_thresholds": rt_offline_th.total.as_dict()}
        for sel in selectors:
            trial = rres.select(sel)
            r = evaluate_replay(tr_test, cfg.thresholds, with_realtime_params(cfg.realtime, trial.params))
            r.mode = f"realtime-{sel}"
            results.append(r)
            provenance["realtime"][sel] = {"params": trial.params, "train": trial.score.as_dict(),
                                           "test": r.total.as_dict()}
            print(f"realtime [{sel}] train {trial.params}: sens {trial.score.sensitivity:.2f} "
                  f"FA/h {trial.score.fa_per_hour:.2f} ({len(rres.trials)} trials)")
        rt_params = rres.select(primary).params
        top["realtime"] = [{"params": t.params, **t.score.as_dict()} for t in rres.top(15)]

    for r in results:
        print(_fmt_score(f"held-out {r.mode}", r))
    if args.write:
        write_overrides(args.overrides_out, best.params, provenance, realtime_params=rt_params)
        cal.save(args.calibration_out)
        print(f"Wrote {args.overrides_out} and {args.calibration_out}")
    if args.out:
        paths = save_results(results, args.out, title=f"Held-out {args.test} (tuned on {args.train})",
                             extra={"provenance": provenance, "top_trials": top})
        print(f"Saved: {paths['json']}, {paths['markdown']}")
    return 0


def _sleep_cap14(args) -> int:
    """Increment 14: clinical stager vs v1 on the never-used CAP test records."""
    import hashlib
    import json

    from .datasets.capsleep import select_test_records_14
    from .datasets.physionet import PhysioNetClient
    from .evaluation.sleep_eval import cap_headers, predict_night, prepare_cap_night, save, score
    from .layer4_detect.sleep_staging import MulticlassTreeModel, rule_based_stages

    def load_frozen(path):
        fz = json.loads(open(path, encoding="utf-8").read())
        if hashlib.sha256(open(fz["model"]["path"], "rb").read()).hexdigest() != fz["model"]["sha256"]:
            raise RuntimeError(f"{fz['model']['path']} does not match the frozen sha256")
        return fz

    fz14 = load_frozen(args.frozen14)
    fz13 = load_frozen(args.frozen)
    client = PhysioNetClient("capsleep", args.cache)
    names = select_test_records_14(cap_headers(client))
    nights = []
    for i, n in enumerate(names, 1):
        nights.append(prepare_cap_night(client, n, args.feature_cache))
        print(f"[{i}/{len(names)}] prepared {n}", flush=True)
    clin = MulticlassTreeModel.load(fz14["model"]["path"])
    v1 = MulticlassTreeModel.load(fz13["model"]["path"])
    pc = [predict_night(clin, n) for n in nights]
    result = {"clinical": score(nights, pc), "v1": score(nights, [predict_night(v1, n) for n in nights]),
              "rule_based": score(nights, [rule_based_stages(n.F, n.names) for n in nights]), "records": names}
    grp = lambda n: n.key.rstrip("0123456789")
    by_group = {g: score([n for n in nights if grp(n) == g], [p for n, p in zip(nights, pc) if grp(n) == g])
                for g in sorted({grp(n) for n in nights})}
    result["by_group"] = {g: {k: v[k] for k in ("kappa", "accuracy", "macro_f1", "n_nights")} for g, v in by_group.items()}
    path = save(result, args.out)
    for k in ("clinical", "v1", "rule_based"):
        v = result[k]
        print(f"{k}: kappa {v['kappa']:.3f} accuracy {v['accuracy']:.3f} macro-F1 {v['macro_f1']:.3f}")
    if args.assess:
        from .evaluation.prereg14 import assess, markdown_report

        a = assess(result, fz14, by_group)
        path.with_name("prereg_assessment.json").write_text(json.dumps(a, indent=1, ensure_ascii=False), encoding="utf-8")
        path.with_name("prereg_report.md").write_text(markdown_report(a), encoding="utf-8")
        print(json.dumps({"hypotheses": a["hypotheses"], "decision": a["decision"]}, indent=1))
    print(f"Saved: {path}")
    return 0


def _sleep_cap(args) -> int:
    """Increment 13b: frozen stager on the pre-registered CAP Sleep Database subset."""
    import hashlib
    import json
    from pathlib import Path

    import numpy as np

    from .datasets.capsleep import select_records
    from .datasets.physionet import PhysioNetClient
    from .evaluation.sleep_eval import cap_headers, predict_night, prepare_cap_night, save, score
    from .layer4_detect.sleep_staging import MulticlassTreeModel, rule_based_stages

    frozen = json.loads(open(args.frozen, encoding="utf-8").read())
    if hashlib.sha256(open(frozen["model"]["path"], "rb").read()).hexdigest() != frozen["model"]["sha256"]:
        raise RuntimeError("sleep model does not match the frozen sha256")
    client = PhysioNetClient("capsleep", args.cache)
    names = select_records(cap_headers(client))
    model = MulticlassTreeModel.load(frozen["model"]["path"])
    nights = []
    for i, n in enumerate(names, 1):
        nights.append(prepare_cap_night(client, n, args.feature_cache))
        print(f"[{i}/{len(names)}] prepared {n}", flush=True)
    learned = [predict_night(model, n) for n in nights]
    rule = [rule_based_stages(n.F, n.names) for n in nights]
    result = {"learned": score(nights, learned), "rule_based": score(nights, rule), "records": names}
    groups = sorted({n.key.rstrip("0123456789") for n in nights})
    by_group = {g: score([n for n in nights if n.key.rstrip("0123456789") == g],
                         [p for n, p in zip(nights, learned) if n.key.rstrip("0123456789") == g]) for g in groups}
    result["by_group"] = {g: {k: v[k] for k in ("kappa", "accuracy", "macro_f1", "n_nights")} for g, v in by_group.items()}
    path = save(result, args.out)
    for k in ("learned", "rule_based"):
        v = result[k]
        print(f"{k}: kappa {v['kappa']:.3f} accuracy {v['accuracy']:.3f} macro-F1 {v['macro_f1']:.3f}")
    if args.assess:
        from .evaluation.prereg13b import assess, markdown_report

        a = assess(result, by_group)
        path.with_name("prereg_assessment.json").write_text(json.dumps(a, indent=1, ensure_ascii=False), encoding="utf-8")
        path.with_name("prereg_report.md").write_text(markdown_report(a), encoding="utf-8")
        print(json.dumps({"hypotheses": a["hypotheses"], "decision": a["decision"]}, indent=1))
    print(f"Saved: {path}")
    return 0


def _sleep(args) -> int:
    """Sleep-EDF: fetch a split, or evaluate a frozen stager on it (+ pre-registered analysis)."""
    import json

    from .datasets.sleepedf import SleepEDFClient

    if args.action == "cap":
        return _sleep_cap(args)
    if args.action == "cap14":
        return _sleep_cap14(args)
    client = SleepEDFClient(args.cache)
    recs = client.split()[args.split]
    if args.action == "fetch":
        res = client.fetch_records(recs, workers=args.workers,
                                   on_done=lambda r, e: print(("ok  " if e is None else f"ERR {e} ") + r, flush=True))
        return 0 if all(not isinstance(v, Exception) for v in res.values()) else 1
    from .evaluation.sleep_eval import evaluate, prepare_nights, save
    from .layer4_detect.sleep_staging import MulticlassTreeModel

    frozen = json.loads(open(args.frozen, encoding="utf-8").read())
    import hashlib

    if hashlib.sha256(open(frozen["model"]["path"], "rb").read()).hexdigest() != frozen["model"]["sha256"]:
        raise RuntimeError("sleep model does not match the frozen sha256")
    nights = prepare_nights(client, recs, args.feature_cache, workers=args.workers, progress=print)
    result = evaluate(nights, MulticlassTreeModel.load(frozen["model"]["path"]))
    path = save(result, args.out)
    for k, v in result.items():
        print(f"{k}: kappa {v['kappa']:.3f} accuracy {v['accuracy']:.3f} macro-F1 {v['macro_f1']:.3f} F1 {v['f1']}")
    if args.assess:
        from .evaluation.prereg13 import assess, markdown_report

        a = assess(result, frozen)
        path.with_name("prereg_assessment.json").write_text(json.dumps(a, indent=1, ensure_ascii=False), encoding="utf-8")
        path.with_name("prereg_report.md").write_text(markdown_report(a), encoding="utf-8")
        print(json.dumps({"hypotheses": a["hypotheses"], "decision": a["decision"]}, indent=1))
    print(f"Saved: {path}")
    return 0


def _ml_eval(args) -> int:
    import json

    from .evaluation.ml_eval import run_and_save

    cfg = _load_cfg()
    items = _items(args.db, args.test, args.cache)
    if not items:
        print("No cached test records — run `neurolens physionet fetch` first.")
        return 2
    out = run_and_save(items, cfg, args.frozen, args.out, compare_path=args.compare,
                       feature_cache=args.feature_cache, workers=args.workers,
                       realtime=not args.offline_only, progress=print,
                       stream_cache=args.stream_cache or None)
    for r in out["results"]:
        print(_fmt_score(r.mode, r))
    if args.assess:
        metrics = json.loads(out["paths"]["json"].read_text(encoding="utf-8"))
        if args.analysis == "prereg12":
            from .evaluation.prereg12 import assess, markdown_report

            amb = {a.file: [(s.onset_s, s.offset_s) for s in a.ambiguous] for a, _ in items}
            a = assess(metrics, ambiguous=amb)
        else:
            if args.analysis == "prereg11":
                from .evaluation.prereg11 import assess, markdown_report
            else:
                from .evaluation.prereg10 import assess, markdown_report
            a = assess(metrics)
        path = out["paths"]["json"].with_name("prereg_assessment.json")
        path.write_text(json.dumps(a, indent=1, ensure_ascii=False), encoding="utf-8")
        out["paths"]["json"].with_name("prereg_report.md").write_text(markdown_report(a), encoding="utf-8")
        print(json.dumps({"hypotheses": a["hypotheses"], "decision": a["decision"],
                          "macro_sensitivity": a["macro_sensitivity"]}, indent=1))
    print(f"Saved: {out['paths']['json']}")
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
    a.add_argument("--age", type=float, default=None, help="Age in years.")
    a.add_argument("--pma", type=float, default=None, help="Postmenstrual age (weeks) — enables neonatal analysis.")
    a.add_argument("--sedative", action="append", default=None)
    a.add_argument("--question", default=None)
    a.add_argument("--calibration", default=None, help="Confidence calibration JSON file.")
    a.add_argument("--overrides", default=None, help="Threshold override YAML (e.g. configs/thresholds.physionet.yaml).")
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
    m.add_argument("--overrides", default=None, help="Threshold override YAML.")
    m.set_defaults(func=_monitor)

    cal = sub.add_parser("calibrate", help="Fit confidence calibration from a feedback log.")
    cal.add_argument("--feedback", required=True, help="Path to the feedback JSONL log.")
    cal.add_argument("--out", default="configs/calibration.json")
    cal.add_argument("--min-samples", type=int, default=10, dest="min_samples")
    cal.set_defaults(func=_calibrate)

    def _data_args(x):
        x.add_argument("--db", default="chbmit", choices=["chbmit", "siena", "helsinki"])
        x.add_argument("--cache", default=None, help="Data cache dir (default data/physionet or $NEUROLENS_DATA).")

    ph = sub.add_parser("physionet", help="List/fetch open annotated EEG from PhysioNet.")
    ph.add_argument("action", choices=["list", "fetch"])
    _data_args(ph)
    ph.add_argument("--subjects", default=None, help="Comma-separated, e.g. chb01,chb03.")
    ph.add_argument("--n-free", type=int, default=2, dest="n_free",
                    help="Seizure-free files per subject to include (FA/h estimate).")
    ph.add_argument("--max-gb", type=float, default=2.0, dest="max_gb")
    ph.add_argument("--workers", type=int, default=6)
    ph.set_defaults(func=_physionet)

    ev = sub.add_parser("evaluate", help="Score seizure detection against expert labels.")
    _data_args(ev)
    ev.add_argument("--subjects", required=True, help="e.g. chb05,chb08 or chb05,siena:PN00")
    ev.add_argument("--overrides", default=None)
    ev.add_argument("--realtime", action="store_true", help="Also score the streaming monitor's alarms.")
    ev.add_argument("--feature-cache", default="data/physionet/.features", dest="feature_cache")
    ev.add_argument("--workers", type=int, default=4, help="Parallel processes for feature preparation.")
    ev.add_argument("--out", default=None)
    ev.set_defaults(func=_evaluate)

    tu = sub.add_parser("tune", help="Tune ictal thresholds on train subjects; report held-out.")
    _data_args(tu)
    tu.add_argument("--train", required=True)
    tu.add_argument("--test", required=True)
    tu.add_argument("--fa-target", type=float, default=1.0, dest="fa_target")
    tu.add_argument("--feature-cache", default="data/physionet/.features", dest="feature_cache")
    tu.add_argument("--realtime", action="store_true",
                    help="Also tune the streaming monitor (separate operating point, replayed traces).")
    tu.add_argument("--workers", type=int, default=4,
                    help="Parallel processes for feature preparation and trace building.")
    tu.add_argument("--selectors", default="strict",
                    help="Comma-separated operating-point rules (strict, robust); the first is "
                         "primary and is written by --write.")
    tu.add_argument("--write", action="store_true", help="Write overrides + calibration files.")
    tu.add_argument("--overrides-out", default="configs/thresholds.physionet.yaml", dest="overrides_out")
    tu.add_argument("--calibration-out", default="configs/calibration.physionet.json", dest="calibration_out")
    tu.add_argument("--out", default=None, help="Report dir (metrics.json + summary.md).")
    tu.set_defaults(func=_tune)

    me = sub.add_parser("ml-eval", help="Evaluate frozen learned detectors vs thresholds on held-out records.")
    _data_args(me)
    me.add_argument("--test", required=True, help="Subjects, e.g. chb04,siena:PN07")
    me.add_argument("--frozen", default="docs/preregistration_increment10_frozen.json")
    me.add_argument("--feature-cache", default="data/physionet/.features", dest="feature_cache")
    me.add_argument("--workers", type=int, default=3)
    me.add_argument("--offline-only", action="store_true", dest="offline_only")
    me.add_argument("--assess", action="store_true", help="Apply the pre-registered analysis (--analysis).")
    me.add_argument("--analysis", default="prereg10", choices=["prereg10", "prereg11", "prereg12"])
    me.add_argument("--compare", default=None, help="Second frozen set scored as *-cmp-A (e.g. previous models).")
    me.add_argument("--stream-cache", default="data/physionet/.features/streams", dest="stream_cache",
                    help="Save each record's monitor replay as it finishes; an interrupted run resumes "
                         "('' disables).")
    me.add_argument("--out", default="out_eval/ml_eval")
    me.set_defaults(func=_ml_eval)

    sl = sub.add_parser("sleep", help="Sleep-EDF: fetch a split or evaluate a frozen sleep stager.")
    sl.add_argument("action", choices=["fetch", "eval", "cap", "cap14"],
                    help="fetch/eval: Sleep-EDF split; cap: clinical transfer test (CAP Sleep Database)")
    sl.add_argument("--split", choices=["dev", "test"], default="test")
    sl.add_argument("--cache", default=None)
    sl.add_argument("--frozen", default="docs/preregistration_increment13_frozen.json")
    sl.add_argument("--feature-cache", default="data/physionet/.features/sleep", dest="feature_cache")
    sl.add_argument("--workers", type=int, default=3)
    sl.add_argument("--frozen14", default="docs/preregistration_increment14_frozen.json")
    sl.add_argument("--assess", action="store_true")
    sl.add_argument("--out", default="out_eval/sleep")
    sl.set_defaults(func=_sleep)

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
