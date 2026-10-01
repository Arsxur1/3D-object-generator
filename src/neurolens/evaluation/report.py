"""Persist evaluation results as JSON + a human-readable markdown table."""

from __future__ import annotations

import json
from pathlib import Path

from .runner import EvalResult


def _fmt(v, nd=2) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def markdown_summary(results: list[EvalResult], title: str = "Evaluation") -> str:
    lines = [f"# {title}", "",
             "| mode | records | hours | seizures | TP | FN | FP | sensitivity | FA/h | latency median s | latency p90 s |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        t = r.total
        lines.append(
            f"| {r.mode} | {t.n_records} | {t.hours:.1f} | {t.n_seizures} | {t.tp} | {t.fn} | {t.fp} "
            f"| {_fmt(t.sensitivity)} | {_fmt(t.fa_per_hour)} | {_fmt(t.latency_median_s, 1)} "
            f"| {_fmt(t.latency_p90_s, 1)} |"
        )
    for r in results:
        lines += ["", f"## Per record — {r.mode}", "",
                  "| file | seizures | TP | FN | FP | latencies s |", "|---|---|---|---|---|---|"]
        for s in r.scores:
            lat = ", ".join(f"{x:.0f}" for x in s.latencies_s) or "—"
            lines.append(f"| {s.file} | {s.n_seizures} | {s.tp} | {s.fn} | {s.fp} | {lat} |")
    return "\n".join(lines) + "\n"


def save_results(results: list[EvalResult], out_dir: str | Path, title: str = "Evaluation",
                 extra: dict | None = None) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"title": title, "results": [r.as_dict() for r in results], **(extra or {})}
    jp = out / "metrics.json"
    jp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    mp = out / "summary.md"
    mp.write_text(markdown_summary(results, title), encoding="utf-8")
    return {"json": jp, "markdown": mp}
