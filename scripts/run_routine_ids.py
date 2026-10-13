"""Run the routine-EEG harness on a fixed list of NMT records (increment 18).

Uses only APIs present in both the current code and commit 286bb2a, so the same
recordings can be scored by either pipeline:

    PYTHONPATH=<checkout>/src python scripts/run_routine_ids.py IDS.json OUT.json CACHE [WORKERS]
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _one(item):
    rid, cache = item
    from neurolens.datasets.nmt import NMTClient
    from neurolens.evaluation.routine_eval import run_record

    c = NMTClient(cache)
    rec = next(r for r in c.labels() if r.record_id == rid)
    try:
        return run_record(c.fetch(c.member_of(rec)), rec)
    except Exception as exc:  # noqa: BLE001
        return {"record_id": rid, "abnormal": rec.abnormal, "error": repr(exc)}


def main() -> int:
    ids = json.loads(Path(sys.argv[1]).read_text())["record_ids"]
    out, cache = Path(sys.argv[2]), sys.argv[3]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    from neurolens.evaluation.routine_eval import save

    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_one, [(i, cache) for i in ids]))
    errors = [r for r in rows if "error" in r]
    s = save([r for r in rows if "error" not in r], out, {"errors": errors, "ids_file": sys.argv[1]})
    print(json.dumps({k: s[k] for k in ("n_normal", "n_abnormal", "impression_specificity",
                                          "impression_sensitivity")}), "errors", len(errors))
    return 0


if __name__ == "__main__":
    sys.exit(main())
