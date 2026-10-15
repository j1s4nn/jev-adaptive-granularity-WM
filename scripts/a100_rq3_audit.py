"""A100-01 / A3 audit: recovery+hysteresis controller ablation (RQ3).

Reads:  results/a100_rq3/results.jsonl
Writes: results/a100_rq3/a100_rq3_audit.json

Per controller (6): mean switch count, fine fraction, area-under-horizon (AUH) and
terminal reference regret, and switch-vs-interior transition cost. RQ3 verdict:
does `full` (recovery+hysteresis) reduce switches vs `threshold_only` without worse
AUH/terminal regret, including under the noisy variants? Never modifies T4 artifacts.
"""

import json
import os
import statistics
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRNAME = sys.argv[1] if len(sys.argv) > 1 else "a100_rq3"
OUT_DIR = os.path.join(ROOT, "results", DIRNAME)
RAW = os.path.join(OUT_DIR, "results.jsonl")
AUDIT = os.path.join(OUT_DIR, "a100_rq3_audit.json")

CONTROLLERS = ["threshold_only", "hysteresis_only", "recovery_only", "full",
               "threshold_only_noisy", "full_noisy"]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def load():
    rows = []
    if not os.path.exists(RAW):
        return rows
    with open(RAW, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                it = json.loads(line)
            except Exception:
                continue
            if it.get("status") == "complete" and it.get("controller") in CONTROLLERS:
                rows.append(it)
    return rows


def agg(rows):
    out = {}
    for c in CONTROLLERS:
        rs = [r for r in rows if r["controller"] == c]
        out[c] = {
            "n": len(rs),
            "switch_count": mean([r.get("switch_count") for r in rs]),
            "fine_fraction": mean([r.get("fine_fraction") for r in rs]),
            "auh_reference_l1": mean([r.get("auh_reference_l1") for r in rs]),
            "terminal_reference_l1": mean([r.get("terminal_reference_l1") for r in rs]),
            "switch_adjacent_l1": mean([r.get("switch_adjacent_l1") for r in rs]),
            "interior_adjacent_l1": mean([r.get("interior_adjacent_l1") for r in rs]),
        }
    return out


def main():
    rows = load()
    by = agg(rows)
    audit = {"phase": "A100-A3-rq3-audit", "generated_utc": datetime.now(timezone.utc).isoformat(),
             "n_complete": len(rows), "by_controller": by}

    v = {}
    t, f = by.get("threshold_only", {}), by.get("full", {})
    tn, fn = by.get("threshold_only_noisy", {}), by.get("full_noisy", {})
    if t.get("switch_count") is not None and f.get("switch_count") is not None:
        v["switches_threshold_to_full"] = [t["switch_count"], f["switch_count"]]
        v["switches_reduction_real"] = (t["switch_count"] - f["switch_count"]) if t["switch_count"] is not None else None
    if tn.get("switch_count") is not None and fn.get("switch_count") is not None:
        v["switches_noisy_reduction"] = (tn["switch_count"] - fn["switch_count"]) if tn["switch_count"] is not None else None
    for k in ("auh_reference_l1", "terminal_reference_l1"):
        if t.get(k) is not None and f.get(k) is not None:
            v[f"{k}_full_minus_threshold"] = f[k] - t[k]
    audit["verdict"] = v

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(AUDIT, "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2, default=str)

    print("A100 A3 audit written:", os.path.relpath(AUDIT, ROOT))
    print(f"{'controller':<22}{'n':>4}{'switches':>10}{'fine%':>8}{'AUH':>9}{'terminal':>10}{'sw_adj':>9}{'int_adj':>9}")
    for c in CONTROLLERS:
        a = by[c]
        def fm(x, d=4):
            return f"{x:.{d}f}" if isinstance(x, (int, float)) else "  -"
        print(f"{c:<22}{a['n']:>4}{fm(a['switch_count'],2):>10}{fm(a['fine_fraction'],2):>8}"
              f"{fm(a['auh_reference_l1']):>9}{fm(a['terminal_reference_l1']):>10}"
              f"{fm(a['switch_adjacent_l1']):>9}{fm(a['interior_adjacent_l1']):>9}")
    print("VERDICT:", json.dumps(v))


if __name__ == "__main__":
    main()
