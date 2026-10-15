"""E-JEV-3b (Phase C-b) audit: Jev hold-else-fine controller.

Reads  results/jev_holdfine/results.jsonl
Writes results/jev_holdfine/jev_holdfine_audit.json
"""

import hashlib
import json
import os
import statistics
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, "results", "jev_holdfine", "results.jsonl")
OUT = os.path.join(ROOT, "results", "jev_holdfine", "jev_holdfine_audit.json")
POLICIES = ["fixed_fine", "jev_holdfine"]


def mean(v):
    v = [x for x in v if isinstance(x, (int, float))]
    return statistics.fmean(v) if v else None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def main():
    rows = [json.loads(l) for l in open(IN, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("status") == "complete" and r.get("policy") in POLICIES]

    report = {"phase": "E-JEV-3b-holdfine-audit", "generated_utc": datetime.now(timezone.utc).isoformat(),
              "input_sha256": sha256_file(IN), "n_rollouts": len(rows), "by_policy": {}, "paired": {}}

    for p in POLICIES:
        sub = [r for r in rows if r["policy"] == p]
        report["by_policy"][p] = {
            "n": len(sub),
            "mean_reference_l1": mean([r["mean_reference_l1"] for r in sub]),
            "terminal_reference_l1": mean([r["terminal_reference_l1"] for r in sub]),
            "dit_s": mean([r["timing"]["dit_s"] for r in sub]),
            "vae_s": mean([r["timing"]["vae_s"] for r in sub]),
            "total_s": mean([r["timing"]["total_s"] for r in sub]),
            "forwards": mean([r["timing"]["forwards"] for r in sub]),
            "hold_count": mean([r["hold_count"] for r in sub]),
            "fine_count": mean([r["fine_count"] for r in sub]),
        }

    dr, dt, dd, dref = [], [], [], []
    for pi in range(8):
        a = next((r for r in rows if r["prompt_idx"] == pi and r["policy"] == "fixed_fine"), None)
        b = next((r for r in rows if r["prompt_idx"] == pi and r["policy"] == "jev_holdfine"), None)
        if a and b:
            dr.append(b["mean_reference_l1"] - a["mean_reference_l1"])
            dt.append(b["timing"]["total_s"] - a["timing"]["total_s"])
            dd.append(b["timing"]["dit_s"] - a["timing"]["dit_s"])
            dref.append(b["timing"]["vae_s"] - a["timing"]["vae_s"])
    report["paired"] = {"jev_holdfine_vs_fixed_fine": {
        "regret_diff_mean": mean(dr), "total_s_diff_mean": mean(dt), "dit_s_diff_mean": mean(dd),
        "vae_s_diff_mean": mean(dref), "n": len(dr)}}

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("E-JEV-3b audit:", os.path.relpath(OUT, ROOT))
    print(f"{'policy':<16}{'hold':>6}{'fine':>6}{'meanRef':>9}{'termRef':>9}{'DiT(s)':>8}{'VAE(s)':>8}{'total(s)':>9}")
    for p in POLICIES:
        e = report["by_policy"][p]
        print(f"{p:<16}{e['hold_count']:>6.1f}{e['fine_count']:>6.1f}{e['mean_reference_l1']:>9.4f}"
              f"{e['terminal_reference_l1']:>9.4f}{e['dit_s']:>8.1f}{e['vae_s']:>8.1f}{e['total_s']:>9.1f}")
    v = report["paired"]["jev_holdfine_vs_fixed_fine"]
    print(f"\njev_holdfine vs fixed_fine: regret {v['regret_diff_mean']:+.4f} | total {v['total_s_diff_mean']:+.1f}s | DiT {v['dit_s_diff_mean']:+.1f}s | VAE {v['vae_s_diff_mean']:+.1f}s")


if __name__ == "__main__":
    main()
