"""F4-lite audit: aggregate online-switching policy results.

Reads  results/f4lite/results.jsonl
Writes results/f4lite/f4lite_audit.json
"""

import hashlib
import json
import os
import statistics
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_JSONL = os.path.join(ROOT, "results", "f4lite", "results.jsonl")
OUT_JSON = os.path.join(ROOT, "results", "f4lite", "f4lite_audit.json")

POLICIES = ["fixed_coarse", "fixed_fine", "random50", "alternating50", "adaptive_warp"]
TIMING_FIELDS = ["dit_s", "vae_s", "total_s", "nfe", "signal_s", "peak_vram_gb"]


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
    rows = []
    with open(IN_JSONL, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("status") == "complete" and r.get("policy") in POLICIES:
                    rows.append(r)

    report = {"phase": "F4-lite-audit", "generated_utc": datetime.now(timezone.utc).isoformat(),
              "input_sha256": sha256_file(IN_JSONL), "n_rollouts": len(rows), "by_policy": {},
              "paired_vs_best_fixed": {}, "per_prompt": {}}

    # best fixed = lowest mean refL1 among fixed_coarse/fixed_fine
    fixed = {p: mean([r["mean_reference_l1"] for r in rows if r["policy"] == p]) for p in ["fixed_coarse", "fixed_fine"]}
    best_fixed = min(fixed, key=lambda k: fixed[k] if fixed[k] is not None else 1e9)
    report["fixed_policies_refL1"] = fixed
    report["best_fixed_policy"] = best_fixed

    for policy in POLICIES:
        sub = [r for r in rows if r["policy"] == policy]
        entry = {"n": len(sub), "fine_fraction_mean": mean([r["fine_fraction"] for r in sub]),
                 "switch_count_mean": mean([r["switch_count"] for r in sub]),
                 "mean_reference_l1": mean([r["mean_reference_l1"] for r in sub])}
        for f in TIMING_FIELDS:
            entry[f] = mean([r["timing"].get(f) for r in sub])
        report["by_policy"][policy] = entry

    # paired differences vs best fixed, per prompt
    for policy in POLICIES:
        if policy == best_fixed:
            continue
        diffs = []
        for idx in range(8):
            a = next((r for r in rows if r["prompt_idx"] == idx and r["policy"] == policy), None)
            b = next((r for r in rows if r["prompt_idx"] == idx and r["policy"] == best_fixed), None)
            if a and b:
                diffs.append(a["mean_reference_l1"] - b["mean_reference_l1"])
        report["paired_vs_best_fixed"][policy] = {
            "n": len(diffs), "mean_refL1_diff": mean(diffs),
            "min": min(diffs) if diffs else None, "max": max(diffs) if diffs else None,
        }

    for idx in range(8):
        sub = [r for r in rows if r["prompt_idx"] == idx]
        if sub:
            report["per_prompt"][str(idx)] = {"prompt": sub[0]["prompt"], "stratum": sub[0]["stratum"],
                                              "refL1": {r["policy"]: r["mean_reference_l1"] for r in sub},
                                              "dit_s": {r["policy"]: r["timing"]["dit_s"] for r in sub}}

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("F4-lite audit:", os.path.relpath(OUT_JSON, ROOT))
    print(f"best fixed policy: {best_fixed} (refL1 {fixed[best_fixed]:.4f})")
    print(f"{'policy':<16}{'fine_frac':>10}{'switches':>9}{'refL1':>9}{'dit_s':>8}{'vae_s':>8}{'total_s':>9}{'signal_s':>9}")
    for p in POLICIES:
        e = report["by_policy"][p]
        print(f"{p:<16}{e['fine_fraction_mean']:>10.2f}{e['switch_count_mean']:>9.2f}{e['mean_reference_l1']:>9.4f}"
              f"{e['dit_s']:>8.1f}{e['vae_s']:>8.1f}{e['total_s']:>9.1f}{e['signal_s']:>9.2f}")
    print()
    for p, d in report["paired_vs_best_fixed"].items():
        print(f"  {p} vs {best_fixed}: mean refL1 diff = {d['mean_refL1_diff']:+.4f} (range {d['min']:+.4f}..{d['max']:+.4f})")


if __name__ == "__main__":
    main()
