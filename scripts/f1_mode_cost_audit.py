"""F1.3 mode-cost audit.

Summarizes results/f1_mode_cost/results.jsonl into a mode cost/quality table
and writes results/f1_mode_cost/f1_mode_cost_audit.json.
"""

import hashlib
import json
import os
import statistics
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_JSONL = os.path.join(ROOT, "results", "f1_mode_cost", "results.jsonl")
OUT_JSON = os.path.join(ROOT, "results", "f1_mode_cost", "f1_mode_cost_audit.json")

MODES = ["ref_c1_1_s4", "fine_c1_1_s1", "fine_c1_1_s2", "coarse_c3_3_s4", "coarse_c3_3_s2", "coarse_c3_3_s1"]
FIELDS = ["dit_s", "vae_s", "total_s", "nfe", "peak_vram_gb", "blocks"]


def mean(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.fmean(values) if values else None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    rows = []
    with open(IN_JSONL, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if rec.get("status") == "complete":
                    rows.append(rec)

    report = {
        "phase": "F1.3-mode-cost-audit",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": {"path": "results/f1_mode_cost/results.jsonl", "sha256": sha256_file(IN_JSONL)},
        "n_cases": len(rows),
        "by_mode": {},
        "by_horizon_mode": {},
        "reference_l1_by_mode": {},
        "nfe_by_mode": {},
    }

    for mode in MODES:
        report["by_mode"][mode] = {
            field: mean([r["modes"][mode][field] for r in rows if mode in r.get("modes", {})])
            for field in FIELDS
        }
        l1s = []
        for r in rows:
            for obs in r.get("observations", []):
                key = f"{mode}_ref_l1"
                if key in obs:
                    l1s.append(obs[key])
        report["reference_l1_by_mode"][mode] = {"n": len(l1s), "mean": mean(l1s), "min": min(l1s) if l1s else None, "max": max(l1s) if l1s else None}
        nfes = [r["modes"][mode]["nfe"] for r in rows if mode in r.get("modes", {})]
        report["nfe_by_mode"][mode] = {"mean": mean(nfes), "values": sorted(set(nfes))}

    for horizon in [45, 57, 81]:
        sub = [r for r in rows if r.get("horizon") == horizon]
        report["by_horizon_mode"][str(horizon)] = {
            mode: {
                "dit_s": mean([r["modes"][mode]["dit_s"] for r in sub if mode in r.get("modes", {})]),
                "vae_s": mean([r["modes"][mode]["vae_s"] for r in sub if mode in r.get("modes", {})]),
                "total_s": mean([r["modes"][mode]["total_s"] for r in sub if mode in r.get("modes", {})]),
                "nfe": mean([r["modes"][mode]["nfe"] for r in sub if mode in r.get("modes", {})]),
            }
            for mode in MODES
        }

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("F1.3 mode-cost audit written:", os.path.relpath(OUT_JSON, ROOT))
    print(f"{'mode':<18}{'DiT(s)':>8}{'VAE(s)':>8}{'total(s)':>9}{'NFE':>6}{'VRAM(GB)':>10}{'refL1':>9}")
    for mode in MODES:
        b = report["by_mode"][mode]
        l1 = report["reference_l1_by_mode"][mode]["mean"]
        print(
            f"{mode:<18}{b['dit_s']:>8.2f}{b['vae_s']:>8.1f}{b['total_s']:>9.1f}"
            f"{b['nfe']:>6.1f}{b['peak_vram_gb']:>10.2f}{(l1 if l1 is not None else float('nan')):>9.4f}"
        )
    print()
    print("NFE by mode:", {m: report["nfe_by_mode"][m]["values"] for m in MODES})


if __name__ == "__main__":
    main()
