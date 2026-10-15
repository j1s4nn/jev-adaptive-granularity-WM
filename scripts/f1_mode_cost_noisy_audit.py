"""F1.3b noisy-context cost audit."""

import hashlib
import json
import os
import statistics
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_JSONL = os.path.join(ROOT, "results", "f1_mode_cost_noisy", "results.jsonl")
OUT_JSON = os.path.join(ROOT, "results", "f1_mode_cost_noisy", "f1_mode_cost_noisy_audit.json")

MODES = ["ref_c1_1_s4", "fine_c1_1_s1", "fine_c1_1_s1_noisy", "coarse_c3_3_s1", "coarse_c3_3_s1_noisy", "coarse_c3_3_s2_noisy"]
FIELDS = ["dit_s", "vae_s", "total_s", "nfe", "forward_count", "peak_vram_gb", "blocks"]


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
                if r.get("status") == "complete":
                    rows.append(r)

    report = {"phase": "F1.3b-noisy-cost-audit", "generated_utc": datetime.now(timezone.utc).isoformat(),
              "input_sha256": sha256_file(IN_JSONL), "n_cases": len(rows), "by_mode": {}, "by_horizon": {}, "ref_l1": {}}
    for mode in MODES:
        report["by_mode"][mode] = {f: mean([r["modes"][mode][f] for r in rows if mode in r.get("modes", {})]) for f in FIELDS}
        l1 = [o[f"{mode}_ref_l1"] for r in rows for o in r.get("observations", []) if f"{mode}_ref_l1" in o]
        report["ref_l1"][mode] = {"n": len(l1), "mean": mean(l1)}
    for h in [57, 81]:
        sub = [r for r in rows if r.get("horizon") == h]
        report["by_horizon"][str(h)] = {m: {"dit_s": mean([r["modes"][m]["dit_s"] for r in sub if m in r.get("modes", {})]),
                                           "forward_count": mean([r["modes"][m]["forward_count"] for r in sub if m in r.get("modes", {})])} for m in MODES}
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("F1.3b audit:", os.path.relpath(OUT_JSON, ROOT))
    print(f"{'mode':<22}{'DiT(s)':>8}{'VAE(s)':>8}{'total(s)':>9}{'NFE':>6}{'fwd':>6}{'refL1':>9}")
    for m in MODES:
        b = report["by_mode"][m]
        print(f"{m:<22}{b['dit_s']:>8.2f}{b['vae_s']:>8.1f}{b['total_s']:>9.1f}{b['nfe']:>6.1f}{b['forward_count']:>6.1f}{report['ref_l1'][m]['mean']:>9.4f}")


if __name__ == "__main__":
    main()
