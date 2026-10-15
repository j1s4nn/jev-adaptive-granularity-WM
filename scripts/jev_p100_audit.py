"""E-JEV-2 (Phase B) audit: fine/coarse DiT seconds-per-forward on the run GPU.

Reads  results/jev_p100_timing/results.jsonl
Writes results/jev_p100_timing/jev_p100_audit.json
"""

import json
import os
import statistics
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, "results", "jev_p100_timing", "results.jsonl")
OUT = os.path.join(ROOT, "results", "jev_p100_timing", "jev_p100_audit.json")

MODES = ["ref_c1_1_s4", "fine_c1_1_s1", "coarse_c3_3_s4", "coarse_c3_3_s1"]


def mean(v):
    v = [x for x in v if isinstance(x, (int, float))]
    return statistics.fmean(v) if v else None


def main():
    rows = [json.loads(l) for l in open(IN, encoding="utf-8") if l.strip()]
    ok = [r for r in rows if r.get("status") == "complete"]
    gpu = ok[0]["gpu"] if ok else "unknown"

    report = {"phase": "E-JEV-2-hardware-retime", "generated_utc": datetime.now(timezone.utc).isoformat(),
              "gpu": gpu, "n": len(ok), "by_mode": {}}

    for m in MODES:
        sub = [r for r in ok if r["mode"] == m]
        report["by_mode"][m] = {
            "n": len(sub),
            "dit_s": mean([r["dit_s"] for r in sub]),
            "nfe": mean([r["nfe"] for r in sub]),
            "forwards": mean([r["forwards"] for r in sub]),
            "s_per_forward": mean([r["s_per_forward"] for r in sub]),
        }

    f = report["by_mode"]["fine_c1_1_s1"]
    c = report["by_mode"]["coarse_c3_3_s1"]
    if f["s_per_forward"] and c["s_per_forward"]:
        report["coarse_over_fine_per_forward"] = c["s_per_forward"] / f["s_per_forward"]
        report["coarse_over_fine_dit"] = c["dit_s"] / f["dit_s"]
        report["nfe_ratio_coarse_over_fine"] = c["nfe"] / f["nfe"]
        report["breakeven_per_forward_ratio"] = f["nfe"] / c["nfe"]
        report["coarse_cheaper"] = c["dit_s"] < f["dit_s"]

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("E-JEV-2 audit:", os.path.relpath(OUT, ROOT), "| GPU:", gpu)
    print(f"{'mode':<16}{'DiT(s)':>8}{'NFE':>6}{'fwd':>6}{'s/fwd':>9}")
    for m in MODES:
        b = report["by_mode"][m]
        print(f"{m:<16}{b['dit_s']:>8.2f}{b['nfe']:>6.1f}{b['forwards']:>6.1f}{b['s_per_forward']:>9.3f}")
    if "coarse_over_fine_per_forward" in report:
        print(f"\ncoarse/fine per-forward ratio = {report['coarse_over_fine_per_forward']:.2f} "
              f"(break-even {report['breakeven_per_forward_ratio']:.2f}) -> coarse cheaper: {report['coarse_cheaper']}")


if __name__ == "__main__":
    main()
