"""A100-01 / A2 audit: aggregate the A100 online-policy demo and apply F2LITE-01.

Reads:  results/a100_online/results.jsonl
Writes: results/a100_online/a100_online_audit.json

Per-policy means over prompts: fine_fraction, hold count, switch_count,
mean_reference_l1 (internal diagnostic), DiT/VAE/total seconds, NFE.
Applies the F2LITE-01 §3 criterion (>=5% relative quality at equal cost, or
>=15% lower DiT at equal quality, and must beat the random same-budget control).
Never modifies T4 artifacts.
"""

import json
import os
import statistics
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DIRNAME = sys.argv[1] if len(sys.argv) > 1 else "a100_online_v2"
OUT_DIR = os.path.join(ROOT, "results", _DIRNAME)
RAW = os.path.join(OUT_DIR, "results.jsonl")
AUDIT = os.path.join(OUT_DIR, "a100_online_audit.json")

POLICIES = ["fixed_coarse", "fixed_fine", "fixed_hold", "random50", "alternating50", "adaptive_warp"]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def load_rows():
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
            if it.get("status") == "complete" and it.get("policy") in POLICIES:
                rows.append(it)
    return rows


def agg(rows):
    out = {}
    for pol in POLICIES:
        rs = [r for r in rows if r["policy"] == pol]
        t = [r.get("timing", {}) for r in rs]
        out[pol] = {
            "n": len(rs),
            "fine_fraction": mean([r.get("fine_fraction") for r in rs]),
            "hold_count": mean([r.get("hold_count") for r in rs]),
            "switch_count": mean([r.get("switch_count") for r in rs]),
            "mean_reference_l1": mean([r.get("mean_reference_l1") for r in rs]),
            "dit_s": mean([x.get("dit_s") for x in t]),
            "vae_s": mean([x.get("vae_s") for x in t]),
            "signal_s": mean([x.get("signal_s") for x in t]),
            "total_s": mean([x.get("total_s") for x in t]),
            "nfe": mean([x.get("nfe") for x in t]),
            "peak_vram_gb": mean([x.get("peak_vram_gb") for x in t]),
        }
    return out


def main():
    rows = load_rows()
    by = agg(rows)
    audit = {"phase": "A100-A2-online-audit", "generated_utc": datetime.now(timezone.utc).isoformat(),
             "n_complete": len(rows), "by_policy": by}

    ff = by.get("fixed_fine", {})
    fc = by.get("fixed_coarse", {})
    fh = by.get("fixed_hold", {})
    rnd = by.get("random50", {})
    ad = by.get("adaptive_warp", {})
    best_fixed = min([p for p in (ff, fc, fh) if p.get("total_s")], key=lambda p: p["total_s"], default=None)

    verdict = {}
    if ff.get("mean_reference_l1") and ad.get("mean_reference_l1"):
        q_rel = (ff["mean_reference_l1"] - ad["mean_reference_l1"]) / ff["mean_reference_l1"]
        dit_rel = (ff["dit_s"] - ad["dit_s"]) / ff["dit_s"] if ff.get("dit_s") and ad.get("dit_s") else None
        verdict["adaptive_vs_fixed_fine_quality_rel"] = q_rel
        verdict["adaptive_vs_fixed_fine_dit_rel"] = dit_rel
        beats_random = (ad.get("mean_reference_l1") is not None and rnd.get("mean_reference_l1") is not None
                        and ad["mean_reference_l1"] <= rnd["mean_reference_l1"])
        passes = (q_rel >= 0.05 and (dit_rel is None or dit_rel >= -0.05)) or (dit_rel is not None and dit_rel >= 0.15 and q_rel >= -0.05)
        verdict["beats_random_same_budget"] = bool(beats_random)
        verdict["f2lite01_pass"] = bool(passes and beats_random)
        verdict["note"] = ("Internal-reference (diagnostic) preliminary verdict; the primary independent "
                           "CLIP/classical endpoint must be computed on the A100 online videos before a final claim.")
    audit["verdict"] = verdict

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(AUDIT, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=str)

    print("A100 A2 audit written:", os.path.relpath(AUDIT, ROOT))
    print(f"{'policy':<16}{'n':>4}{'fine':>6}{'hold':>6}{'sw':>5}{'refL1':>9}{'dit_s':>8}{'vae_s':>8}{'total_s':>9}{'nfe':>7}")
    for pol in POLICIES:
        a = by[pol]
        def f(v, d=3):
            return f"{v:.{d}f}" if isinstance(v, (int, float)) else "  -"
        print(f"{pol:<16}{a['n']:>4}{f(a['fine_fraction'],2):>6}{f(a['hold_count'],1):>6}{f(a['switch_count'],2):>5}"
              f"{f(a['mean_reference_l1'],4):>9}{f(a['dit_s'],2):>8}{f(a['vae_s'],2):>8}{f(a['total_s'],2):>9}{f(a['nfe'],1):>7}")
    print("VERDICT:", json.dumps(verdict))


if __name__ == "__main__":
    main()
