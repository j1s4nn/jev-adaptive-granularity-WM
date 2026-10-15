"""A100-01 / A1 audit: aggregate the A100 mode-cost sweep and compare to T4.

Reads:
  results/a100_mode_cost/results.jsonl
  results/a100_mode_cost/a100_quality_classical.json   (if present)
  results/a100_mode_cost/a100_quality_clip.json        (if present)
Writes:
  results/a100_mode_cost/a100_mode_cost_audit.json

Compares per-mode DiT/VAE/total/NFE/VRAM/refL1 means on the A100 and reports the
coarse-vs-fine per-forward ratio against the T4 reference (3.01) and the
break-even ratio (2.33). Independent of any T4 result files (never modifies them).
"""

import json
import os
import statistics
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "a100_mode_cost")
RESULTS_JSONL = os.path.join(OUT_DIR, "results.jsonl")
CLASSICAL_JSON = os.path.join(OUT_DIR, "a100_quality_classical.json")
CLIP_JSON = os.path.join(OUT_DIR, "a100_quality_clip.json")
AUDIT_JSON = os.path.join(OUT_DIR, "a100_mode_cost_audit.json")

MODES = ["ref_c1_1_s4", "fine_c1_1_s1", "fine_c1_1_s2",
         "coarse_c3_3_s4", "coarse_c3_3_s2", "coarse_c3_3_s1"]
BREAK_EVEN = 2.33
T4_RATIO = 3.01


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def load_results():
    rows = []
    if not os.path.exists(RESULTS_JSONL):
        return rows
    with open(RESULTS_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if item.get("status") == "complete":
                rows.append(item)
    return rows


def aggregate(rows):
    agg = {}
    for mode in MODES:
        dit, vae, total, nfe, vram, fwd, refl1 = [], [], [], [], [], [], []
        for r in rows:
            m = r.get("modes", {}).get(mode)
            if not m:
                continue
            dit.append(m.get("dit_s"))
            vae.append(m.get("vae_s"))
            total.append(m.get("total_s"))
            nfe.append(m.get("nfe"))
            vram.append(m.get("peak_vram_gb"))
            # true forward count: blocks * steps-per-chunk, from step_counts
            sc = m.get("step_counts") or []
            fwd.append(sum(sc) + (m.get("blocks") or 0))
            for obs in r.get("observations", []):
                key = f"{mode}_ref_l1"
                if key in obs:
                    refl1.append(obs[key])
        agg[mode] = {
            "n_cases": sum(1 for r in rows if mode in r.get("modes", {})),
            "dit_s": mean(dit),
            "vae_s": mean(vae),
            "total_s": mean(total),
            "nfe": mean(nfe),
            "peak_vram_gb": mean(vram),
            "forwards": mean(fwd),
            "ref_l1": mean(refl1),
            "s_per_forward": (mean(dit) / mean(fwd)) if (mean(dit) and mean(fwd)) else None,
            "s_per_nfe": (mean(dit) / mean(nfe)) if (mean(dit) and mean(nfe)) else None,
        }
    return agg


def attach_quality(audit):
    for path, key in ((CLASSICAL_JSON, "classical"), (CLIP_JSON, "clip")):
        if os.path.exists(path):
            try:
                data = json.load(open(path, encoding="utf-8"))
                audit["quality_" + key] = data.get("by_mode")
            except Exception as e:
                audit["quality_" + key] = {"error": repr(e)}


def main():
    rows = load_results()
    audit = {
        "phase": "A100-A1-mode-cost-audit",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_complete_cases": len(rows),
        "break_even_ratio": BREAK_EVEN,
        "t4_reference_ratio": T4_RATIO,
        "by_mode": aggregate(rows),
    }
    attach_quality(audit)

    fine = audit["by_mode"].get("fine_c1_1_s1") or {}
    coarse = audit["by_mode"].get("coarse_c3_3_s1") or {}
    verdict = "INSUFFICIENT_DATA"
    ratio = None
    if fine.get("s_per_forward") and coarse.get("s_per_forward"):
        ratio = coarse["s_per_forward"] / fine["s_per_forward"]
        if ratio < BREAK_EVEN:
            verdict = (f"TRADE-OFF EXISTS on A100: coarse/fine per-forward ratio {ratio:.2f} "
                       f"< break-even {BREAK_EVEN} -> coarse cheaper in wall-clock DiT time")
        else:
            verdict = (f"NO trade-off: coarse/fine per-forward ratio {ratio:.2f} "
                       f">= break-even {BREAK_EVEN}")
    audit["coarse_over_fine_per_forward_ratio"] = ratio
    audit["verdict"] = verdict

    with open(AUDIT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=str)

    print("A100 A1 audit written:", os.path.relpath(AUDIT_JSON, ROOT))
    print(f"{'mode':<18}{'dit_s':>9}{'vae_s':>9}{'total_s':>10}{'nfe':>7}{'fwd':>7}{'vram':>8}{'refL1':>9}{'s/fwd':>9}")
    for mode in MODES:
        a = audit["by_mode"][mode]
        def fmt(v, d=2):
            return f"{v:.{d}f}" if isinstance(v, (int, float)) else "  -"
        print(f"{mode:<18}{fmt(a['dit_s']):>9}{fmt(a['vae_s']):>9}{fmt(a['total_s']):>10}"
              f"{fmt(a['nfe'],1):>7}{fmt(a['forwards'],1):>7}{fmt(a['peak_vram_gb']):>8}"
              f"{fmt(a['ref_l1'],4):>9}{fmt(a['s_per_forward'],3):>9}")
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
