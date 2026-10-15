"""RQ4 figure: uncertainty-benefit relationship (signal vs fine_gap)."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.join(ROOT, "results", "rq4_monotonicity", "rq4_audit.json")
OUT = os.path.join(ROOT, "docs", "figures", "rq4_monotonicity.png")


def main():
    with open(AUDIT, "r", encoding="utf-8") as fh:
        a = json.load(fh)
    pts = a["points"]
    sig = [p["signal"] for p in pts]
    gap = [p["fine_gap"] for p in pts]
    bins = a["bins"]

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.scatter(sig, gap, s=18, alpha=0.6, c="#1f77b4", label="segment (fine_gap = regret_fine - regret_coarse)")
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.plot([b["signal_mean"] for b in bins], [b["fine_gap_mean"] for b in bins],
            "-o", color="#d62728", lw=2, label="tercile bin mean")
    ax.set_xlabel("pre-generation uncertainty signal (warp), higher = more uncertain")
    ax.set_ylabel("fine - coarse reference regret (negative = fine better)")
    ax.set_title(f"RQ4: no monotone relationship (Spearman rho = {a['spearman_rho']:+.3f}); "
                 f"fine benefit peaks at intermediate uncertainty", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print("wrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
