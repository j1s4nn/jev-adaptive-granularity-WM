"""F4-lite figure: policy cost vs independent quality.

Reads  results/f4lite/f4lite_audit.json, results/f4lite/f4lite_quality.json
Writes docs/figures/f4lite_policy_pareto.png
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.join(ROOT, "results", "f4lite", "f4lite_audit.json")
QUAL = os.path.join(ROOT, "results", "f4lite", "f4lite_quality.json")
OUT = os.path.join(ROOT, "docs", "figures", "f4lite_policy_pareto.png")

POLICIES = ["fixed_coarse", "fixed_fine", "random50", "alternating50", "adaptive_warp"]


def main():
    with open(AUDIT, "r", encoding="utf-8") as fh:
        audit = json.load(fh)
    with open(QUAL, "r", encoding="utf-8") as fh:
        qual = json.load(fh)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for policy in POLICIES:
        a = audit["by_policy"][policy]
        q = qual["by_policy"][policy]
        color = "#d62728" if policy == "adaptive_warp" else ("#1f77b4" if policy == "fixed_fine" else "#7f7f7f")
        marker = "*" if policy in ("adaptive_warp", "fixed_fine") else "o"
        axes[0].scatter(a["dit_s"], q["subject_consistency"], c=color, marker=marker, s=110)
        axes[0].annotate(policy, (a["dit_s"], q["subject_consistency"]), textcoords="offset points", xytext=(6, 4), fontsize=7)
        axes[1].scatter(a["total_s"], q["subject_consistency"], c=color, marker=marker, s=110)
        axes[1].annotate(policy, (a["total_s"], q["subject_consistency"]), textcoords="offset points", xytext=(6, 4), fontsize=7)
    for ax, xlabel in zip(axes, ["DiT sampling time (s)", "total wall-clock (s)"]):
        ax.set_xlabel(xlabel + ", lower = cheaper")
        ax.set_ylabel("CLIP subject consistency, higher = better")
        ax.grid(alpha=0.3)
    axes[0].set_title("Mode-dependent cost (DiT) vs independent quality", fontsize=9)
    axes[1].set_title("Full cost (incl. repeated VAE decode) vs independent quality", fontsize=9)
    fig.suptitle("F4-lite: fixed fine dominates; adaptive switching is slower and not better", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT, dpi=160)
    print("wrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
