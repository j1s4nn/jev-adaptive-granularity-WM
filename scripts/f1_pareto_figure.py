"""F1 Pareto figure: DiT cost vs reference distance for the mode sweeps.

Left panel  = F1.3 clean-context 6-mode sweep (horizons 45/57/81).
Right panel = F1.3b clean vs noisy-context (horizons 57/81).

Reads  results/f1_mode_cost/f1_mode_cost_audit.json
       results/f1_mode_cost_noisy/f1_mode_cost_noisy_audit.json
Writes docs/figures/f1_mode_pareto.png
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(ROOT, "results", "f1_mode_cost", "f1_mode_cost_audit.json")
NOISY = os.path.join(ROOT, "results", "f1_mode_cost_noisy", "f1_mode_cost_noisy_audit.json")
OUT = os.path.join(ROOT, "docs", "figures", "f1_mode_pareto.png")

LABELS = {
    "ref_c1_1_s4": "ref c1-1 s4",
    "fine_c1_1_s1": "fine c1-1 s1",
    "fine_c1_1_s1_noisy": "fine c1-1 s1 (noisy)",
    "fine_c1_1_s2": "fine c1-1 s2",
    "coarse_c3_3_s4": "coarse c3-3 s4",
    "coarse_c3_3_s1": "coarse c3-3 s1",
    "coarse_c3_3_s1_noisy": "coarse c3-3 s1 (noisy)",
    "coarse_c3_3_s2": "coarse c3-3 s2",
    "coarse_c3_3_s2_noisy": "coarse c3-3 s2 (noisy)",
}


def load(path, l1_key):
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    rows = {}
    for mode, stats in d["by_mode"].items():
        l1 = d[l1_key][mode]["mean"]
        rows[mode] = {"dit_s": stats["dit_s"], "total_s": stats["total_s"], "ref_l1": l1}
    return rows


def scatter(ax, rows, title):
    for mode, r in rows.items():
        color = "#1f77b4" if mode.startswith("fine") else ("#d62728" if mode.startswith("coarse") else "#2ca02c")
        marker = "o" if "noisy" not in mode else "^"
        ax.scatter(r["dit_s"], r["ref_l1"], c=color, marker=marker, s=70)
        ax.annotate(LABELS.get(mode, mode), (r["dit_s"], r["ref_l1"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("DiT sampling time (s), lower = cheaper")
    ax.set_ylabel("mean decoded-pixel L1 to c1-1 s4 ref (lower = closer)")
    ax.grid(alpha=0.3)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    clean = load(CLEAN, "reference_l1_by_mode")
    noisy = load(NOISY, "ref_l1")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    scatter(axes[0], clean, "F1.3 clean-context mode sweep (45/57/81)")
    scatter(axes[1], noisy, "F1.3b clean vs noisy context (57/81)")
    axes[0].legend(handles=[
        plt.Line2D([], [], color="#2ca02c", marker="o", ls="", label="reference"),
        plt.Line2D([], [], color="#1f77b4", marker="o", ls="", label="fine c1-1"),
        plt.Line2D([], [], color="#d62728", marker="o", ls="", label="coarse c3-3 (clean)"),
        plt.Line2D([], [], color="#d62728", marker="^", ls="", label="coarse c3-3 (noisy ctx)"),
    ], fontsize=7, loc="upper right")
    fig.suptitle("F1: fine (c1-1) dominates coarse (c3-3) on both cost and reference distance", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT, dpi=160)
    print("wrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
