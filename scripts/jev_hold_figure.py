"""E-JEV-3 figure: cost vs quality for the hold-mode controllers."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(ROOT, "results", "jev_hold_mode", "jev_hold_audit.json")
CB = os.path.join(ROOT, "results", "jev_holdfine", "jev_holdfine_audit.json")
OUT = os.path.join(ROOT, "docs", "figures", "jev_hold_mode.png")


def main():
    with open(C, "r", encoding="utf-8") as fh:
        c = json.load(fh)
    with open(CB, "r", encoding="utf-8") as fh:
        cb = json.load(fh)

    points = []
    for p in ["fixed_fine", "fixed_coarse", "fixed_hold", "jev_routed", "jev_routed_nohold"]:
        e = c["by_policy"][p]
        points.append((p, e["total_s"], e["mean_reference_l1"]))
    for p in ["fixed_fine", "jev_holdfine"]:
        e = cb["by_policy"][p]
        points.append((p + " (C-b)", e["total_s"], e["mean_reference_l1"]))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name, total, ref in points:
        if "holdfine" in name:
            color, marker = "#2ca02c", "*"
        elif name.startswith("fixed_fine"):
            color, marker = "#1f77b4", "o"
        elif name.startswith("fixed_coarse"):
            color, marker = "#d62728", "o"
        elif name.startswith("fixed_hold"):
            color, marker = "#9467bd", "s"
        else:
            color, marker = "#7f7f7f", "^"
        ax.scatter(total, ref, c=color, marker=marker, s=130)
        ax.annotate(name, (total, ref), textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xlabel("total wall-clock (s), lower = cheaper")
    ax.set_ylabel("mean reference L1, lower = better")
    ax.set_title("E-JEV-3: Jev-routed 'hold' mode (static->hold, else->fine) beats fixed fine\n"
                 "on BOTH quality and cost; coarse routing does not", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print("wrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
