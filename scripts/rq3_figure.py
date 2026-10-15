"""RQ3 figure: switch count, AUH regret, and transition cost by controller."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.join(ROOT, "results", "rq3_controller", "rq3_audit.json")
OUT = os.path.join(ROOT, "docs", "figures", "rq3_controller_ablation.png")

CONTROLLERS = ["threshold_only", "hysteresis_only", "recovery_only", "full", "threshold_only_noisy", "full_noisy"]


def main():
    with open(AUDIT, "r", encoding="utf-8") as fh:
        a = json.load(fh)
    bc = a["by_controller"]
    labels = CONTROLLERS
    sw = [bc[c]["switch_count_mean"] for c in labels]
    auh = [bc[c]["auh_reference_l1"] for c in labels]
    swadj = [bc[c]["switch_adjacent_l1"] if bc[c]["switch_adjacent_l1"] is not None else 0 for c in labels]
    intadj = [bc[c]["interior_adjacent_l1"] for c in labels]

    x = range(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728", "#8c564b", "#e377c2"]

    axes[0].bar(x, sw, color=colors)
    axes[0].set_title("Mean switch count (lower = less oscillation)", fontsize=9)
    axes[1].bar(x, auh, color=colors)
    axes[1].set_title("Area-under-horizon reference regret (lower = better)", fontsize=9)
    axes[2].bar([i - 0.2 for i in x], swadj, width=0.4, label="at switch boundary", color="#d62728")
    axes[2].bar([i + 0.2 for i in x], intadj, width=0.4, label="interior", color="#7f7f7f")
    axes[2].set_title("Adjacent-frame L1: switch vs interior", fontsize=9)
    axes[2].legend(fontsize=7)

    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels([c.replace("_", "\n") for c in labels], fontsize=6.5)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("RQ3: recovery+hysteresis reduces switches without worsening regret; switching has a transition cost", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT, dpi=160)
    print("wrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
