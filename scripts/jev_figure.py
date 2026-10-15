"""E-JEV-1 figure: AUROC of the Jev signal vs hand-crafted signals."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.join(ROOT, "results", "jev_signal", "jev_audit.json")
OUT = os.path.join(ROOT, "docs", "figures", "jev_signal_auroc.png")

SIGNALS = ["jev_difficulty", "jev_drift", "jev_motion", "motion", "warp", "complexity"]


def main():
    with open(AUDIT, "r", encoding="utf-8") as fh:
        a = json.load(fh)
    vals = [a["signals"][s]["auroc"] for s in SIGNALS]
    lo = [a["signals"][s]["ci_low"] for s in SIGNALS]
    hi = [a["signals"][s]["ci_high"] for s in SIGNALS]
    err_lo = [v - l for v, l in zip(vals, lo)]
    err_hi = [h - v for v, h in zip(vals, hi)]
    colors = ["#2ca02c" if s.startswith("jev") else "#7f7f7f" for s in SIGNALS]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = range(len(SIGNALS))
    ax.bar(x, vals, yerr=[err_lo, err_hi], capsize=4, color=colors)
    ax.axhline(0.5, color="k", ls=":", lw=1, label="chance (0.5)")
    ax.axhline(0.491, color="#d62728", ls="--", lw=1, label="Phase 5B residual (0.491)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(SIGNALS, rotation=20, fontsize=8)
    ax.set_ylabel("AUROC (coarse-unsafe label)")
    ax.set_ylim(0, 1.05)
    ax.set_title("E-JEV-1: a Jev System One signal predicts coarse-unsafe segments far better\n"
                 "than hand-crafted signals (grouped bootstrap 95% CI over 8 prompts, n=80 segments)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print("wrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
