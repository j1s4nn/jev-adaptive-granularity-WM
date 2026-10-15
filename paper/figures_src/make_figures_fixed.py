# Regenerate a compact, non-overlapping subset of the paper figures:
#   architecture.png, fig_cost_pareto.*, fig_hw_scaling.*, fig_auroc.*, fig_phase3_ci.*
# Same audited data as make_figures.py; only layout / sizing changed.
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = r"C:\Users\jisan\OneDrive\Desktop\research\idea_05_adaptive_granularity_WM\04_Experiment"
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "font.family": "serif",
    "axes.titlesize": 8.3, "axes.labelsize": 7.8,
    "legend.fontsize": 6.4, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.grid": True, "grid.alpha": 0.28, "axes.axisbelow": True,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})
C_FINE, C_COARSE, C_REF, C_JEV, C_HOLD, C_HC = "#1f77b4", "#d62728", "#2ca02c", "#2ca02c", "#9467bd", "#7f7f7f"

def load(p):
    with open(os.path.join(ROOT, p), "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(p):
    rows = []
    with open(os.path.join(ROOT, p), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

# =====================================================================
# Fig 1: compact architecture (Jev highlighted), single clean flow
# =====================================================================
def make_architecture():
    fig, ax = plt.subplots(figsize=(7.6, 2.55), dpi=300)
    ax.set_xlim(0, 15.2); ax.set_ylim(0.6, 7.5); ax.axis("off")

    def box(x, y, w, h, title, sub, fc, ec="#25324a", lw=1.2, tf=8.0, sf=6.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.10", fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
                fontsize=tf, weight="bold", color="#14213d")
        if sub:
            ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                    fontsize=sf, color="#33415c", linespacing=1.15)

    def arrow(x1, y1, x2, y2, ls="-", color="#34495e", lw=1.3, ms=10):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=ms, lw=lw, color=color, linestyle=ls,
                     shrinkA=0, shrinkB=0))

    yt, ht = 5.30, 1.60
    box(0.15, yt, 2.80, ht, "Prompt + context", "text + prev\nlatent frame", "#d9ecff")
    box(3.25, yt, 2.45, ht, "Feature state", "BLIP caption,\nmotion, index", "#e9e0ff")
    box(6.00, yt, 2.60, ht, "Jev oracle", "difficulty / drift /\nmotion class", "#ffe7c2",
        ec="#c47f00", lw=2.2, tf=8.6)
    box(8.90, yt, 2.60, ht, "Controller", "threshold +\nhysteresis + recovery", "#d9f2df")
    box(11.80, yt, 3.25, ht, "Mode selector", "fine c1-1 |\ncoarse c3-3 | hold", "#ffd9df")

    for xa, xb in [(2.95, 3.25), (5.70, 6.00), (8.60, 8.90), (11.50, 11.80)]:
        arrow(xa, yt + ht / 2, xb, yt + ht / 2)

    box(0.15, 1.45, 14.85, 1.55, "Causal-rCM causal rollout",
        "decoded frames  +  independent audit (CLIP subject/text, optical flow)  +  decision record",
        "#f2f4f7", tf=8.2, sf=6.5)

    arrow(13.40, yt, 12.40, 3.00)
    ax.add_patch(FancyArrowPatch((1.40, 3.00), (1.40, yt), arrowstyle="-|>",
                 mutation_scale=9, lw=1.1, color="#8a8f98", linestyle="--",
                 shrinkA=0, shrinkB=0))
    ax.text(1.58, 4.15, "next context", ha="left", va="center", fontsize=6.0,
            color="#6b7280", rotation=90)

    fig.savefig(os.path.join(FIG, "architecture.png"))
    fig.savefig(os.path.join(FIG, "architecture.pdf"))
    plt.close(fig)

# =====================================================================
# Fig 2: cost Pareto, T4 vs A100  (no legend box, short non-overlapping labels)
# =====================================================================
def make_cost_pareto():
    f1 = load(r"results/f1_mode_cost/f1_mode_cost_audit.json")
    a1 = load(r"results/a100_mode_cost/a100_mode_cost_audit.json")
    order = ["ref_c1_1_s4", "fine_c1_1_s1", "fine_c1_1_s2",
             "coarse_c3_3_s4", "coarse_c3_3_s2", "coarse_c3_3_s1"]
    short = {"ref_c1_1_s4": "ref s4", "fine_c1_1_s1": "fine s1", "fine_c1_1_s2": "fine s2",
             "coarse_c3_3_s4": "coarse s4", "coarse_c3_3_s2": "coarse s2", "coarse_c3_3_s1": "coarse s1"}
    off_t4 = {"ref_c1_1_s4": (7, 3, "left"), "fine_c1_1_s1": (-7, -11, "right"),
              "fine_c1_1_s2": (0, -13, "center"), "coarse_c3_3_s4": (-7, 6, "right"),
              "coarse_c3_3_s2": (0, 8, "center"), "coarse_c3_3_s1": (-7, 4, "right")}
    off_a100 = {"ref_c1_1_s4": (7, 3, "left"), "fine_c1_1_s1": (-7, -11, "right"),
                "fine_c1_1_s2": (0, -13, "center"), "coarse_c3_3_s4": (7, 3, "left"),
                "coarse_c3_3_s2": (7, 3, "left"), "coarse_c3_3_s1": (-7, 4, "right")}

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.15))
    for axi, (ax, data, title) in enumerate([(axes[0], f1, "(a) Tesla T4 (16 GB)"),
                                             (axes[1], a1, "(b) A100-SXM4-80GB")]):
        bm = data["by_mode"]
        offs = off_t4 if axi == 0 else off_a100
        raw = data.get("reference_l1_by_mode")
        if raw is None:
            refl1 = {m: bm[m].get("ref_l1", 0.0) for m in order}
        else:
            refl1 = {m: (raw[m]["mean"] if isinstance(raw[m], dict) else raw[m]) for m in order}
        for m in order:
            d = bm[m]; rl1 = refl1[m]
            col = C_REF if m.startswith("ref") else (C_FINE if m.startswith("fine") else C_COARSE)
            mk = "o" if not m.startswith("coarse") else "s"
            ax.scatter(d["dit_s"], rl1, c=col, marker=mk, s=40, zorder=3,
                       edgecolors="white", linewidths=0.6)
            dx, dy, ha = offs[m]
            ax.annotate(short[m], (d["dit_s"], rl1), textcoords="offset points",
                        xytext=(dx, dy), fontsize=6.3, ha=ha, va="center")
        ax.set_xlabel("DiT sampling time (s)  $\\rightarrow$ cheaper")
        ax.set_title(title, pad=6)
    axes[0].set_ylabel("Reference $L_1$ to c1-1 s4  $\\rightarrow$ closer")
    axes[0].set_xlim(18, 70); axes[0].set_ylim(-0.035, 0.325)
    axes[1].set_xlim(2, 10.5); axes[1].set_ylim(-0.035, 0.325)
    fig.subplots_adjust(wspace=0.30)
    fig.savefig(os.path.join(FIG, "fig_cost_pareto.png"))
    fig.savefig(os.path.join(FIG, "fig_cost_pareto.pdf"))
    plt.close(fig)

# =====================================================================
# Fig 3: hardware scaling / break-even  (break-even labelled inline, no legend box)
# =====================================================================
def make_hw_scaling():
    t4_t = load(r"results/f1_mode_cost/f1_mode_cost_audit.json")["by_mode"]
    t4_noisy = load(r"results/f1_mode_cost_noisy/f1_mode_cost_noisy_audit.json")["by_mode"]
    p100_t = load(r"results/jev_p100_timing/jev_p100_audit.json")
    a100_t = load(r"results/a100_timing/a100_audit.json")
    a100_m = load(r"results/a100_mode_cost/a100_mode_cost_audit.json")["by_mode"]

    regimes = ["T4 clean\n(SDPA)", "T4 noisy\n(SDPA)", "A100 timing\n(FlashAttn)", "A100 full\n(FlashAttn)"]
    fine_dit = [t4_t["fine_c1_1_s1"]["dit_s"], t4_noisy["fine_c1_1_s1_noisy"]["dit_s"],
                4.758, a100_m["fine_c1_1_s1"]["dit_s"]]
    coarse_dit = [t4_t["coarse_c3_3_s1"]["dit_s"], t4_noisy["coarse_c3_3_s1_noisy"]["dit_s"],
                  3.022, a100_m["coarse_c3_3_s1"]["dit_s"]]
    ratios = [p100_t["coarse_over_fine_per_forward"],
              (t4_noisy["coarse_c3_3_s1_noisy"]["dit_s"] / t4_noisy["coarse_c3_3_s1_noisy"]["forward_count"]) /
              (t4_noisy["fine_c1_1_s1_noisy"]["dit_s"] / t4_noisy["fine_c1_1_s1_noisy"]["forward_count"]),
              a100_t["coarse_over_fine_per_forward"],
              a100_m["coarse_c3_3_s1"]["s_per_forward"] / a100_m["fine_c1_1_s1"]["s_per_forward"]]

    x = np.arange(len(regimes)); w = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.15))

    ax = axes[0]
    b1 = ax.bar(x - w / 2, fine_dit, width=w, color=C_FINE, label="fine c1-1 s1", zorder=3)
    b2 = ax.bar(x + w / 2, coarse_dit, width=w, color=C_COARSE, label="coarse c3-3 s1", zorder=3)
    for rect in b1:
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.5,
                f"{rect.get_height():.1f}", ha="center", fontsize=6.2)
    for rect in b2:
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.5,
                f"{rect.get_height():.1f}", ha="center", fontsize=6.2)
    ax.set_xticks(x); ax.set_xticklabels(regimes, fontsize=6.6)
    ax.set_ylabel("Mean DiT sampling time (s)")
    ax.set_title("(a) Rollout DiT latency", pad=6)
    ax.set_ylim(0, 36)
    ax.legend(loc="upper right", fontsize=6.4, frameon=False)

    ax = axes[1]
    cols_r = [C_COARSE if r > 2.333 else "#2ca02c" for r in ratios]
    ax.bar(x, ratios, width=0.5, color=cols_r, zorder=3)
    ax.axhline(2.333, ls="--", color="black", lw=1.1, zorder=4)
    for i, r in enumerate(ratios):
        status = "slower" if r > 2.333 else "faster"
        ax.text(i, 0.14, f"{r:.2f}$\\times$\n({status})", ha="center", va="bottom",
                fontsize=6.1, fontweight="bold", color="white", zorder=5)
    ax.text(0.02, 0.96, "break-even 2.33$\\times$", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.2,
            bbox=dict(fc="white", ec="none", pad=1.0))
    ax.set_xticks(x); ax.set_xticklabels(regimes, fontsize=6.6)
    ax.set_ylabel("Coarse / fine per-forward ratio")
    ax.set_title("(b) Hardware break-even test", pad=6)
    ax.set_ylim(0, 4.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_hw_scaling.png"))
    fig.savefig(os.path.join(FIG, "fig_hw_scaling.pdf"))
    plt.close(fig)

# =====================================================================
# Fig 4: signal AUROC + score separation (labels above bars, legends clear)
# =====================================================================
def make_auroc():
    jev = load(r"results/jev_signal/jev_audit.json")["signals"]
    ans_rows = load_jsonl(r"results/jev_signal/jev_answers.jsonl")

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.15))
    names = ["jev_difficulty", "jev_drift", "jev_motion", "motion", "warp", "complexity"]
    disp = {"jev_difficulty": "Jev\ndifficulty", "jev_drift": "Jev\ndrift", "jev_motion": "Jev\nmotion",
            "motion": "motion\n(flow)", "warp": "warp\nresidual", "complexity": "complexity"}
    ax = axes[0]
    for i, n in enumerate(names):
        s = jev[n]
        col = C_JEV if n.startswith("jev") else C_HC
        ax.bar(i, s["auroc"], color=col, width=0.62, zorder=3)
        ax.text(i, s["auroc"] + 0.02, f"{s['auroc']:.3f}", ha="center", va="bottom",
                fontsize=6.4, fontweight="bold", zorder=5)
    ax.axhline(0.5, ls=":", c="black", lw=1)
    ax.axhline(0.491, ls="--", c="#b22222", lw=1)
    ax.legend(handles=[Line2D([], [], ls=":", c="black", lw=1, label="chance 0.500"),
                       Line2D([], [], ls="--", c="#b22222", lw=1, label="locked residual 0.491")],
              loc="upper right", fontsize=5.9, frameon=False)
    ax.set_xticks(range(len(names))); ax.set_xticklabels([disp[n] for n in names], fontsize=6.1)
    ax.set_ylabel("Held-out AUROC (coarse-unsafe)")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.6, 6.6)
    ax.set_title("(a) Signal discrimination (80 segments)", fontsize=8.0, pad=6)

    ax = axes[1]
    safe_diff = [r["answer"]["answers"]["difficulty"]["score"] for r in ans_rows if r["coarse_unsafe"] == 0]
    unsafe_diff = [r["answer"]["answers"]["difficulty"]["score"] for r in ans_rows if r["coarse_unsafe"] == 1]
    safe_drift = [r["answer"]["answers"]["coarse_drift"]["noul"] for r in ans_rows if r["coarse_unsafe"] == 0]
    unsafe_drift = [r["answer"]["answers"]["coarse_drift"]["noul"] for r in ans_rows if r["coarse_unsafe"] == 1]
    bp_data = [safe_diff, unsafe_diff, [v * 4.0 for v in safe_drift], [v * 4.0 for v in unsafe_drift]]
    positions = [0.85, 1.5, 2.85, 3.5]
    bplot = ax.boxplot(bp_data, positions=positions, widths=0.45, patch_artist=True,
                       medianprops=dict(color="black", linewidth=1.2),
                       flierprops=dict(marker="o", markersize=2.5, alpha=0.5))
    for patch, c in zip(bplot["boxes"], ["#72b7b2", "#2ca02c", "#aec7e8", "#1f77b4"]):
        patch.set_facecolor(c); patch.set_alpha(0.85)
    ax.axhline(1.72, ls="--", color="#d62728", lw=0.9)
    ax.set_xticks([1.175, 3.175])
    ax.set_xticklabels(["Jev difficulty $[0,4]$", "Jev drift risk ($\\times 4$)"], fontsize=6.6)
    ax.set_ylabel("Pre-generation oracle score")
    ax.set_ylim(0.4, 4.5); ax.set_xlim(0.35, 4.0)
    ax.set_title("(b) Safe ($n{=}24$) vs. unsafe ($n{=}56$)", fontsize=8.0, pad=6)
    legend_bp = [
        Rectangle((0, 0), 1, 1, facecolor="#72b7b2", alpha=0.85, label="safe (difficulty)"),
        Rectangle((0, 0), 1, 1, facecolor="#2ca02c", alpha=0.85, label="unsafe (difficulty)"),
        Rectangle((0, 0), 1, 1, facecolor="#aec7e8", alpha=0.85, label="safe (drift)"),
        Rectangle((0, 0), 1, 1, facecolor="#1f77b4", alpha=0.85, label="unsafe (drift)"),
        Line2D([], [], ls="--", color="#d62728", lw=0.9, label="gate $\\tau=1.72$"),
    ]
    ax.legend(handles=legend_bp, loc="upper center", fontsize=5.5, frameon=False, ncol=3,
              columnspacing=0.8, handlelength=1.1, handletextpad=0.4)
    fig.subplots_adjust(wspace=0.30)
    fig.savefig(os.path.join(FIG, "fig_auroc.png")); fig.savefig(os.path.join(FIG, "fig_auroc.pdf"))
    plt.close(fig)

# =====================================================================
# Fig 9: confirmatory CI forest plot (3 panels; labels no longer overlap)
# =====================================================================
def make_phase3_ci():
    ci = load(r"results/a100_online_jev_phase3/phase3_ci.json")["comparisons"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.15), sharey=True)
    pols = [("jev_single", "Jev-single\n(1 query/clip)"),
            ("jev_routed", "Jev-routed\n(4 queries/clip)"),
            ("latent", "Latent refresh\n(heuristic)")]

    panel_meta = [("subject_consistency", "(a) $\\Delta$CLIP subject consistency", (-0.037, 0.007)),
                  ("text_alignment", "(b) $\\Delta$CLIP text alignment", (-0.011, 0.012))]
    for axi, (mk, mtitle, xlims) in enumerate(panel_meta):
        ax = axes[axi]
        for i, (pk, _) in enumerate(pols):
            d = ci[pk][mk]; y = len(pols) - 1 - i
            ax.errorbar(d["mean"], y, xerr=[[d["mean"] - d["ci95"][0]], [d["ci95"][1] - d["mean"]]],
                        fmt="o", color=C_FINE, capsize=2.8, markersize=4.2, zorder=3)
            lo, hi = d["ci95"]
            ax.annotate(f"{d['mean']:+.4f}\n[{lo:+.4f}, {hi:+.4f}]", (d["mean"], y),
                        textcoords="offset points", xytext=(0, 8), fontsize=5.9,
                        ha="center", va="bottom", linespacing=1.1)
        ax.axvline(0, c="black", lw=0.9, ls="--")
        ax.set_yticks(range(len(pols)))
        if axi == 0:
            ax.set_yticklabels([p[1] for p in reversed(pols)], fontsize=6.9)
        ax.set_title(mtitle, fontsize=7.9, pad=6)
        ax.set_xlabel("Paired $\\Delta$ vs. fixed fine", fontsize=7.0)
        ax.set_xlim(*xlims); ax.set_ylim(-0.6, 2.95)

    ax = axes[2]
    for i, (pk, _) in enumerate(pols):
        y = len(pols) - 1 - i
        dd = ci[pk]["dit_s"]; dt = ci[pk]["total_s"]
        ax.errorbar(dd["mean"], y + 0.24, xerr=[[dd["mean"] - dd["ci95"][0]], [dd["ci95"][1] - dd["mean"]]],
                    fmt="o", color=C_FINE, capsize=2.4, markersize=4.0, zorder=3,
                    label="$\\Delta$DiT (s)" if i == 0 else "")
        ax.annotate(f"DiT {dd['mean']:+.2f}", (dd["mean"], y + 0.24), textcoords="offset points",
                    xytext=(0, 6), fontsize=5.6, ha="center", va="bottom", color=C_FINE)
        ax.errorbar(dt["mean"], y - 0.24, xerr=[[dt["mean"] - dt["ci95"][0]], [dt["ci95"][1] - dt["mean"]]],
                    fmt="s", color=C_COARSE, capsize=2.4, markersize=4.0, zorder=3,
                    label="$\\Delta$Total (s)" if i == 0 else "")
        ax.annotate(f"Total {dt['mean']:+.2f}", (dt["mean"], y - 0.24), textcoords="offset points",
                    xytext=(0, -6), fontsize=5.6, ha="center", va="top", color=C_COARSE)
    ax.axvline(0, c="black", lw=0.9, ls="--")
    ax.set_title("(c) $\\Delta$DiT & $\\Delta$Total time (s)", fontsize=7.9, pad=6)
    ax.set_xlabel("Paired $\\Delta$ time vs. fixed fine (s)", fontsize=7.0)
    ax.set_xlim(-14.2, 3.2); ax.set_ylim(-0.7, 3.05)
    ax.legend(loc="lower left", fontsize=6.2, frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_phase3_ci.png")); fig.savefig(os.path.join(FIG, "fig_phase3_ci.pdf"))
    plt.close(fig)

make_architecture()
make_cost_pareto()
make_hw_scaling()
make_auroc()
make_phase3_ci()
print("regenerated: architecture, fig_cost_pareto, fig_hw_scaling, fig_auroc, fig_phase3_ci")
