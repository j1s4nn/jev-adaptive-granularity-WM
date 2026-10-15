# Generate all quantitative and qualitative figures for the paper from audited numbers.
# Output: paper/figures/*.png (300 dpi) and *.pdf where useful.
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw, ImageFont

ROOT = r"C:\Users\jisan\OneDrive\Desktop\research\idea_05_adaptive_granularity_WM\04_Experiment"
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 8.5, "font.family": "serif",
    "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
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

# ---------------- Fig 2: cost Pareto, T4 vs A100 ----------------
f1 = load(r"results/f1_mode_cost/f1_mode_cost_audit.json")
a1 = load(r"results/a100_mode_cost/a100_mode_cost_audit.json")

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.0))
order = ["ref_c1_1_s4", "fine_c1_1_s1", "fine_c1_1_s2", "coarse_c3_3_s4", "coarse_c3_3_s2", "coarse_c3_3_s1"]
labels = {"ref_c1_1_s4": "reference c1-1 s4", "fine_c1_1_s1": "fine c1-1 s1",
          "fine_c1_1_s2": "fine c1-1 s2", "coarse_c3_3_s4": "coarse c3-3 s4",
          "coarse_c3_3_s2": "coarse c3-3 s2", "coarse_c3_3_s1": "coarse c3-3 s1"}
offsets_t4 = {"ref_c1_1_s4": (-6, 10), "fine_c1_1_s1": (6, 4), "fine_c1_1_s2": (6, 4),
              "coarse_c3_3_s4": (-10, 8), "coarse_c3_3_s2": (0, -16), "coarse_c3_3_s1": (-8, 10)}
offsets_a100 = {"ref_c1_1_s4": (-6, 10), "fine_c1_1_s1": (6, 6), "fine_c1_1_s2": (6, -13),
                "coarse_c3_3_s4": (8, -4), "coarse_c3_3_s2": (6, -14), "coarse_c3_3_s1": (7, 8)}

for axi, (ax, data, title) in enumerate([(axes[0], f1, "(a) Tesla T4 (16 GB)"), (axes[1], a1, "(b) A100-SXM4-80GB")]):
    bm = data["by_mode"]
    offsets = offsets_t4 if axi == 0 else offsets_a100
    raw = data.get("reference_l1_by_mode")
    if raw is None:
        refl1 = {m: bm[m].get("ref_l1", 0.0) for m in order}
    else:
        refl1 = {m: (raw[m]["mean"] if isinstance(raw[m], dict) else raw[m]) for m in order}
    for m in order:
        d = bm[m]
        rl1 = refl1[m]
        col = C_REF if m.startswith("ref") else (C_FINE if m.startswith("fine") else C_COARSE)
        mk = "o" if not m.startswith("coarse") else "s"
        ax.scatter(d["dit_s"], rl1, c=col, marker=mk, s=42, zorder=3,
                   edgecolors="white", linewidths=0.6)
        ha = "left" if offsets[m][0] > 0 else "right"
        ax.annotate(labels[m], (d["dit_s"], rl1),
                    textcoords="offset points", xytext=offsets[m], fontsize=6.8, ha=ha)
    ax.set_xlabel("DiT sampling time (s)  $\\rightarrow$ cheaper")
    ax.set_title(title)
axes[0].set_ylabel("Reference $L_1$ to c1-1 s4  $\\rightarrow$ closer")
axes[0].set_xlim(18, 70); axes[0].set_ylim(-0.02, 0.30)
axes[1].set_xlim(2, 10.5); axes[1].set_ylim(-0.02, 0.30)
handles = [Line2D([], [], marker="o", ls="", c=C_REF, label="reference (c1-1, s4)"),
           Line2D([], [], marker="o", ls="", c=C_FINE, label="fine (c1-1)"),
           Line2D([], [], marker="s", ls="", c=C_COARSE, label="coarse (c3-3)")]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))
fig.savefig(os.path.join(FIG, "fig_cost_pareto.png"))
fig.savefig(os.path.join(FIG, "fig_cost_pareto.pdf"))
plt.close(fig)

# ---------------- Fig 3: Hardware Scaling & Noisy-Context Regime Across T4 and A100 ----------------
t4_t = load(r"results/f1_mode_cost/f1_mode_cost_audit.json")["by_mode"]
t4_noisy = load(r"results/f1_mode_cost_noisy/f1_mode_cost_noisy_audit.json")["by_mode"]
p100_t = load(r"results/jev_p100_timing/jev_p100_audit.json")
a100_t = load(r"results/a100_timing/a100_audit.json")
a100_m = load(r"results/a100_mode_cost/a100_mode_cost_audit.json")["by_mode"]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.0))
regimes = ["T4 Clean\n(SDPA)", "T4 Noisy-Ctx\n(SDPA)", "A100 Timing\n(FlashAttn)", "A100 Full\n(FlashAttn)"]
fine_dit = [t4_t["fine_c1_1_s1"]["dit_s"],
            t4_noisy["fine_c1_1_s1_noisy"]["dit_s"],
            4.758,
            a100_m["fine_c1_1_s1"]["dit_s"]]
coarse_dit = [t4_t["coarse_c3_3_s1"]["dit_s"],
              t4_noisy["coarse_c3_3_s1_noisy"]["dit_s"],
              3.022,
              a100_m["coarse_c3_3_s1"]["dit_s"]]
ratios = [p100_t["coarse_over_fine_per_forward"],
          (t4_noisy["coarse_c3_3_s1_noisy"]["dit_s"] / t4_noisy["coarse_c3_3_s1_noisy"]["forward_count"]) / (t4_noisy["fine_c1_1_s1_noisy"]["dit_s"] / t4_noisy["fine_c1_1_s1_noisy"]["forward_count"]),
          a100_t["coarse_over_fine_per_forward"],
          a100_m["coarse_c3_3_s1"]["s_per_forward"] / a100_m["fine_c1_1_s1"]["s_per_forward"]]

x = np.arange(len(regimes))
w = 0.34
ax = axes[0]
b1 = ax.bar(x - w/2, fine_dit, width=w, color=C_FINE, label="fine c1-1 s1", zorder=3)
b2 = ax.bar(x + w/2, coarse_dit, width=w, color=C_COARSE, label="coarse c3-3 s1", zorder=3)
for rect in b1:
    h = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, h + 0.6, f"{h:.2f}s", ha="center", fontsize=6.3)
for rect in b2:
    h = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, h + 0.6, f"{h:.2f}s", ha="center", fontsize=6.3)
ax.set_xticks(x); ax.set_xticklabels(regimes, fontsize=6.8)
ax.set_ylabel("Mean DiT sampling time (s)")
ax.set_title("(a) Rollout DiT latency across GPUs & context modes")
ax.set_ylim(0, 35.0)
ax.legend(loc="upper right", fontsize=6.8, frameon=False)

ax = axes[1]
cols_r = ["#d62728" if r > 2.333 else "#2ca02c" for r in ratios]
bars = ax.bar(x, ratios, width=0.48, color=cols_r, zorder=3)
ax.axhline(2.333, ls="--", color="black", lw=1.1, zorder=4, label="Break-even ratio ($7/3 = 2.33\\times$)")
for i, r in enumerate(ratios):
    status = "coarse slower" if r > 2.333 else "coarse faster"
    ax.text(i, r + 0.08, f"{r:.2f}$\\times$\n({status})", ha="center", fontsize=6.2, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(regimes, fontsize=6.8)
ax.set_ylabel("Coarse / fine per-forward time ratio")
ax.set_title("(b) Hardware break-even test ($s_{\\mathrm{fwd}}^{\\mathrm{c3\\text{-}3}} / s_{\\mathrm{fwd}}^{\\mathrm{c1\\text{-}1}}$)")
ax.set_ylim(0, 4.15)
ax.legend(loc="upper left", fontsize=6.8, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_hw_scaling.png"))
fig.savefig(os.path.join(FIG, "fig_hw_scaling.pdf"))
plt.close(fig)

# ---------------- Fig 4: Jev signal AUROC + distributions ----------------
jev = load(r"results/jev_signal/jev_audit.json")["signals"]
ans_rows = load_jsonl(r"results/jev_signal/jev_answers.jsonl")

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.0))
names = ["jev_difficulty", "jev_drift", "jev_motion", "motion", "warp", "complexity"]
disp = {"jev_difficulty": "Jev difficulty", "jev_drift": "Jev drift risk", "jev_motion": "Jev motion class",
        "motion": "motion (flow)", "warp": "warp residual", "complexity": "frame complexity"}
ax = axes[0]
xs = np.arange(len(names))
for i, n in enumerate(names):
    s = jev[n]
    col = C_JEV if n.startswith("jev") else C_HC
    ax.bar(i, s["auroc"], color=col, width=0.60, zorder=3)
    ax.errorbar(i, s["auroc"], yerr=[[s["auroc"] - s["ci_low"]], [s["ci_high"] - s["auroc"]]],
                color="black", capsize=2.5, linewidth=0.9, zorder=4)
    ax.text(i, 0.04, f"{s['auroc']:.3f}", ha="center", fontsize=6.5, color="white", fontweight="bold", zorder=5, rotation=90)
ax.axhline(0.5, ls=":", c="black", lw=1)
ax.axhline(0.491, ls="--", c="#b22222", lw=1)
line_handles = [Line2D([], [], ls=":", c="black", lw=1, label="chance (0.500)"),
                Line2D([], [], ls="--", c="#b22222", lw=1, label="locked residual (0.491)")]
ax.legend(handles=line_handles, loc="upper right", fontsize=6.5, frameon=False)
ax.set_xticks(xs); ax.set_xticklabels([disp[n] for n in names], rotation=22, ha="right")
ax.set_ylabel("Held-out AUROC (coarse-unsafe)")
ax.set_ylim(0, 1.05)
ax.set_title("(a) Discrimination on 80 segments (95% CI)", fontsize=8.5)

ax = axes[1]
safe_diff = [r["answer"]["answers"]["difficulty"]["score"] for r in ans_rows if r["coarse_unsafe"] == 0]
unsafe_diff = [r["answer"]["answers"]["difficulty"]["score"] for r in ans_rows if r["coarse_unsafe"] == 1]
safe_drift = [r["answer"]["answers"]["coarse_drift"]["noul"] for r in ans_rows if r["coarse_unsafe"] == 0]
unsafe_drift = [r["answer"]["answers"]["coarse_drift"]["noul"] for r in ans_rows if r["coarse_unsafe"] == 1]

bp_data = [safe_diff, unsafe_diff, [x * 4.0 for x in safe_drift], [x * 4.0 for x in unsafe_drift]]
positions = [0.8, 1.4, 2.6, 3.2]
bplot = ax.boxplot(bp_data, positions=positions, widths=0.42, patch_artist=True,
                   medianprops=dict(color="black", linewidth=1.2),
                   flierprops=dict(marker="o", markersize=3, alpha=0.5))
colors_bp = ["#72b7b2", "#2ca02c", "#aec7e8", "#1f77b4"]
for patch, c in zip(bplot["boxes"], colors_bp):
    patch.set_facecolor(c)
    patch.set_alpha(0.85)
ax.axhline(1.72, ls="--", color="#d62728", lw=0.9, label="calibrated gate $\\tau_{\\mathrm{Jev}}=1.72$")
ax.set_xticks([1.1, 2.9])
ax.set_xticklabels(["Jev difficulty $[0,4]$", "Jev drift risk (scaled $\\times 4$)"])
ax.set_ylabel("Pre-generation oracle score")
ax.set_title("(b) Score separation: safe ($n{=}24$) vs. unsafe ($n{=}56$)", fontsize=8.5)
legend_bp = [
    plt.Rectangle((0,0),1,1, facecolor="#72b7b2", alpha=0.85, label="coarse-safe (diff)"),
    plt.Rectangle((0,0),1,1, facecolor="#2ca02c", alpha=0.85, label="coarse-unsafe (diff)"),
    plt.Rectangle((0,0),1,1, facecolor="#aec7e8", alpha=0.85, label="coarse-safe (drift)"),
    plt.Rectangle((0,0),1,1, facecolor="#1f77b4", alpha=0.85, label="coarse-unsafe (drift)"),
    Line2D([], [], ls="--", color="#d62728", lw=0.9, label="gate $\\tau_{\\mathrm{Jev}}=1.72$")
]
ax.legend(handles=legend_bp, loc="lower right", fontsize=5.6, frameon=False, ncol=2)
ax.set_ylim(0.4, 3.8)
fig.subplots_adjust(wspace=0.28)
fig.savefig(os.path.join(FIG, "fig_auroc.png")); fig.savefig(os.path.join(FIG, "fig_auroc.pdf"))
plt.close(fig)

# ---------------- Fig 4b: Jev per-segment routing traces (T4, 57 frames) ----------------
routing = load(r"config/jev_routing.json")
tr = routing["traces"]
detail = routing["detail"]
prompt_names = {
    "0": "a boat", "1": "a book", "2": "baseball stadium", "3": "basement",
    "4": "bird over snowy forest", "5": "bird soaring in sky",
    "6": "person bungee jumping", "7": "person canoeing",
}
action_code = {"fine": 2, "coarse": 1, "hold": 0}
cmap = matplotlib.colors.ListedColormap([C_HOLD, C_COARSE, C_FINE])
norm = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

fig, ax = plt.subplots(figsize=(4.2, 2.85))
grid = np.zeros((len(tr), 4))
for i, k in enumerate(sorted(tr, key=int)):
    for j, a in enumerate(tr[k]):
        grid[i, j] = action_code[a]
im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
for i, k in enumerate(sorted(tr, key=int)):
    for j in range(4):
        a = tr[k][j]
        d = detail[k][str(j + 1)]["difficulty"]
        ax.text(j, i, f"{a[0].upper()}\n{d:.1f}", ha="center", va="center",
                fontsize=6.2, color="white", fontweight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels(["seg 1", "seg 2", "seg 3", "seg 4"], fontsize=7)
ax.set_yticks(range(len(tr)))
ax.set_yticklabels([f"P{k} ({prompt_names[k]})" for k in sorted(tr, key=int)], fontsize=6.6)
cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2], fraction=0.045, pad=0.03)
cbar.ax.set_yticklabels(["hold", "coarse", "fine"], fontsize=6.6)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_jev_routing.png")); fig.savefig(os.path.join(FIG, "fig_jev_routing.pdf"))
plt.close(fig)

# ---------------- Fig 6: merged T4 online policies + hold-mode routing (3 panels, 1 row) ----------------
f4 = load(r"results/f4lite/f4lite_audit.json")["by_policy"]
f4q = load(r"results/f4lite/f4lite_quality.json")["by_policy"]
hm = load(r"results/jev_hold_mode/jev_hold_audit.json")["by_policy"]
hf_b = load(r"results/jev_holdfine/jev_holdfine_audit.json")["by_policy"]
hf = dict(hm)
hf["jev_holdfine"] = hf_b["jev_holdfine"]
hf["fixed_fine"] = hf_b["fixed_fine"]

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.0))
p_order = ["fixed_fine", "fixed_coarse", "random50", "alternating50", "adaptive_warp"]
p_labels = {"fixed_fine": "fixed fine", "fixed_coarse": "fixed coarse",
            "random50": "random-50", "alternating50": "alternating-50", "adaptive_warp": "adaptive warp"}
p_colors = {"fixed_fine": C_FINE, "fixed_coarse": C_COARSE, "random50": "#7f7f7f",
            "alternating50": "#bcbd22", "adaptive_warp": "#ff7f0e"}
p_markers = {"fixed_fine": "o", "fixed_coarse": "s", "random50": "^", "alternating50": "v", "adaptive_warp": "D"}
off_l = {"fixed_fine": (5, -8, "left"), "fixed_coarse": (-5, 4, "right"),
         "random50": (5, 3, "left"), "alternating50": (5, -8, "left"), "adaptive_warp": (5, -3, "left")}
off_r = {"fixed_fine": (5, -8, "left"), "fixed_coarse": (5, 4, "left"),
         "random50": (5, 4, "left"), "alternating50": (5, -9, "left"), "adaptive_warp": (-5, -3, "right")}

for p in p_order:
    dit = f4[p]["dit_s"]
    tot = f4[p]["total_s"]
    clip_s = f4q[p]["subject_consistency"]
    axes[0].scatter(dit, clip_s, c=p_colors[p], marker=p_markers[p], s=42, zorder=3, edgecolors="white", linewidths=0.6)
    o = off_l[p]
    axes[0].annotate(p_labels[p], (dit, clip_s), textcoords="offset points", xytext=(o[0], o[1]), fontsize=6.1, ha=o[2])
    axes[1].scatter(tot, clip_s, c=p_colors[p], marker=p_markers[p], s=42, zorder=3, edgecolors="white", linewidths=0.6)
    o2 = off_r[p]
    axes[1].annotate(p_labels[p], (tot, clip_s), textcoords="offset points", xytext=(o2[0], o2[1]), fontsize=6.1, ha=o2[2])

axes[0].set_xlabel("DiT time (s)  $\\rightarrow$ cheaper")
axes[0].set_ylabel("CLIP subject consistency $\\uparrow$")
axes[0].set_title("(a) Quality vs. DiT time (T4)", fontsize=8.5)
axes[0].set_xlim(15.5, 38.0); axes[0].set_ylim(0.944, 0.989)

axes[1].set_xlabel("Total time (s)  $\\rightarrow$ cheaper")
axes[1].set_ylabel("CLIP subject consistency $\\uparrow$")
axes[1].set_title("(b) Quality vs. total time (T4)", fontsize=8.5)
axes[1].set_xlim(165, 425); axes[1].set_ylim(0.944, 0.989)

h_order = ["fixed_fine", "fixed_coarse", "fixed_hold", "jev_routed", "jev_routed_nohold", "jev_holdfine"]
h_labels = {"fixed_fine": "fixed fine", "fixed_coarse": "fixed coarse", "fixed_hold": "fixed hold",
            "jev_routed": "Jev-routed", "jev_routed_nohold": "Jev-routed (no-hold)", "jev_holdfine": "Jev-holdfine"}
h_cols = {"fixed_fine": C_FINE, "fixed_coarse": C_COARSE, "fixed_hold": C_HOLD,
          "jev_routed": "#ff7f0e", "jev_routed_nohold": "#8c564b", "jev_holdfine": C_JEV}
h_mk = {"fixed_fine": "o", "fixed_coarse": "s", "fixed_hold": "^",
        "jev_routed": "D", "jev_routed_nohold": "v", "jev_holdfine": "*"}
h_off = {"fixed_fine": (6, 3, "left"), "fixed_coarse": (-6, -3, "right"),
         "fixed_hold": (6, -3, "left"), "jev_routed": (-6, -4, "right"),
         "jev_routed_nohold": (-6, -3, "right"), "jev_holdfine": (-6, 4, "right")}
ax = axes[2]
for p in h_order:
    d = hf[p]
    sz = 80 if p == "jev_holdfine" else 40
    ax.scatter(d["total_s"], d["mean_reference_l1"], c=h_cols[p], marker=h_mk[p], s=sz, zorder=3, edgecolors="white", linewidths=0.5)
    o = h_off[p]
    fw = "bold" if p == "jev_holdfine" else "normal"
    ax.annotate(h_labels[p], (d["total_s"], d["mean_reference_l1"]),
                textcoords="offset points", xytext=(o[0], o[1]), fontsize=6.0, ha=o[2], fontweight=fw)
ax.set_xlabel("Total time (s)  $\\rightarrow$ cheaper")
ax.set_ylabel("Mean reference $L_1$  $\\rightarrow$ closer")
ax.set_title("(c) Hold-mode routing on T4", fontsize=8.5)
ax.set_xlim(128, 218); ax.set_ylim(0.296, 0.322)

fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_t4_online.png"))
fig.savefig(os.path.join(FIG, "fig_t4_online.pdf"))
plt.close(fig)

# ---------------- (merged into Fig 6 above) ----------------

# ---------------- Fig 7: A100 online policy frontier + action mix ----------------
a100 = load(r"results/a100_online_jev/a100_online_jev_audit.json")
bp = a100["by_policy"]
q = a100["quality_clip"]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
policies = ["fixed_fine", "fixed_coarse", "fixed_hold", "random50", "warp", "jev_difficulty", "jev_routed", "jev_holdfine"]
plabel = {"fixed_fine": "fixed fine", "fixed_coarse": "fixed coarse", "fixed_hold": "fixed hold",
          "random50": "random-50", "warp": "adapt. warp", "jev_difficulty": "Jev-difficulty",
          "jev_routed": "Jev-routed", "jev_holdfine": "Jev-holdfine"}
pcol = {"fixed_fine": C_FINE, "fixed_coarse": C_COARSE, "fixed_hold": C_HOLD, "random50": "#8c8c8c",
        "warp": "#b8860b", "jev_difficulty": C_JEV, "jev_routed": "#17becf", "jev_holdfine": "#bcbd22"}
ax = axes[0]
for p in policies:
    ax.scatter(bp[p]["dit_s"], bp[p]["mean_reference_l1"], c=pcol[p], s=42, zorder=3, edgecolors="white", linewidths=0.5)
ax.set_xlabel("DiT time (s) $\\rightarrow$ cheaper"); ax.set_ylabel("Mean refL1 (diagnostic)")
ax.set_title("(a) Internal diagnostic (A100)")
ax.set_xlim(0.3, 4.5); ax.set_ylim(0.264, 0.287)

ax = axes[1]
sel = [p for p in policies if p in q]
for p in sel:
    ax.scatter(bp[p]["dit_s"], q[p]["subject_consistency"], c=pcol[p], s=42, zorder=3, edgecolors="white", linewidths=0.5)
ax.set_xlim(0.3, 4.5); ax.set_ylim(0.936, 0.996)
ax.set_xlabel("DiT time (s) $\\rightarrow$ cheaper"); ax.set_ylabel("CLIP subject consistency $\\uparrow$")
ax.set_title("(b) Independent endpoint (A100)")

ax = axes[2]
mix_pols = ["warp", "jev_difficulty", "jev_routed", "jev_holdfine"]
mix_labs = ["adaptive\nwarp", "Jev\ndifficulty", "Jev\nrouted", "Jev\nholdfine"]
a100_rows = load_jsonl(r"results/a100_online_jev/results.jsonl")
f_cnt, c_cnt, h_cnt = [], [], []
for p in mix_pols:
    prows = [r for r in a100_rows if r["policy"] == p and r.get("status") == "complete"]
    f_cnt.append(float(np.mean([r.get("fine_count", 0) for r in prows])))
    c_cnt.append(float(np.mean([r.get("coarse_count", 0) for r in prows])))
    h_cnt.append(float(np.mean([r.get("hold_count", 0) for r in prows])))
xm = np.arange(len(mix_pols))
ax.bar(xm, f_cnt, width=0.52, color=C_FINE, label="fine", zorder=3)
ax.bar(xm, c_cnt, bottom=f_cnt, width=0.52, color=C_COARSE, label="coarse", zorder=3)
ax.bar(xm, h_cnt, bottom=np.array(f_cnt)+np.array(c_cnt), width=0.52, color=C_HOLD, label="hold", zorder=3)
ax.set_xticks(xm); ax.set_xticklabels(mix_labs, fontsize=6.6)
ax.set_ylabel("Mean segments per rollout (of 4)")
ax.set_title("(c) Routed action mix (A100)")
ax.set_ylim(0, 4.65)
ax.legend(loc="upper right", fontsize=6.2, ncol=3, frameon=False, columnspacing=0.6, handlelength=1.0)

# shared color-dot key for panels (a) and (b), placed outside/below the axes
handles = [Line2D([], [], marker="o", ls="", markersize=4.4, markerfacecolor=pcol[p],
                  markeredgecolor="white", markeredgewidth=0.5, label=plabel[p]) for p in policies]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.345, -0.015), ncol=4,
           fontsize=5.8, frameon=False, columnspacing=1.2, handletextpad=0.35,
           labelspacing=0.45, borderpad=0.0)
fig.subplots_adjust(left=0.085, right=0.985, top=0.85, bottom=0.31, wspace=0.45)
fig.savefig(os.path.join(FIG, "fig_a100_online.png")); fig.savefig(os.path.join(FIG, "fig_a100_online.pdf"))
plt.close(fig)

# ---------------- Fig 8: Confirmatory 24-prompt A100 CI forest plot (fixed collision, 4 panels) ----------------
ci = load(r"results/a100_online_jev_phase3/phase3_ci.json")["comparisons"]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.0), sharey=True)
metrics = [
    ("subject_consistency", "(a) $\\Delta$CLIP subject consistency", "{:+.4f}\n[{:+.4f}, {:+.4f}]", (-0.036, 0.008)),
    ("text_alignment", "(b) $\\Delta$CLIP text alignment", "{:+.4f}\n[{:+.4f}, {:+.4f}]", (-0.011, 0.011)),
    ("dit_s", "(c) $\\Delta$DiT & $\\Delta$Total time (s)", None, (-14.0, 2.5)),
]
pols = [("jev_single", "Jev-single\n(1 query/clip)"), ("jev_routed", "Jev-routed\n(4 queries/clip)"), ("latent", "Latent refresh\n(heuristic)")]

for axi in [0, 1]:
    ax = axes[axi]
    mk, mtitle, fmt_str, xlims = metrics[axi]
    for i, (pk, plabel) in enumerate(pols):
        d = ci[pk][mk]
        y = len(pols) - 1 - i
        ax.errorbar(d["mean"], y, xerr=[[d["mean"] - d["ci95"][0]], [d["ci95"][1] - d["mean"]]],
                    fmt="o", color="#1f77b4", capsize=3, markersize=4.5, zorder=3)
        lo, hi = d["ci95"]
        ax.annotate(f"{d['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]",
                    (d["mean"], y), textcoords="offset points", xytext=(0, 8), fontsize=6.0,
                    ha="center", va="bottom")
    ax.axvline(0, c="black", lw=0.9, ls="--")
    ax.set_yticks(range(len(pols)))
    if axi == 0:
        ax.set_yticklabels([p[1] for p in reversed(pols)], fontsize=7.2)
    ax.set_title(mtitle, fontsize=8.3)
    ax.set_xlabel("Paired $\\Delta$ vs. fixed fine (95% CI)", fontsize=7.3)
    ax.set_xlim(*xlims)
    ax.set_ylim(-0.45, 2.65)

ax = axes[2]
for i, (pk, plabel) in enumerate(pols):
    y = len(pols) - 1 - i
    d_dit = ci[pk]["dit_s"]
    d_tot = ci[pk]["total_s"]
    ax.errorbar(d_dit["mean"], y + 0.14, xerr=[[d_dit["mean"] - d_dit["ci95"][0]], [d_dit["ci95"][1] - d_dit["mean"]]],
                fmt="o", color=C_FINE, capsize=2.5, markersize=4.2, zorder=3, label="$\\Delta$DiT (s)" if i == 0 else "")
    ax.annotate(f"DiT: {d_dit['mean']:+.2f}s", (d_dit["mean"], y + 0.14),
                textcoords="offset points", xytext=(-7, 0), fontsize=5.8, ha="right", va="center", color=C_FINE)
    ax.errorbar(d_tot["mean"], y - 0.14, xerr=[[d_tot["mean"] - d_tot["ci95"][0]], [d_tot["ci95"][1] - d_tot["mean"]]],
                fmt="s", color="#d62728", capsize=2.5, markersize=4.2, zorder=3, label="$\\Delta$Total (s)" if i == 0 else "")
    ax.annotate(f"Tot: {d_tot['mean']:+.2f}s", (d_tot["mean"], y - 0.14),
                textcoords="offset points", xytext=(-7, 0), fontsize=5.8, ha="right", va="center", color="#d62728")
ax.axvline(0, c="black", lw=0.9, ls="--")
ax.set_title("(c) $\\Delta$DiT & $\\Delta$Total latency (s)", fontsize=8.3)
ax.set_xlabel("Paired $\\Delta$ time vs. fixed fine (s)", fontsize=7.3)
ax.set_xlim(-14.0, 2.8)
ax.set_ylim(-0.45, 2.65)
ax.legend(loc="lower left", fontsize=6.2, frameon=False)

fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_phase3_ci.png")); fig.savefig(os.path.join(FIG, "fig_phase3_ci.pdf"))
plt.close(fig)

# ---------------- Fig 9: RQ3 controller ablation (T4 + A100) ----------------
rq3 = load(r"results/rq3_controller/rq3_audit.json")["by_controller"]
a3 = load(r"results/a100_rq3/a100_rq3_audit.json")["by_controller"]
ctrl = ["threshold_only", "hysteresis_only", "recovery_only", "full", "threshold_only_noisy", "full_noisy"]
clab = ["thr", "hys", "rec", "full", "thr\nnoisy", "full\nnoisy"]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.1))
for ax, data, title in [(axes[0], rq3, "(a) Switch count on T4"), (axes[1], a3, "(b) Switch count on A100")]:
    vals = [data[c]["switch_count_mean" if "switch_count_mean" in data[c] else "switch_count"] for c in ctrl]
    cols = ["#4c78a8" if "noisy" not in c else "#e45756" for c in ctrl]
    ax.bar(range(len(ctrl)), vals, color=cols, width=0.62, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.4)
    ax.set_xticks(range(len(ctrl))); ax.set_xticklabels(clab, fontsize=7)
    ax.set_title(title); ax.set_ylim(0, 1.3)
axes[0].set_ylabel("Mean switch count per rollout $\\downarrow$")
ax = axes[2]
for name, data, mcol, off in [("T4", rq3, "#4c78a8", -0.16), ("A100", a3, "#f58518", 0.16)]:
    seam = [data[c]["switch_adjacent_l1"] for c in ctrl]
    inter = [data[c]["interior_adjacent_l1"] for c in ctrl]
    x = np.arange(len(ctrl))
    ax.bar(x + off - 0.0, seam, width=0.15, color=mcol, zorder=3, label=f"{name} boundary")
    ax.bar(x + off + 0.16 if name == "A100" else x + off, inter, width=0.15, color=mcol, alpha=0.45, zorder=3,
           label=f"{name} interior")
ax.set_xticks(range(len(ctrl))); ax.set_xticklabels(clab, fontsize=7)
ax.set_title("(c) Seam cost: adjacent-frame $L_1$", pad=4); ax.set_ylabel("Adjacent-frame $L_1$")
ax.set_ylim(0, 0.115)
ax.legend(fontsize=5.6, ncol=2, frameon=False, loc="upper left", columnspacing=0.8, handlelength=1.2)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_rq3.png")); fig.savefig(os.path.join(FIG, "fig_rq3.pdf"))
plt.close(fig)

# ---------------- Fig 10: RQ4 monotonicity Jev vs warp (T4) + A100 endpoint contrast ----------------
rq4j = load(r"results/rq4_jev/jev_rq4_audit.json")["all"]
rq4a_raw = load(r"results/a100_online_jev_phase3/rq4_jev_a100.json")["difficulty"]["terciles_mean_fine_gap"]
rq4a_t = [{"mean_fine_gap": v, "n": 32} for v in rq4a_raw]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.1), sharey=True)
panels = [
    (axes[0], rq4j["_diff"]["terciles"], "(a) T4: Jev difficulty ($\\rho=-0.52$)", C_JEV),
    (axes[1], rq4j["warp"]["terciles"], "(b) T4: Warp residual ($\\rho=-0.10$)", C_HC),
    (axes[2], rq4a_t, "(c) A100: Jev difficulty ($\\rho=+0.06$)", "#1f77b4"),
]
for ax, t, title, col in panels:
    xs = [0, 1, 2]
    ys = [tt["mean_fine_gap"] for tt in t]
    ns = [tt["n"] for tt in t]
    ax.bar(xs, ys, color=col, width=0.52, zorder=3)
    for x, y, n in zip(xs, ys, ns):
        offset_y = 0.004 if y >= 0 else -0.014
        ax.text(x, y + offset_y, f"{y:+.3f}\n(n={n})", ha="center", fontsize=6.4)
    ax.axhline(0, c="black", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(["Low", "Mid", "High"])
    ax.set_xlabel("Pre-generation signal tercile")
    ax.set_title(title, fontsize=8.3)
    ax.set_ylim(-0.082, 0.036)
axes[0].set_ylabel("Mean fine gap ($\\mathrm{refL1}_F - \\mathrm{refL1}_C$)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_rq4.png")); fig.savefig(os.path.join(FIG, "fig_rq4.pdf"))
plt.close(fig)

# ---------------- Fig 11: High-resolution Qualitative Grid (4 prompts x 5 columns) ----------------
CAND = os.path.join(ROOT, "paper", "figures_src", "candidates")
row_specs = [
    ("p6", "Prompt P6 (Human Action): \"A person is bungee jumping\"  —  Jev selects [fine, fine, fine, fine] (fast motion)"),
    ("p4", "Prompt P4 (Motion): \"A bird flying over a snowy forest\"  —  Jev selects [fine, fine, coarse, coarse]"),
    ("p0", "Prompt P0 (Object): \"A boat\"  —  hull appearance varies across policies"),
    ("p2", "Prompt P2 (Scene): \"Baseball stadium\"  —  Jev holds static segments (hold, hold, coarse, hold)"),
]
col_specs = [
    ("reference_t0", "Shared Prefix ($t{=}0$)"),
    ("reference_t4", "Reference c1-1 s4 ($t{=}56$)"),
    ("fixed_fine_t4", "Fixed Fine c1-1 s1 ($t{=}56$)"),
    ("fixed_coarse_t4", "Fixed Coarse c3-3 s1 ($t{=}56$)"),
    ("jev_routed_t4", "Jev-Routed Policy ($t{=}56$)"),
]

fig, axes = plt.subplots(len(row_specs), len(col_specs), figsize=(11.2, 3.6))
for r_idx, (p_key, row_title) in enumerate(row_specs):
    for c_idx, (c_suf, col_title) in enumerate(col_specs):
        ax = axes[r_idx, c_idx]
        img_path = os.path.join(CAND, f"{p_key}_{c_suf}.png")
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
            spine.set_linewidth(0.6)
        if r_idx == 0:
            ax.set_title(col_title, fontsize=7.8, fontweight="bold", pad=16)
        if c_idx == 0:
            ax.set_ylabel(f"{p_key.upper()}", fontsize=8.5, fontweight="bold", rotation=0, labelpad=16, va="center")
    # Add row prompt banner above each row
    axes[r_idx, 2].text(0.5, 1.04, row_title,
                        transform=axes[r_idx, 2].transAxes, fontsize=7.6,
                        ha="center", va="bottom", style="italic", color="#222222")

plt.subplots_adjust(wspace=0.04, hspace=0.26)
fig.savefig(os.path.join(FIG, "qualitative_grid.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(FIG, "qualitative_grid.pdf"), dpi=300, bbox_inches="tight")
plt.close(fig)

print("All figures successfully generated in", FIG)
for f in sorted(os.listdir(FIG)):
    print(" ", f)

