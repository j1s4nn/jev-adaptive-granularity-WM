# Regenerate the system architecture diagram with non-overlapping labels.
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = r"C:\Users\jisan\OneDrive\Desktop\research\idea_05_adaptive_granularity_WM\04_Experiment"
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

fig, ax = plt.subplots(figsize=(13.5, 6.6), dpi=300)
ax.set_xlim(0, 14); ax.set_ylim(0.1, 7.2); ax.axis("off")


def box(x, y, w, h, title, sub, color):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                       fc=color, ec="#25324a", lw=1.6)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center",
            fontsize=12.5, weight="bold", color="#14213d")
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
            fontsize=9, color="#25324a")


def arrow(x1, y1, x2, y2):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                        lw=1.7, color="#34495e")
    ax.add_patch(a)


box(0.35, 5.75, 2.30, 1.30, "Prompt + context", "text, previous latent frame", "#d9ecff")
box(3.15, 5.75, 2.40, 1.30, "Feature state", "caption + motion stats\n+ segment index", "#e9e0ff")
box(6.10, 5.75, 2.55, 1.30, "Jev oracle", "difficulty / drift /\nmotion decision", "#ffe7c2")
box(9.25, 5.75, 2.20, 1.30, "Policy state", "threshold +\nhysteresis + recovery", "#d9f2df")
box(12.00, 5.75, 1.60, 1.30, "Action", "fine / coarse / hold", "#ffd9df")
arrow(2.65, 6.40, 3.15, 6.40); arrow(5.55, 6.40, 6.10, 6.40)
arrow(8.65, 6.40, 9.25, 6.40); arrow(11.45, 6.40, 12.00, 6.40)

box(1.10, 2.65, 3.20, 1.50, "Fine mode c1-1", "1 latent frame / chunk\nmore forwards, higher consistency", "#d9ecff")
box(5.40, 2.65, 3.20, 1.50, "Coarse mode c3-3", "3 latent frames / chunk\nfewer NFEs, heavier forwards", "#ffe7c2")
box(9.70, 2.65, 3.20, 1.50, "Hold mode", "copy previous latent\nnear-zero DiT cost", "#d9f2df")
for x in [2.70, 7.00, 11.30]:
    arrow(12.80, 5.75, x, 4.15)
ax.text(7.0, 5.15, "the selected mode executes the next causal segment",
        ha="center", fontsize=9.5, color="#34495e")

box(2.10, 0.35, 3.20, 1.25, "Generated segment", "decoded RGB frames\n+ timing trace", "#f4f4f4")
box(5.95, 0.35, 3.20, 1.25, "Independent audit", "CLIP subject/text\n+ optical-flow suite", "#f4f4f4")
box(9.80, 0.35, 3.20, 1.25, "Decision record", "quality-cost pair\n+ switch boundary cost", "#f4f4f4")
arrow(2.70, 2.65, 3.70, 1.60); arrow(7.00, 2.65, 7.55, 1.60); arrow(11.30, 2.65, 11.40, 1.60)

fig.savefig(os.path.join(FIG, "architecture.png"), bbox_inches="tight")
fig.savefig(os.path.join(FIG, "architecture.pdf"), bbox_inches="tight")
plt.close(fig)
print("architecture written")
