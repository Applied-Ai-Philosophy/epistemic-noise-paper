"""Figure 4 — Experiment 005: Partial 2x2 alignment vs size effect."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150,
})

BLUE   = "#2563EB"
ORANGE = "#EA580C"
GREEN  = "#16A34A"
GRAY   = "#6B7280"

with open("/media/bjorn/iic/cognos-standalone/research/exp_005_aligned_vs_base/metrics.json") as f:
    metrics = json.load(f)

MODEL_META = {
    "llama3.2:1b": ("llama3.2:1b\n(RLHF, 1B)",   BLUE),
    "tinyllama":   ("tinyllama\n(minimal, 1.1B)", ORANGE),
    "phi3:mini":   ("phi3:mini\n(RLHF, 3.8B)",   GREEN),
}

levels = [0, 1, 2, 3, 4]
level_labels = ["0\nFactual", "1\nEmpirical", "2\nPolicy", "3\nParadox", "4\nExistential"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# ── Left: overall divergence bars ─────────────────────────────────────────────
names  = [MODEL_META[m][0] for m in metrics]
divs   = [metrics[m]["mean_divergence_rate"] * 100 for m in metrics]
colors = [MODEL_META[m][1] for m in metrics]

bars = ax1.bar(names, divs, color=colors, alpha=0.85, width=0.5)
for bar, val in zip(bars, divs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
             f"{val:.0f}%", ha="center", va="bottom", fontsize=10)

ax1.set_ylabel("Mean divergence rate (%)")
ax1.set_ylim(0, 65)
ax1.set_title("Overall Divergence by Model", fontsize=10)

# Alignment-effect bracket between bar 0 and bar 1
ax1.annotate("", xy=(0.88, 26), xytext=(0.12, 26),
             arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.2))
ax1.text(0.5, 28, "Alignment effect\n(same size, ~1B): 12×",
         ha="center", fontsize=8, color=GRAY)

# ── Right: by-determinacy line chart ──────────────────────────────────────────
for m_id, (label, color) in MODEL_META.items():
    by_det = metrics[m_id]["by_determinacy"]
    vals = [by_det.get(str(l), 0) * 100 for l in levels]
    ax2.plot(levels, vals, marker="o", color=color, linewidth=1.8,
             markersize=6, label=label.replace("\n", " "))
    for lvl, val in zip(levels, vals):
        offset = 3 if val < 90 else -8
        ax2.text(lvl, val + offset, f"{val:.0f}", ha="center",
                 fontsize=7.5, color=color)

ax2.set_xticks(levels)
ax2.set_xticklabels(level_labels, fontsize=9)
ax2.set_ylabel("Divergence rate (%)")
ax2.set_ylim(-8, 100)
ax2.set_title("Divergence by Determinacy Level", fontsize=10)
ax2.legend(frameon=False, fontsize=8.5, loc="lower left")

# Calibrated zone annotation
ax2.axvspan(-0.4, 0.4, color=BLUE, alpha=0.07)
ax2.text(0, 93, "Low:\ncalibrated\nvs suppressed",
         ha="center", fontsize=7, color=GRAY, style="italic")

fig.suptitle(
    "Figure 4 — Partial 2×2: Alignment vs Size Effect on Epistemic Variance "
    "(Experiment 005, N=5 samples × 10 questions)",
    fontsize=11, y=1.02,
)
fig.tight_layout()

out = "/media/bjorn/iic/workspace/02_WRITING/papers/pågående/epistemic-noise/figures"
fig.savefig(f"{out}/fig4_alignment_vs_size.png", bbox_inches="tight")
fig.savefig(f"{out}/fig4_alignment_vs_size.pdf", bbox_inches="tight")
print("Fig 4 saved.")
