"""
Figure generation for:
When Alignment Reduces Uncertainty: Epistemic Variance Collapse
and Its Implications for Metacognitive AI

Figures:
  Fig 1 — Divergence Activation by Model (Exp 001)
  Fig 2 — Epistemic Gain: Accuracy vs Calibration (Exp 002)
  Fig 3 — Confidence vs Determinacy Level (Exp 004)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "figure.dpi": 150,
})

BLUE   = "#2563EB"
ORANGE = "#EA580C"
GRAY   = "#6B7280"
LIGHT  = "#E5E7EB"

BASE = "/media/bjorn/iic/cognos-standalone/research"
OUT  = "/media/bjorn/iic/workspace/02_WRITING/papers/pågående/epistemic-noise/figures"


# ── Figure 1: Divergence Activation by Model ──────────────────────────────────
def fig1():
    models = {
        "phi3:mini\n(RLHF, 3.8B)":      f"{BASE}/exp_001_divergence/metrics.json",
        "kimi-k2.5\n(RLHF+, frontier)": f"{BASE}/exp_001_divergence_kimi/metrics.json",
        "tinyllama\n(minimal, 1.1B)":    f"{BASE}/exp_001_divergence_tinyllama/metrics.json",
    }

    labels, div_rates, syn_rates = [], [], []
    for label, path in models.items():
        with open(path) as f:
            m = json.load(f)
        labels.append(label)
        div_rates.append(m["divergence_detected_rate"] * 100)
        syn_rates.append(m["synthesis_success_rate"] * 100)

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - w/2, div_rates, w, label="Divergence rate (%)", color=BLUE, alpha=0.85)
    bars2 = ax.bar(x + w/2, syn_rates,  w, label="Synthesis rate (%)", color=ORANGE, alpha=0.85)

    # value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=9, color=BLUE)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=9, color=ORANGE)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 42)
    ax.set_title("Figure 1 — Epistemic Variance Activation by Model (Experiment 001)",
                 fontsize=11, pad=10)
    ax.legend(frameon=False, fontsize=9)
    ax.axhline(0, color=GRAY, linewidth=0.5)

    # annotation: kimi 0%
    ax.annotate("Frontier collapse\n(0% both signals)",
                xy=(1, 1), xytext=(1.4, 12),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8),
                fontsize=8.5, color=GRAY)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_divergence_by_model.png", bbox_inches="tight")
    fig.savefig(f"{OUT}/fig1_divergence_by_model.pdf", bbox_inches="tight")
    print("Fig 1 saved.")


# ── Figure 2: Epistemic Gain ───────────────────────────────────────────────────
def fig2():
    with open(f"{BASE}/exp_002_epistemic_gain/metrics.json") as f:
        m = json.load(f)

    baseline_acc = m["baseline_accuracy"] * 100
    cognos_acc   = m["cognos_accuracy"] * 100
    baseline_ece = m["baseline_ece"]
    cognos_ece   = m["cognos_ece"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Accuracy
    bars = ax1.bar(["Baseline", "CognOS"], [baseline_acc, cognos_acc],
                   color=[GRAY, BLUE], alpha=0.85, width=0.5)
    for bar, val in zip(bars, [baseline_acc, cognos_acc]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 90)
    ax1.set_title("Accuracy (n=11)", fontsize=10)
    delta_acc = cognos_acc - baseline_acc
    ax1.annotate(f"Δ = {delta_acc:+.1f}%", xy=(0.5, 0.88), xycoords="axes fraction",
                 ha="center", fontsize=9, color=ORANGE,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ORANGE, lw=0.8))

    # ECE (lower = better)
    bars2 = ax2.bar(["Baseline", "CognOS"], [baseline_ece, cognos_ece],
                    color=[GRAY, BLUE], alpha=0.85, width=0.5)
    for bar, val in zip(bars2, [baseline_ece, cognos_ece]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    ax2.set_ylabel("Expected Calibration Error (↓ better)")
    ax2.set_ylim(0, 0.22)
    ax2.set_title("Calibration Error (n=11)", fontsize=10)
    delta_ece = cognos_ece - baseline_ece
    ax2.annotate(f"Δ = {delta_ece:+.3f}", xy=(0.5, 0.88), xycoords="axes fraction",
                 ha="center", fontsize=9, color=BLUE,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=BLUE, lw=0.8))

    fig.suptitle("Figure 2 — Epistemic Gain: Accuracy vs Calibration (Experiment 002)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_epistemic_gain.png", bbox_inches="tight")
    fig.savefig(f"{OUT}/fig2_epistemic_gain.pdf", bbox_inches="tight")
    print("Fig 2 saved.")


# ── Figure 3: Confidence vs Determinacy Level ──────────────────────────────────
def fig3():
    with open(f"{BASE}/mech_exp/gradient_results.json") as f:
        data = json.load(f)

    # aggregate mean confidence per model × determinacy level
    from collections import defaultdict
    confs = defaultdict(lambda: defaultdict(list))
    for row in data:
        confs[row["model"]][row["determinacy_level"]].append(row["confidence"])

    level_labels = [
        "0\nArithmetic\n/ Factual",
        "1\nEmpirical\n(contested)",
        "2\nPreference\n/ Policy",
        "3\nPhilosophical\nParadox",
        "4\nUnfalsifiable",
    ]
    levels = [0, 1, 2, 3, 4]

    model_display = {
        "llama-3.3-70b-versatile": ("llama-3.3-70b", BLUE, "o"),
        "llama-3.1-8b-instant":    ("llama-3.1-8b",  ORANGE, "s"),
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for model, (label, color, marker) in model_display.items():
        means = [np.mean(confs[model][lvl]) for lvl in levels]
        ax.plot(levels, means, marker=marker, color=color, linewidth=1.8,
                markersize=7, label=label)
        for lvl, val in zip(levels, means):
            ax.text(lvl, val + 0.02, f"{val:.2f}", ha="center", fontsize=8, color=color)

    # Highlight Level 3 anomaly
    ax.axvspan(2.5, 3.5, color=ORANGE, alpha=0.08, label="Level 3 anomaly zone")
    ax.annotate("Both models: conf ≈ 1.0\non ill-posed paradoxes",
                xy=(3, 0.98), xytext=(3.2, 0.75),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8),
                fontsize=8.5, color=GRAY)

    ax.set_xticks(levels)
    ax.set_xticklabels(level_labels, fontsize=9)
    ax.set_ylabel("Mean expressed confidence")
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Determinacy level")
    ax.set_title("Figure 3 — Expressed Confidence vs Question Determinacy (Experiment 004)",
                 fontsize=11, pad=10)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.axhline(1.0, color=GRAY, linewidth=0.5, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_determinacy_gradient.png", bbox_inches="tight")
    fig.savefig(f"{OUT}/fig3_determinacy_gradient.pdf", bbox_inches="tight")
    print("Fig 3 saved.")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    print("All figures saved to:", OUT)
