"""
plot_inference.py — Q-BENK: Visualização dos Resultados de Inferência
=======================================================================
Lê data/inference_results.csv e gera 3 painéis:
  1. Distribuição do CATE estimado (histograma + KDE)
  2. E[T(0)] vs E[T(1)] por unidade (scatter com linha de referência)
  3. CATE por unidade ordenado (dot plot)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#333333",
    "axes.labelcolor":   "black",
    "axes.titlecolor":   "black",
    "xtick.color":       "black",
    "ytick.color":       "black",
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "grid.color":        "#dddddd",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "grid.alpha":        0.8,
    "legend.facecolor":  "white",
    "legend.edgecolor":  "#cccccc",
    "font.family":       "sans-serif",
    "font.size":         18,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN  = "#2ca02c"
RED    = "#d62728"

# ── Load ──────────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join("data", "inference_results.csv")
df = pd.read_csv(CSV_PATH)

cate   = df["CATE_hat"].values
e_ctrl = df["E_T_control"].values
e_trt  = df["E_T_treatment"].values
n      = len(df)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 7), facecolor="white")
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35, left=0.06, right=0.97,
                        top=0.88, bottom=0.14)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 1 — Distribuição do CATE (histograma + KDE)
# ─────────────────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])

ax1.hist(cate, bins=10, color=BLUE, alpha=0.55, edgecolor="white",
         linewidth=0.8, density=True, zorder=2)

# KDE
x_kde = np.linspace(cate.min() - 1, cate.max() + 1, 300)
kde   = gaussian_kde(cate, bw_method=0.4)
ax1.plot(x_kde, kde(x_kde), color=BLUE, linewidth=2.5, zorder=3)

# Linha zero
ax1.axvline(0, color=RED, linewidth=1.8, linestyle="--",
            label="CATE = 0  (no effect)", zorder=4)

# Média
ax1.axvline(cate.mean(), color=ORANGE, linewidth=1.8, linestyle="-",
            label=f"Mean = {cate.mean():.2f}", zorder=4)

ax1.set_xlabel("Estimated CATE  (τ̂)", fontsize=18)
ax1.set_ylabel("Density", fontsize=18)
ax1.set_title("CATE Distribution", fontsize=20, fontweight="bold", pad=10)
ax1.legend(fontsize=14)
ax1.yaxis.grid(True)
ax1.set_axisbelow(True)
ax1.tick_params(labelsize=16)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 2 — E[T(0)] vs E[T(1)] scatter
# ─────────────────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])

# Colour-code by sign of CATE
colors = np.where(cate >= 0, GREEN, RED)
ax2.scatter(e_ctrl, e_trt, c=colors, s=80, zorder=3,
            edgecolors="white", linewidths=0.8, alpha=0.9)

# Diagonal: E[T(0)] == E[T(1)]  →  CATE = 0
lim_min = min(e_ctrl.min(), e_trt.min()) * 0.9
lim_max = max(e_ctrl.max(), e_trt.max()) * 1.05
ax2.plot([lim_min, lim_max], [lim_min, lim_max],
         color="#888888", linewidth=1.5, linestyle="--",
         label="No effect  (CATE = 0)", zorder=2)

ax2.set_xlim(lim_min, lim_max)
ax2.set_ylim(lim_min, lim_max)
ax2.set_xlabel("E[T | Control]", fontsize=18)
ax2.set_ylabel("E[T | Treatment]", fontsize=18)
ax2.set_title("Counterfactual Survival Times", fontsize=20, fontweight="bold", pad=10)
ax2.tick_params(labelsize=16)
ax2.grid(True)
ax2.set_axisbelow(True)

# Manual legend patches
from matplotlib.patches import Patch
legend_els = [
    Patch(facecolor=GREEN, label="Benefit  (τ̂ ≥ 0)"),
    Patch(facecolor=RED,   label="Harm  (τ̂ < 0)"),
    plt.Line2D([0], [0], color="#888888", linestyle="--", label="No effect"),
]
ax2.legend(handles=legend_els, fontsize=13, loc="upper left")

# ─────────────────────────────────────────────────────────────────────────────
# Panel 3 — CATE ordenado por unidade (dot plot)
# ─────────────────────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])

order   = np.argsort(cate)
x_units = np.arange(n)
c_sorted = cate[order]
dot_colors = np.where(c_sorted >= 0, GREEN, RED)

ax3.bar(x_units, c_sorted, color=dot_colors, width=0.7,
        edgecolor="white", linewidth=0.5, zorder=2)
ax3.axhline(0, color="#333333", linewidth=1.2, zorder=3)
ax3.axhline(cate.mean(), color=ORANGE, linewidth=1.8, linestyle="--",
            label=f"Mean τ̂ = {cate.mean():.2f}", zorder=4)

ax3.set_xlabel("Unit  (sorted by τ̂)", fontsize=18)
ax3.set_ylabel("Estimated CATE  (τ̂)", fontsize=18)
ax3.set_title("Individual Treatment Effects", fontsize=20, fontweight="bold", pad=10)
ax3.tick_params(axis="x", labelsize=0, length=0)   # hide x-tick labels
ax3.tick_params(axis="y", labelsize=16)
ax3.legend(fontsize=14)
ax3.yaxis.grid(True)
ax3.set_axisbelow(True)

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
fig.suptitle("Q-BENK  ·  CATE Inference Results", fontsize=22,
             fontweight="bold", y=0.97)

out_path = os.path.join("data", "inference_plot.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Plot saved to: {out_path}")
plt.close()
