"""
plot_stress_tests.py — Q-BENK Stress Test Results Visualisation
Produces publication-ready figures matching the style of combined_comparison_plot.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── Publication style (white background, clean axes, serif-free) ──────────────
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
    "grid.color":        "#cccccc",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "grid.alpha":        0.7,
    "legend.facecolor":  "white",
    "legend.edgecolor":  "#cccccc",
    "legend.framealpha": 1.0,
    "text.color":        "black",
    "font.family":       "sans-serif",
    "font.size":         22,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Per-topology colours and markers (distinct, colour-blind safe) ────────────
TOPOLOGY_STYLE = {
    "Spiral":      {"color": "#1f77b4", "marker": "o", "ls": "-"},
    "Logarithmic": {"color": "#ff7f0e", "marker": "s", "ls": "--"},
    "Power":       {"color": "#2ca02c", "marker": "^", "ls": "-."},
}

SCENARIO_META = {
    "c":       {"title": "Scenario 1 — Sample Scalability",
                "xlabel": "Control group size  (c)",
                "xfmt":  lambda v: str(int(v))},
    "epsilon": {"title": "Scenario 2 — Noise Robustness",
                "xlabel": "Noise level  (ε)",
                "xfmt":  lambda v: f"{v:.0%}"},
    "q":       {"title": "Scenario 3 — Group Imbalance",
                "xlabel": "Treatment/control ratio  (q)",
                "xfmt":  lambda v: f"{v:.0%}"},
    "p":       {"title": "Scenario 4 — Censorship Sensitivity",
                "xlabel": "Censoring rate  (p)",
                "xfmt":  lambda v: f"{v:.0%}"},
}

PARAMS_ORDER = ["c", "epsilon", "q", "p"]
TOPOLOGIES   = ["Spiral", "Logarithmic", "Power"]

# ── Load data ─────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join("data", "stress_tests", "stress_test_all_results.csv")
df = pd.read_csv(CSV_PATH)

# ── Main figure: 2×2 grid ─────────────────────────────────────────────────────
fig, axes = plt.subplots(
    2, 2,
    figsize=(20, 15),
    facecolor="white",
)
fig.subplots_adjust(hspace=0.45, wspace=0.32, left=0.08, right=0.97,
                    top=0.93, bottom=0.13)

for idx, param in enumerate(PARAMS_ORDER):
    meta   = SCENARIO_META[param]
    ax     = axes[idx // 2, idx % 2]
    subset = df[df["parameter_name"] == param].sort_values("parameter_value")
    x_vals = sorted(subset["parameter_value"].unique())
    x_pos  = np.arange(len(x_vals))

    for topology in TOPOLOGIES:
        st   = TOPOLOGY_STYLE[topology]
        topo = subset[subset["topology"] == topology]
        y    = [topo[topo["parameter_value"] == x]["rmse"].values[0] for x in x_vals]

        ax.plot(
            x_pos, y,
            color=st["color"],
            marker=st["marker"],
            linestyle=st["ls"],
            linewidth=2.5,
            markersize=11,
            markerfacecolor="white",
            markeredgewidth=2.5,
            markeredgecolor=st["color"],
            label=topology,
            zorder=3,
        )

    # Axes formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels([meta["xfmt"](v) for v in x_vals], fontsize=20)
    ax.set_ylabel("RMSE  (CATE)", fontsize=22)
    ax.set_xlabel(meta["xlabel"], fontsize=22)
    ax.set_title(meta["title"], fontsize=24, fontweight="bold", pad=12)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=20)

# ── Shared legend below the plots ────────────────────────────────────────────
legend_handles = [
    Line2D([0], [0],
           color=TOPOLOGY_STYLE[t]["color"],
           marker=TOPOLOGY_STYLE[t]["marker"],
           linestyle=TOPOLOGY_STYLE[t]["ls"],
           linewidth=2.5, markersize=12,
           markerfacecolor="white",
           markeredgewidth=2.5,
           markeredgecolor=TOPOLOGY_STYLE[t]["color"],
           label=t)
    for t in TOPOLOGIES
]
fig.legend(
    handles=legend_handles,
    title="Topology", title_fontsize=20,
    loc="lower center",
    ncol=3,
    frameon=True,
    fontsize=20,
    bbox_to_anchor=(0.5, 0.01),
)

out_path = os.path.join("data", "stress_tests", "stress_test_plot.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Plot saved to: {out_path}")
plt.close()

# ── Summary heatmap ───────────────────────────────────────────────────────────
param_labels = {
    "c":       "Scalability\n(c)",
    "epsilon": "Noise\n(ε)",
    "q":       "Imbalance\n(q)",
    "p":       "Censoring\n(p)",
}

heat_data = np.zeros((3, 4))
for j, param in enumerate(PARAMS_ORDER):
    for i, topo in enumerate(TOPOLOGIES):
        vals = df[(df["parameter_name"] == param) & (df["topology"] == topo)]["rmse"]
        heat_data[i, j] = vals.mean()

fig2, ax2 = plt.subplots(figsize=(11, 4.5), facecolor="white")
ax2.set_facecolor("white")

im = ax2.imshow(
    heat_data, aspect="auto",
    cmap="RdYlGn_r",
    vmin=heat_data.min() * 0.95,
    vmax=heat_data.max() * 1.05,
)

for i in range(3):
    for j in range(4):
        val = heat_data[i, j]
        # Use black or white text depending on cell brightness
        norm_val = (val - heat_data.min()) / (heat_data.max() - heat_data.min())
        txt_color = "white" if norm_val > 0.6 else "black"
        ax2.text(j, i, f"{val:.3f}",
                 ha="center", va="center",
                 fontsize=17, fontweight="bold",
                 color=txt_color)

ax2.set_xticks(range(4))
ax2.set_xticklabels([param_labels[p] for p in PARAMS_ORDER], fontsize=18)
ax2.set_yticks(range(3))
ax2.set_yticklabels(TOPOLOGIES, fontsize=18)
ax2.tick_params(left=False, bottom=False)
for spine in ax2.spines.values():
    spine.set_visible(False)

cbar = fig2.colorbar(im, ax=ax2, fraction=0.03, pad=0.03)
cbar.set_label("Mean RMSE", fontsize=18)
cbar.ax.tick_params(labelsize=15)

fig2.suptitle(
    "Q-BENK  ·  Mean RMSE Heatmap by Scenario × Topology",
    fontsize=20, fontweight="bold", y=1.03,
)
fig2.tight_layout()

heat_path = os.path.join("data", "stress_tests", "stress_test_heatmap.png")
plt.savefig(heat_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Heatmap saved to:  {heat_path}")
plt.close()
