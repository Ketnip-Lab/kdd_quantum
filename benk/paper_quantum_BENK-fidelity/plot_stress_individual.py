"""
plot_stress_individual.py — Q-BENK Stress Test: 4 individual plots (one per scenario)
Same style as combined_comparison_plot.png. Output: stress_test_scenario_{1..4}.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Publication style ─────────────────────────────────────────────────────────
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
    "grid.linewidth":    0.7,
    "grid.alpha":        0.8,
    "legend.facecolor":  "white",
    "legend.edgecolor":  "#cccccc",
    "legend.framealpha": 1.0,
    "font.family":       "sans-serif",
    "font.size":         22,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

TOPOLOGY_STYLE = {
    "Spiral":      {"color": "#1f77b4", "marker": "o",  "ls": "-"},
    "Logarithmic": {"color": "#ff7f0e", "marker": "s",  "ls": "--"},
    "Power":       {"color": "#2ca02c", "marker": "^",  "ls": "-."},
}
TOPOLOGIES = ["Spiral", "Logarithmic", "Power"]

SCENARIOS = [
    {
        "param":  "c",
        "title":  "Scenario 1 — Sample Scalability",
        "xlabel": "Control group size  (c)",
        "xfmt":   lambda v: str(int(v)),
        "fname":  "stress_test_scenario_1.png",
    },
    {
        "param":  "epsilon",
        "title":  "Scenario 2 — Noise Robustness",
        "xlabel": "Noise level  (ε)",
        "xfmt":   lambda v: f"{v:.0%}",
        "fname":  "stress_test_scenario_2.png",
    },
    {
        "param":  "q",
        "title":  "Scenario 3 — Group Imbalance",
        "xlabel": "Treatment/control ratio  (q)",
        "xfmt":   lambda v: f"{v:.0%}",
        "fname":  "stress_test_scenario_3.png",
    },
    {
        "param":  "p",
        "title":  "Scenario 4 — Censorship Sensitivity",
        "xlabel": "Censoring rate  (p)",
        "xfmt":   lambda v: f"{v:.0%}",
        "fname":  "stress_test_scenario_4.png",
    },
]

# ── Load data ─────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join("data", "stress_tests", "stress_test_all_results.csv")
df = pd.read_csv(CSV_PATH)
OUT_DIR = os.path.join("data", "stress_tests")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Shared legend handles ─────────────────────────────────────────────────────
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

# ── One figure per scenario ───────────────────────────────────────────────────
for sc in SCENARIOS:
    fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")

    subset = df[df["parameter_name"] == sc["param"]].sort_values("parameter_value")
    x_vals = sorted(subset["parameter_value"].unique())
    x_pos  = np.arange(len(x_vals))

    for topo in TOPOLOGIES:
        st   = TOPOLOGY_STYLE[topo]
        data = subset[subset["topology"] == topo]
        y    = [data[data["parameter_value"] == x]["rmse"].values[0] for x in x_vals]

        ax.plot(
            x_pos, y,
            color=st["color"],
            marker=st["marker"],
            linestyle=st["ls"],
            linewidth=2.5,
            markersize=12,
            markerfacecolor="white",
            markeredgewidth=2.5,
            markeredgecolor=st["color"],
            zorder=3,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([sc["xfmt"](v) for v in x_vals], fontsize=20)
    ax.set_ylabel("RMSE  (CATE)", fontsize=22)
    ax.set_xlabel(sc["xlabel"], fontsize=22)
    ax.set_title(sc["title"], fontsize=24, fontweight="bold", pad=12)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=20)

    ax.legend(
        handles=legend_handles,
        title="Topology", title_fontsize=20,
        loc="best",
        fontsize=20,
        frameon=True,
    )

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, sc["fname"])
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.close()

print("\nAll 4 individual scenario plots saved.")
