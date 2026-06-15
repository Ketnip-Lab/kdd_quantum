"""
plot_circuit.py — Diagrama do Circuito Quântico do Q-BENK
==========================================================
Gera uma figura publication-ready mostrando:
  - Bloco de Angle Embedding (RY gates)
  - 2 camadas de StronglyEntanglingLayers (RZ/RY/RZ + CNOTs)
  - Medições ⟨Z⟩ em cada qubit
  - Esquema do kernel RBF quântico abaixo
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Estilo ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 14,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# ── Configurações do circuito ─────────────────────────────────────────────────
N_SHOW   = 5     # qubits mostrados explicitamente (depois "..." e qubit 9)
N_QUBITS = 10
N_LAYERS = 2

# Cores
C_EMBED  = "#4C72B0"   # azul — Angle Embedding
C_ROT    = "#DD8452"   # laranja — Rotações (RZ/RY/RZ)
C_CNOT   = "#55A868"   # verde — CNOT
C_MEAS   = "#C44E52"   # vermelho — Medição
C_KERNEL = "#8172B2"   # roxo — Kernel RBF
C_WIRE   = "#333333"
BG_LAYER = "#F0F4FF"   # fundo das camadas SEL

fig, ax = plt.subplots(figsize=(18, 12), facecolor="white")
ax.set_xlim(0, 16)
ax.set_ylim(-1.2, 14.5)
ax.axis("off")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def wire(ax, x0, x1, y, lw=1.5, color=C_WIRE, ls="-"):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, ls=ls, zorder=1)

def gate_box(ax, x, y, label, color, width=0.85, height=0.55,
             fontsize=11, fc_alpha=0.92):
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor="white",
                         alpha=fc_alpha, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            color="white", zorder=4)

def meas_gate(ax, x, y, size=0.5):
    """Medidor estilizado."""
    box = FancyBboxPatch((x - size/2, y - size/2), size, size,
                         boxstyle="round,pad=0.04",
                         facecolor=C_MEAS, edgecolor="white",
                         alpha=0.9, zorder=3)
    ax.add_patch(box)
    # arco de medição
    theta = np.linspace(np.pi, 0, 60)
    r = size * 0.28
    ax.plot(x + r*np.cos(theta), y - size*0.1 + r*np.sin(theta),
            color="white", lw=1.5, zorder=5)
    ax.annotate("", xy=(x + size*0.22, y + size*0.2),
                xytext=(x, y - size*0.1),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.2),
                zorder=5)

def cnot(ax, ctrl_y, tgt_y, x):
    """CNOT: ponto de controle + alvo."""
    ax.plot([x, x], [ctrl_y, tgt_y], color=C_CNOT, lw=1.5, zorder=2)
    ax.plot(x, ctrl_y, "o", color=C_CNOT, markersize=7, zorder=4)
    circle = plt.Circle((x, tgt_y), 0.2, color=C_CNOT,
                         fill=False, lw=1.8, zorder=4)
    ax.add_patch(circle)
    ax.plot([x - 0.2, x + 0.2], [tgt_y, tgt_y], color=C_CNOT, lw=1.8, zorder=5)
    ax.plot([x, x], [tgt_y - 0.2, tgt_y + 0.2], color=C_CNOT, lw=1.8, zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# Eixo Y — posições dos qubits
# ─────────────────────────────────────────────────────────────────────────────
q_labels = [f"q{i}" for i in range(N_SHOW)] + ["⋮", f"q{N_QUBITS-1}"]
q_y      = [9.5 - i * 1.5 for i in range(N_SHOW)] + [9.5 - 5*1.5, 9.5 - 6*1.5]
# Use only defined wires (N_SHOW + 1 qubit shown)
n_wires_shown = N_SHOW + 1
wire_ys = q_y[:N_SHOW] + [q_y[-1]]

# Labels
for label, y in zip(q_labels, q_y):
    if label == "⋮":
        ax.text(0.5, y, "⋮", ha="center", va="center",
                fontsize=24, color=C_WIRE)
    else:
        ax.text(0.5, y, label, ha="center", va="center",
                fontsize=13, fontweight="bold", color=C_WIRE,
                bbox=dict(boxstyle="round,pad=0.2", fc="#EEEEEE", ec="none"))

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Input
# ─────────────────────────────────────────────────────────────────────────────
x_start = 0.9
x_embed_start = 1.5
for y in wire_ys:
    wire(ax, x_start, x_embed_start, y)

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Angle Embedding block
# ─────────────────────────────────────────────────────────────────────────────
x_ae0, x_ae1 = x_embed_start, 3.8
# Background box
bg = FancyBboxPatch((x_ae0, wire_ys[-1] - 0.45),
                    x_ae1 - x_ae0, wire_ys[0] - wire_ys[-1] + 0.9,
                    boxstyle="round,pad=0.1",
                    facecolor="#E8F0FE", edgecolor=C_EMBED,
                    linewidth=1.5, alpha=0.6, zorder=1)
ax.add_patch(bg)
ax.text((x_ae0 + x_ae1)/2, wire_ys[0] + 0.65,
        "Angle Embedding", ha="center", va="center",
        fontsize=13, fontweight="bold", color=C_EMBED)

x_ry = (x_ae0 + x_ae1) / 2
for i, y in enumerate(wire_ys):
    wire(ax, x_ae0, x_ae1, y)
    gate_box(ax, x_ry, y, f"RY\n(x{i})", C_EMBED, width=0.9, height=0.6, fontsize=10)

# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — StronglyEntanglingLayers (2 layers)
# ─────────────────────────────────────────────────────────────────────────────
layer_configs = [
    {"x0": 3.8, "x1": 8.2,  "label": "StronglyEntanglingLayer  (ℓ = 1)"},
    {"x0": 8.2, "x1": 12.6, "label": "StronglyEntanglingLayer  (ℓ = 2)"},
]

for lc in layer_configs:
    lx0, lx1 = lc["x0"], lc["x1"]
    # Background
    bg2 = FancyBboxPatch((lx0, wire_ys[-1] - 0.45),
                         lx1 - lx0, wire_ys[0] - wire_ys[-1] + 0.9,
                         boxstyle="round,pad=0.1",
                         facecolor=BG_LAYER, edgecolor=C_ROT,
                         linewidth=1.5, alpha=0.5, zorder=1)
    ax.add_patch(bg2)
    ax.text((lx0 + lx1)/2, wire_ys[0] + 0.65,
            lc["label"], ha="center", va="center",
            fontsize=13, fontweight="bold", color=C_ROT)

    # Rotation gates (RZ RY RZ per qubit)
    x_r1 = lx0 + 0.8
    x_r2 = lx0 + 1.7
    x_r3 = lx0 + 2.6
    for y in wire_ys:
        wire(ax, lx0, lx1, y)
        gate_box(ax, x_r1, y, "RZ\n(θ₁)", C_ROT, width=0.75, height=0.5, fontsize=9)
        gate_box(ax, x_r2, y, "RY\n(θ₂)", C_ROT, width=0.75, height=0.5, fontsize=9)
        gate_box(ax, x_r3, y, "RZ\n(θ₃)", C_ROT, width=0.75, height=0.5, fontsize=9)

    # CNOT entanglement pattern (cyclic: q0→q1, q1→q2, ...)
    x_cnot1 = lx0 + 3.5
    x_cnot2 = lx0 + 4.1
    for i in range(len(wire_ys) - 1):
        cnot(ax, wire_ys[i], wire_ys[i+1], x_cnot1 + i*0.0)
    # Long-range (q0 → q_last) shown as dashed
    cnot(ax, wire_ys[0], wire_ys[-1], x_cnot2)

# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Measurements
# ─────────────────────────────────────────────────────────────────────────────
x_meas_start = 12.6
x_meas       = 13.2
x_end        = 14.0
for i, y in enumerate(wire_ys):
    wire(ax, x_meas_start, x_meas - 0.25, y)
    meas_gate(ax, x_meas, y)
    # Output label
    label_z = f"⟨Z{i}⟩" if i < N_SHOW else f"⟨Z{N_QUBITS-1}⟩"
    ax.text(x_end + 0.1, y, label_z,
            ha="left", va="center",
            fontsize=13, color=C_MEAS, fontweight="bold")

# Vertical dots between ⟨Z4⟩ and ⟨Z9⟩ on the measurement side
y_last_shown = wire_ys[N_SHOW - 1]   # y of ⟨Z4⟩
y_q9         = wire_ys[-1]            # y of ⟨Z9⟩
y_dots_meas  = (y_last_shown + y_q9) / 2
ax.text(x_end + 0.22, y_dots_meas, "⋮",
        ha="center", va="center",
        fontsize=26, color=C_MEAS, fontweight="bold", zorder=6)

# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — RBF Kernel scheme (TOP panel — above circuit)
# ─────────────────────────────────────────────────────────────────────────────
ky = 12.7   # vertical centre of the kernel panel
# Thin separator below the panel
ax.axhline(ky - 0.85, color="#cccccc", lw=0.9, ls="--", xmin=0.03, xmax=0.97)

# phi_i box
bw, bh = 3.0, 0.62
b1x = 1.0
phi1 = FancyBboxPatch((b1x, ky - bh/2), bw, bh,
                       boxstyle="round,pad=0.08",
                       facecolor=C_KERNEL, edgecolor="white",
                       linewidth=1.2, alpha=0.88, zorder=3)
ax.add_patch(phi1)
ax.text(b1x + bw/2, ky,
        r"$\varphi(\mathbf{x}_i)=[\langle Z_0\rangle,\ldots,\langle Z_9\rangle]$",
        ha="center", va="center", fontsize=12, color="white",
        fontweight="bold", zorder=4)

# phi_j box
b2x = 5.0
phi2 = FancyBboxPatch((b2x, ky - bh/2), bw, bh,
                       boxstyle="round,pad=0.08",
                       facecolor=C_KERNEL, edgecolor="white",
                       linewidth=1.2, alpha=0.88, zorder=3)
ax.add_patch(phi2)
ax.text(b2x + bw/2, ky,
        r"$\varphi(\mathbf{x}_j)=[\langle Z_0\rangle,\ldots,\langle Z_9\rangle]$",
        ha="center", va="center", fontsize=12, color="white",
        fontweight="bold", zorder=4)

# Single arrow from both boxes to the kernel formula
x_arrow_start = b2x + bw + 0.15
x_arrow_end   = 9.1
ax.annotate("", xy=(x_arrow_end, ky), xytext=(x_arrow_start, ky),
            arrowprops=dict(arrowstyle="-|>", color=C_KERNEL, lw=2.0), zorder=4)

# Kernel formula box
kbx = x_arrow_end + 0.1
kbw = 6.5
kernel_box = FancyBboxPatch((kbx, ky - bh/2 - 0.06), kbw, bh + 0.12,
                             boxstyle="round,pad=0.1",
                             facecolor="#F3EEFF", edgecolor=C_KERNEL,
                             linewidth=1.8, alpha=0.95, zorder=3)
ax.add_patch(kernel_box)
ax.text(kbx + kbw/2, ky,
        r"$\kappa(\mathbf{x}_i,\mathbf{x}_j)="
        r"\exp\!\left(-\gamma\|\varphi(\mathbf{x}_i)-\varphi(\mathbf{x}_j)\|^2\right)$",
        ha="center", va="center", fontsize=13.5, color=C_KERNEL,
        fontweight="bold", zorder=4)

# Section label
ax.text(8, ky - 0.68, "Quantum RBF Kernel",
        ha="center", va="center",
        fontsize=12, fontstyle="italic", color=C_KERNEL, zorder=4)

# ─────────────────────────────────────────────────────────────────────────────
# Legend
# ─────────────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_EMBED,  label="Angle Embedding  (RY)"),
    mpatches.Patch(facecolor=C_ROT,    label="Rotações  (RZ–RY–RZ)"),
    mpatches.Patch(facecolor=C_CNOT,   label="Entanglement  (CNOT)"),
    mpatches.Patch(facecolor=C_MEAS,   label="Medição  ⟨Z⟩"),
    mpatches.Patch(facecolor=C_KERNEL, label="Kernel RBF Quântico"),
]
ax.legend(handles=legend_items, loc="lower left",
          bbox_to_anchor=(0.0, -0.06), fontsize=12,
          frameon=True, facecolor="white", edgecolor="#cccccc",
          ncol=5, columnspacing=1.2)

ax.set_title(
    "Q-BENK  ·  Parametrized Quantum Circuit  "
    f"({N_QUBITS} qubits, {N_LAYERS} layers, 60 trainable parameters)",
    fontsize=16, fontweight="bold", pad=12, color="#222222"
)

out = "data/quantum_circuit_diagram.png"
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
plt.close()
