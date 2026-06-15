"""
compare_results.py — Comparação Q-BENK Fidelity vs Baselines do Paper BENK
===========================================================================

Executa o Q-BENK com kernel de fidelidade quântica para as três funções
(Spiral, Logarithmic, Power) e plota a comparação com:
  - Baselines do paper BENK original (NW, Cox, SF em variantes T/S/X)
  - BENK clássico (do paper)
  - Q-BENK RBF (implementação anterior — carregado do JSON da pasta main)
  - Q-BENK Fidelidade (esta implementação)

Saída
-----
  data/quantum_benk_fidelity_rmses.json   ← resultados desta execução (cache)
  data/combined_comparison_plot.png       ← gráfico completo
  data/updated_baselines_fidelity.csv     ← tabela CSV com todos os modelos
"""

import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Caminhos ─────────────────────────────────────────────────────────────────
HERE       = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, "data", "quantum_benk_fidelity_rmses.json")
BASELINE   = os.path.join(HERE, "data", "baselines.csv")
# RMSEs do Q-BENK RBF antigo (pasta paper_quantum_BENK-main)
OLD_RMSES  = os.path.join(HERE, "..", "paper_quantum_BENK-main",
                          "data", "quantum_benk_rmses.json")


# ── Funções avaliadas ─────────────────────────────────────────────────────────
FUNCTIONS = ["Spiral", "Logarithmic", "Power"]


# ── Execução do experimento ───────────────────────────────────────────────────
def run_all_benchmarks(force_run: bool = False) -> dict:
    """
    Roda o Q-BENK com kernel de fidelidade para cada função.

    Usa cache JSON se disponível (a menos que force_run=True).
    Cada experimento é executado em subprocesso separado para isolar
    o estado JAX e evitar conflitos de compilação JIT.

    Parâmetros
    ----------
    force_run : bool
        Se True, ignora o cache e executa novamente.

    Retorna
    -------
    dict  {função → RMSE}
    """
    if not force_run and os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r") as f:
                cached = json.load(f)
            if all(ft in cached for ft in FUNCTIONS):
                print(f"[Cache] Q-BENK Fidelidade carregado: {cached}")
                return cached
        except Exception as e:
            print(f"[Aviso] Falha ao ler cache: {e}")

    q_rmses = {}
    for f_type in FUNCTIONS:
        print(f"\n{'='*50}")
        print(f"  Rodando Q-BENK Fidelidade — {f_type}")
        print(f"{'='*50}")
        try:
            cmd = [
                sys.executable, "-c",
                (
                    f"import sys; sys.path.insert(0, r'{HERE}');"
                    f"from main import run_experiment;"
                    f"val = run_experiment('{f_type}');"
                    f"print(f'RESULT:{{val}}')"
                ),
            ]
            res = subprocess.run(
                cmd, capture_output=True, text=True, cwd=HERE
            )
            if res.returncode != 0:
                print(f"[Erro] Subprocesso falhou (código {res.returncode}):")
                print(res.stderr[-2000:])
                q_rmses[f_type] = np.nan
            else:
                val = np.nan
                for line in res.stdout.splitlines():
                    if line.startswith("RESULT:"):
                        val = float(line.split(":", 1)[1].strip())
                    else:
                        print(line)
                if res.stderr:
                    # Exibe apenas avisos relevantes (filtrar ruído de JAX)
                    for ln in res.stderr.splitlines():
                        if "WARNING" not in ln and "INFO" not in ln:
                            print(ln)
                q_rmses[f_type] = val
                print(f"  → RMSE ({f_type}): {val:.4f}")
        except Exception as e:
            print(f"[Erro] {f_type}: {e}")
            q_rmses[f_type] = np.nan

    # Salvar cache
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(q_rmses, f, indent=2)
    print(f"\n[Cache] Resultados salvos em {CACHE_PATH}")
    return q_rmses


# ── Plot de comparação ────────────────────────────────────────────────────────
def plot_combined_results(q_fidelity_rmses: dict) -> None:
    """
    Gera o gráfico de barras comparando todos os modelos.

    Modelos incluídos:
      - Baselines do paper BENK (NW, Cox, SF em variantes T/S/X)
      - BENK clássico (paper)
      - Q-BENK RBF (implementação anterior)
      - Q-BENK Fidelidade (esta implementação)

    A comparação direta entre Q-BENK RBF e Q-BENK Fidelidade permite avaliar
    o impacto da mudança de paradigma de kernel.

    Parâmetros
    ----------
    q_fidelity_rmses : dict  {função → RMSE}  resultados do kernel de fidelidade
    """
    if not os.path.exists(BASELINE):
        print(f"[Erro] Arquivo de baselines não encontrado: {BASELINE}")
        return

    df = pd.read_csv(BASELINE)

    # Carrega RMSEs do Q-BENK RBF antigo, se disponível
    rbf_rmses = {}
    if os.path.exists(OLD_RMSES):
        with open(OLD_RMSES, "r") as f:
            rbf_rmses = json.load(f)
        print(f"[Info] Q-BENK RBF (antigo) carregado: {rbf_rmses}")
    else:
        print(f"[Aviso] Arquivo de RMSEs do Q-BENK RBF não encontrado: {OLD_RMSES}")

    # Lista completa de modelos na ordem desejada
    base_models = df["Model"].unique().tolist()
    all_models  = base_models.copy()
    if rbf_rmses:
        all_models.append("Q-BENK RBF")
    all_models.append("Q-BENK Fidelidade")

    # Monta matriz de dados
    data_matrix = {m: [] for m in all_models}
    for f_type in FUNCTIONS:
        df_f = df[df["Function"] == f_type]
        for model in all_models:
            if model == "Q-BENK RBF":
                data_matrix[model].append(rbf_rmses.get(f_type, np.nan))
            elif model == "Q-BENK Fidelidade":
                data_matrix[model].append(q_fidelity_rmses.get(f_type, np.nan))
            else:
                row = df_f[df_f["Model"] == model]["RMSE"]
                data_matrix[model].append(row.values[0] if not row.empty else np.nan)

    # ── Estilo de publicação ──────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":  "sans-serif",
        "font.size":    13,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(16, 7))

    x     = np.arange(len(FUNCTIONS))
    n     = len(all_models)
    width = 0.82 / n

    # Paleta: destaque para os dois Q-BENKs
    palette = plt.cm.tab20.colors
    for i, model in enumerate(all_models):
        offset = (i - n / 2) * width + width / 2
        if model == "Q-BENK Fidelidade":
            color, edge, lw, zorder = "#4B0082", "#1a0033", 1.5, 5   # roxo escuro
        elif model == "Q-BENK RBF":
            color, edge, lw, zorder = "#228b22", "#0d4b0d", 1.5, 4   # verde escuro
        elif model == "BENK":
            color, edge, lw, zorder = "#1f77b4", "#0d3d6e", 1.5, 4   # azul
        else:
            color, edge, lw, zorder = palette[i % len(palette)], "white", 0.5, 3

        bars = ax.bar(
            x + offset, data_matrix[model], width,
            label=model, color=color, edgecolor=edge, linewidth=lw,
            zorder=zorder, alpha=0.92
        )
        # Adiciona valores nas barras dos modelos de interesse
        if model in ("Q-BENK Fidelidade", "Q-BENK RBF", "BENK"):
            for bar in bars:
                h = bar.get_height()
                if not np.isnan(h):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, h + 0.15,
                        f"{h:.2f}", ha="center", va="bottom",
                        fontsize=9, fontweight="bold",
                        color=color
                    )

    ax.set_ylabel("RMSE (CATE)", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(FUNCTIONS, fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Legenda fora do gráfico
    ax.legend(
        bbox_to_anchor=(1.01, 1), loc="upper left",
        fontsize=11, frameon=True, edgecolor="#cccccc"
    )

    ax.set_title(
        "Comparação de Performance — CATE RMSE\n"
        "Q-BENK Fidelidade vs Q-BENK RBF vs Baselines do Paper BENK",
        fontsize=15, fontweight="bold", pad=14, color="#222222"
    )

    plt.tight_layout()
    out = os.path.join(HERE, "data", "combined_comparison_plot.png")
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nGráfico salvo em '{out}'")
    plt.close()

    # ── Tabela completa ───────────────────────────────────────────────────────
    records = []
    for f_type in FUNCTIONS:
        fi = FUNCTIONS.index(f_type)
        for model in all_models:
            records.append({
                "Model":    model,
                "Function": f_type,
                "RMSE":     round(data_matrix[model][fi], 4)
                            if not np.isnan(data_matrix[model][fi]) else np.nan
            })
    df_out = pd.DataFrame(records)
    print("\nTabela Completa:")
    print(df_out.pivot(index="Model", columns="Function", values="RMSE").to_string())
    out_csv = os.path.join(HERE, "data", "updated_baselines_fidelity.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"\nCSV salvo em '{out_csv}'")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    force = "--force" in sys.argv
    q_fidelity = run_all_benchmarks(force_run=force)
    plot_combined_results(q_fidelity)
