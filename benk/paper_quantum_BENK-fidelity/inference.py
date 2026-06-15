"""
inference.py — Q-BENK: Inferência de CATE em novos dados
=========================================================
Como usar:
  1. Treine o modelo via main.py (gera os pesos e os datasets de referência).
  2. Forneça seus dados novos (grupo de interesse para estimar CATE).
  3. Execute este script:

     python inference.py --topology Spiral --n_samples_new 50

O script:
  - Carrega o dataset de referência gerado pelo main.py (controle + tratamento).
  - Instancia pesos treinados (re-treino rápido, ou você pode salvar/carregar).
  - Estima E[T(0)|X] e E[T(1)|X] para cada unidade nova.
  - Calcula tau_hat = E[T(1)|X] - E[T(0)|X]  (CATE estimado).
  - Salva os resultados em data/inference_results.csv.
"""

import os
import argparse
import numpy as np
import pandas as pd
import jax.numpy as jnp

# ── Importa componentes do Q-BENK ─────────────────────────────────────────────
from data_generator import generate_synthetic_survival_data
from quantum_kernel import initialize_weights
from beran_estimator import expected_survival_time_treatment
from main import prepare_data, train_model


# ─────────────────────────────────────────────────────────────────────────────
# 1. Parseamento de argumentos
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Q-BENK Inference: estima CATE para novas unidades."
    )
    parser.add_argument(
        "--topology",
        type=str,
        default="Spiral",
        choices=["Spiral", "Logarithmic", "Power"],
        help="Topologia do gerador de dados (default: Spiral).",
    )
    parser.add_argument(
        "--n_samples_new",
        type=int,
        default=50,
        help="Número de novas unidades para inferência (default: 50).",
    )
    parser.add_argument(
        "--c",
        type=int,
        default=200,
        help="Tamanho do grupo de controle no treino (default: 200).",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=0.2,
        help="Proporção tratamento/controle (default: 0.2).",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.25,
        help="Taxa de censura (default: 0.25).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Nível de ruído (default: 0.05).",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=20,
        help="Épocas de treino (default: 20).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate do otimizador Adam (default: 0.1).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("data", "inference_results.csv"),
        help="Caminho do CSV de saída.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Treino (ou carregamento) dos pesos
# ─────────────────────────────────────────────────────────────────────────────
def get_trained_weights(args):
    """
    Gera dados de referência e treina o modelo.
    Para produção, substitua por: jnp.load("pesos.npy", allow_pickle=True)
    """
    print(f"\n[1/3] Gerando dados de referência ({args.topology})...")
    df_ref, _ = generate_synthetic_survival_data(
        c=args.c,
        q=args.q,
        p=args.p,
        epsilon=args.epsilon,
        n_features=10,
        function_type=args.topology,
    )
    data_dict = prepare_data(df_ref)

    print(f"[2/3] Treinando Q-BENK ({args.n_epochs} épocas, lr={args.lr})...")
    weights, _ = train_model(data_dict, n_epochs=args.n_epochs, lr=args.lr)

    return weights, data_dict


# ─────────────────────────────────────────────────────────────────────────────
# 3. Inferência em novas unidades
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(weights, data_dict_ref, args):
    """
    Gera (ou recebe) novas unidades e estima o CATE.
    
    ► Para usar seus próprios dados, substitua a seção abaixo por:
        X_new = jnp.array(seu_dataframe[colunas_X].values)
    """
    print(f"\n[3/3] Inferindo CATE para {args.n_samples_new} novas unidades...")

    # ── Simulação de dados novos (substitua por dados reais) ──────────────────
    df_new, _ = generate_synthetic_survival_data(
        c=args.n_samples_new,
        q=0.0,          # só queremos features, sem grupo tratamento
        p=0.0,
        epsilon=0.0,
        n_features=10,
        function_type=args.topology,
    )
    X_cols = [col for col in df_new.columns if col.startswith("X_")]
    X_new = jnp.array(df_new[X_cols].values)
    # ── Fim da seção de dados novos ───────────────────────────────────────────

    # Dados de referência (controle e tratamento do treino)
    X_0     = data_dict_ref["control"]["X"]
    T_0     = data_dict_ref["control"]["T_obs"]
    delta_0 = data_dict_ref["control"]["delta"]

    X_1     = data_dict_ref["treatment"]["X"]
    T_1     = data_dict_ref["treatment"]["T_obs"]
    delta_1 = data_dict_ref["treatment"]["delta"]

    # Estimativa de tempo de sobrevivência esperado sob controle e tratamento
    E_0, E_1 = expected_survival_time_treatment(
        X_new,
        X_0, T_0, delta_0,
        X_1, T_1, delta_1,
        weights,
    )

    tau_hat = np.array(E_1 - E_0)    # CATE estimado
    E_0_np  = np.array(E_0)
    E_1_np  = np.array(E_1)

    return tau_hat, E_0_np, E_1_np, df_new


# ─────────────────────────────────────────────────────────────────────────────
# 4. Salvar e exibir resultados
# ─────────────────────────────────────────────────────────────────────────────
def save_results(tau_hat, E_0, E_1, df_new, args):
    X_cols = [col for col in df_new.columns if col.startswith("X_")]
    n = min(len(tau_hat), len(df_new))

    results = df_new[X_cols].iloc[:n].copy().reset_index(drop=True)
    results["E_T_control"]   = E_0[:n]
    results["E_T_treatment"] = E_1[:n]
    results["CATE_hat"]      = tau_hat[:n]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results.to_csv(args.output, index=False)

    print(f"\n{'='*55}")
    print(f"  Topologia        : {args.topology}")
    print(f"  Novas unidades   : {n}")
    print(f"  CATE médio       : {tau_hat[:n].mean():.4f}")
    print(f"  CATE std         : {tau_hat[:n].std():.4f}")
    print(f"  CATE mín / máx   : {tau_hat[:n].min():.4f} / {tau_hat[:n].max():.4f}")
    print(f"  Resultados salvos: {args.output}")
    print(f"{'='*55}\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    weights, data_dict_ref = get_trained_weights(args)
    tau_hat, E_0, E_1, df_new = run_inference(weights, data_dict_ref, args)
    results = save_results(tau_hat, E_0, E_1, df_new, args)

    # Prévia dos 5 primeiros resultados
    print(results[["E_T_control", "E_T_treatment", "CATE_hat"]].head())
