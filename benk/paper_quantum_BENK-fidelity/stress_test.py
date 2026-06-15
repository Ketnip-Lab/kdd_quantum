"""
stress_test.py — Pipeline de Testes de Estresse para Q-BENK
============================================================
Avalia o RMSE do CATE estimado pelo Q-BENK sob quatro cenários de estresse,
iterando sobre as três topologias matemáticas: Espiral, Logarítmica e Potência.

Cenários implementados
----------------------
1. Escalabilidade de Amostra (c)  : c ∈ {100, 200, 300, 500, 1000}
2. Robustez ao Ruído (ε)          : ε ∈ {0.00, 0.05, 0.10, 0.15}
3. Desbalanceamento de Grupos (q) : q ∈ {0.1, 0.2, 0.3, 0.4, 0.5}
4. Sensibilidade à Censura (p)    : p ∈ {0.1, 0.2, 0.3, 0.4, 0.5}

Valores de âncora (fixos quando outro parâmetro varia)
------------------------------------------------------
  c = 200 | q = 0.2 | p = 0.25 | ε = 0.05

Saída
-----
  data/stress_tests/scenario_1_sample_scalability.csv
  data/stress_tests/scenario_2_noise_robustness.csv
  data/stress_tests/scenario_3_group_imbalance.csv
  data/stress_tests/scenario_4_censorship_sensitivity.csv
"""

# ---------------------------------------------------------------------------
# Environment / JAX configuration — must happen before any jax import
# ---------------------------------------------------------------------------
import os
import json
import multiprocessing

config_path = os.path.join(os.path.dirname(__file__), "config.json")
if os.path.exists(config_path):
    with open(config_path, "r") as _f:
        _config_data = json.load(_f)
else:
    _config_data = {}

_use_parallel = _config_data.get("execution", {}).get("use_parallel", False)
if _use_parallel:
    _num_cores = str(multiprocessing.cpu_count())
    os.environ["XLA_FLAGS"] = (
        f"--xla_cpu_multi_thread_eigen=true "
        f"intra_op_parallelism_threads={_num_cores}"
    )
    os.environ["OMP_NUM_THREADS"] = _num_cores
    os.environ["MKL_NUM_THREADS"] = _num_cores

# ---------------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------------
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

# Q-BENK modules
from data_generator import generate_synthetic_survival_data
from main import prepare_data, train_model, evaluate_cate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOPOLOGIES = ["Spiral", "Logarithmic", "Power"]

# Anchor (default) values — held fixed while the target parameter varies
ANCHOR = dict(c=200, q=0.2, p=0.25, epsilon=0.05)

# Training hyper-parameters (read from config.json; stress tests use same values)
_training_cfg = _config_data.get("training", {})
N_EPOCHS = _training_cfg.get("n_epochs", 20)
LR       = _training_cfg.get("learning_rate", 0.1)
N_FEATURES = _config_data.get("data", {}).get("n_features", 10)

OUTPUT_DIR = os.path.join("data", "stress_tests")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _compute_true_cate(df: pd.DataFrame) -> np.ndarray:
    """
    Derive the ground-truth CATE for every TREATMENT unit in df
    using the same analytical formula from main.py.
    """
    t_latent = df["t_latent"].values
    time_control   = -np.log(0.02) / (0.1 * np.exp(0.5  * t_latent))
    time_treatment = -np.log(0.3)  / (0.1 * np.exp(0.15 * t_latent))
    df = df.copy()
    df["true_CATE"] = time_treatment - time_control

    df_sorted = df.sort_values("T_obs").reset_index(drop=True)
    df_trt    = df_sorted[df_sorted["A"] == 1]
    return df_trt["true_CATE"].values


def run_stress_scenario(
    c: int,
    q: float,
    p: float,
    epsilon: float,
    function_type: str,
    seed: int = 42,
) -> float:
    """
    Generate data with the given parameters, train Q-BENK, evaluate CATE,
    and return RMSE(τ̂, τ_true).

    Parameters
    ----------
    c            : number of control units
    q            : treatment/control ratio
    p            : right-censoring rate
    epsilon      : additive Gaussian noise std
    function_type: topology name ("Spiral", "Logarithmic", "Power")
    seed         : random seed

    Returns
    -------
    rmse : float — RMSE between estimated and true CATE
    """
    # 1. Data generation
    df, _ = generate_synthetic_survival_data(
        c=c,
        q=q,
        p=p,
        epsilon=epsilon,
        n_features=N_FEATURES,
        function_type=function_type,
        seed=seed,
    )

    # 2. Prepare tensors
    data_dict = prepare_data(df)

    # 3. Train Q-BENK on the control group
    weights, _ = train_model(data_dict, n_epochs=N_EPOCHS, lr=LR)

    # 4. Estimate CATE
    tau_hat, _, _ = evaluate_cate(weights, data_dict)

    # 5. True CATE for treatment units
    true_cate = _compute_true_cate(df)

    # Safety: align lengths (tau_hat indexed by sorted treatment rows)
    n_min = min(len(tau_hat), len(true_cate))
    tau_hat   = np.array(tau_hat)[:n_min]
    true_cate = true_cate[:n_min]

    rmse = float(np.sqrt(np.mean((tau_hat - true_cate) ** 2)))
    return rmse


# ---------------------------------------------------------------------------
# Scenario runners
# ---------------------------------------------------------------------------

def _run_scenario(
    scenario_id: int,
    scenario_name: str,
    param_name: str,
    param_values: list,
    fixed_params: dict,
) -> pd.DataFrame:
    """
    Generic loop: for each value of `param_name` and each topology, run
    one Q-BENK experiment and collect the RMSE.

    Returns a DataFrame with columns:
        topology | parameter_name | parameter_value | rmse | elapsed_sec
    """
    records = []
    total   = len(param_values) * len(TOPOLOGIES)

    print(f"\n{'='*60}")
    print(f"  Cenário {scenario_id}: {scenario_name}")
    print(f"  Parâmetro variado: {param_name} → {param_values}")
    print(f"  Âncoras: { {k: v for k, v in fixed_params.items() if k != param_name} }")
    print(f"  Total de runs: {total}  ({len(param_values)} valores × {len(TOPOLOGIES)} topologias)")
    print(f"{'='*60}")

    with tqdm(total=total, desc=f"Cenário {scenario_id}", unit="run") as pbar:
        for val in param_values:
            # Build the full parameter dict for this run
            params = {**fixed_params, param_name: val}
            for topology in TOPOLOGIES:
                pbar.set_postfix({param_name: val, "topology": topology})
                t0 = time.time()
                try:
                    rmse = run_stress_scenario(
                        c=params["c"],
                        q=params["q"],
                        p=params["p"],
                        epsilon=params["epsilon"],
                        function_type=topology,
                    )
                    status = "ok"
                except Exception as exc:
                    rmse   = float("nan")
                    status = f"error: {exc}"
                    tqdm.write(
                        f"  [!] {topology} | {param_name}={val} → ERRO: {exc}"
                    )

                elapsed = time.time() - t0
                records.append(
                    {
                        "topology":        topology,
                        "parameter_name":  param_name,
                        "parameter_value": val,
                        "rmse":            rmse,
                        "elapsed_sec":     round(elapsed, 2),
                        "status":          status,
                    }
                )
                tqdm.write(
                    f"  [{topology:12s}] {param_name}={val:<6} "
                    f"→ RMSE={rmse:.4f}  ({elapsed:.1f}s)"
                )
                pbar.update(1)

    return pd.DataFrame(records)


def scenario_1_sample_scalability() -> pd.DataFrame:
    """Cenário 1 — Escalabilidade de Amostra: varia c."""
    return _run_scenario(
        scenario_id=1,
        scenario_name="Escalabilidade de Amostra",
        param_name="c",
        param_values=[100, 200, 300, 500, 1000],
        fixed_params=ANCHOR,
    )


def scenario_2_noise_robustness() -> pd.DataFrame:
    """Cenário 2 — Robustez ao Ruído: varia epsilon."""
    return _run_scenario(
        scenario_id=2,
        scenario_name="Robustez ao Ruído",
        param_name="epsilon",
        param_values=[0.00, 0.05, 0.10, 0.15],
        fixed_params=ANCHOR,
    )


def scenario_3_group_imbalance() -> pd.DataFrame:
    """Cenário 3 — Desbalanceamento de Grupos: varia q."""
    return _run_scenario(
        scenario_id=3,
        scenario_name="Desbalanceamento de Grupos",
        param_name="q",
        param_values=[0.1, 0.2, 0.3, 0.4, 0.5],
        fixed_params=ANCHOR,
    )


def scenario_4_censorship_sensitivity() -> pd.DataFrame:
    """Cenário 4 — Sensibilidade à Censura: varia p."""
    return _run_scenario(
        scenario_id=4,
        scenario_name="Sensibilidade à Censura",
        param_name="p",
        param_values=[0.1, 0.2, 0.3, 0.4, 0.5],
        fixed_params=ANCHOR,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    scenarios = [
        (scenario_1_sample_scalability, "scenario_1_sample_scalability.csv"),
        (scenario_2_noise_robustness,   "scenario_2_noise_robustness.csv"),
        (scenario_3_group_imbalance,    "scenario_3_group_imbalance.csv"),
        (scenario_4_censorship_sensitivity, "scenario_4_censorship_sensitivity.csv"),
    ]

    all_results = []
    t_global_start = time.time()

    for runner_fn, filename in scenarios:
        df_result = runner_fn()

        # Export individual scenario CSV
        out_path = os.path.join(OUTPUT_DIR, filename)
        df_result.to_csv(out_path, index=False)
        print(f"\n  ✔ Exportado → {out_path}")

        all_results.append(df_result)

    # Also export a single consolidated CSV
    df_all = pd.concat(all_results, ignore_index=True)
    consolidated_path = os.path.join(OUTPUT_DIR, "stress_test_all_results.csv")
    df_all.to_csv(consolidated_path, index=False)

    elapsed_total = time.time() - t_global_start
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETO em {elapsed_total/60:.1f} minutos")
    print(f"  Resultados consolidados → {consolidated_path}")
    print(f"{'='*60}\n")

    # Print a quick summary table
    print("\n=== RESUMO FINAL ===\n")
    summary = (
        df_all.groupby(["parameter_name", "topology"])["rmse"]
        .agg(["min", "max", "mean"])
        .round(4)
    )
    print(summary.to_string())
    print()


if __name__ == "__main__":
    main()
