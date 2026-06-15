import os
import json
import multiprocessing

config_path = os.path.join(os.path.dirname(__file__), "config.json")
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config_data = json.load(f)
else:
    config_data = {}

use_parallel = config_data.get("execution", {}).get("use_parallel", False)
if use_parallel:
    num_cores = str(multiprocessing.cpu_count())
    os.environ["XLA_FLAGS"] = f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={num_cores}"
    os.environ["OMP_NUM_THREADS"] = num_cores
    os.environ["MKL_NUM_THREADS"] = num_cores

import jax
import optax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from data_generator import generate_synthetic_survival_data
from quantum_kernel import initialize_weights
from beran_estimator import expected_survival_time, loss_fn, expected_survival_time_treatment, beran_survival_function

def prepare_data(df):
    """
    Sort data by T_obs (ascending) as required by Beran estimator.
    """
    df_sorted = df.sort_values(by='T_obs').reset_index(drop=True)
    
    # Split into Control and Treatment
    df_0 = df_sorted[df_sorted['A'] == 0]
    df_1 = df_sorted[df_sorted['A'] == 1]
    
    X_cols = [col for col in df.columns if col.startswith('X_')]
    
    return {
        'all': {
            'X': jnp.array(df_sorted[X_cols].values),
            'T_obs': jnp.array(df_sorted['T_obs'].values),
            'delta': jnp.array(df_sorted['delta'].values),
            'T_true': jnp.array(df_sorted['T_true'].values)
        },
        'control': {
            'X': jnp.array(df_0[X_cols].values),
            'T_obs': jnp.array(df_0['T_obs'].values),
            'delta': jnp.array(df_0['delta'].values),
            'T_true': jnp.array(df_0['T_true'].values)
        },
        'treatment': {
            'X': jnp.array(df_1[X_cols].values),
            'T_obs': jnp.array(df_1['T_obs'].values),
            'delta': jnp.array(df_1['delta'].values),
            'T_true': jnp.array(df_1['T_true'].values)
        }
    }

def train_model(data_dict, n_epochs=20, lr=0.1):
    # Initialize weights
    weights = initialize_weights(n_layers=2)
    
    # Optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(weights)
    
    # Data to train on (Control group)
    X_train = data_dict['control']['X']
    T_train = data_dict['control']['T_obs']
    delta_train = data_dict['control']['delta']
    T_true_train = data_dict['control']['T_true']
    
    # Loss and grad function
    loss_and_grad = jax.value_and_grad(loss_fn)
    
    from tqdm import tqdm
    print("Starting Training on Control Group...")
    losses = []
    
    pbar = tqdm(range(n_epochs), desc="Training")
    for epoch in pbar:
        loss_val, grads = loss_and_grad(weights, X_train, T_true_train, delta_train, X_train, T_train, delta_train)
        updates, opt_state = optimizer.update(grads, opt_state)
        weights = optax.apply_updates(weights, updates)
        
        losses.append(float(loss_val))
        pbar.set_postfix({'RMSE': f"{loss_val:.4f}"})
            
    return weights, losses

def evaluate_cate(weights, data_dict):
    print("Evaluating CATE...")
    
    X_trt = data_dict['treatment']['X']
    
    X_0 = data_dict['control']['X']
    T_0 = data_dict['control']['T_obs']
    delta_0 = data_dict['control']['delta']
    
    X_1 = data_dict['treatment']['X']
    T_1 = data_dict['treatment']['T_obs']
    delta_1 = data_dict['treatment']['delta']
    
    # Calculate expected survival times
    E_0, E_1 = expected_survival_time_treatment(
        X_trt, X_0, T_0, delta_0, X_1, T_1, delta_1, weights
    )
    
    # Estimated CATE
    tau_hat = E_1 - E_0
    
    return tau_hat, E_0, E_1

def plot_survival_curves(weights, data_dict):
    X_0 = data_dict['control']['X']
    T_0 = data_dict['control']['T_obs']
    delta_0 = data_dict['control']['delta']
    
    X_1 = data_dict['treatment']['X']
    T_1 = data_dict['treatment']['T_obs']
    delta_1 = data_dict['treatment']['delta']
    
    # Compute Beran survival curves for all samples
    S_t_0 = beran_survival_function(X_0, X_0, T_0, delta_0, weights)
    S_t_1 = beran_survival_function(X_1, X_1, T_1, delta_1, weights)
    
    plt.figure(figsize=(10, 6))
    
    # Average survival curve
    plt.plot(T_0, jnp.mean(S_t_0, axis=0), label="Quantum-Beran (Control Avg)", color='blue')
    plt.plot(T_1, jnp.mean(S_t_1, axis=0), label="Quantum-Beran (Treatment Avg)", color='orange')
    
    # Kaplan-Meier
    kmf_0 = KaplanMeierFitter()
    kmf_0.fit(np.array(T_0), np.array(delta_0), label="Kaplan-Meier (Control)")
    kmf_0.plot(ax=plt.gca(), color='blue', linestyle='--')
    
    kmf_1 = KaplanMeierFitter()
    kmf_1.fit(np.array(T_1), np.array(delta_1), label="Kaplan-Meier (Treatment)")
    kmf_1.plot(ax=plt.gca(), color='orange', linestyle='--')
    
    plt.title("Survival Curves: Kaplan-Meier vs Quantum-Beran")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("survival_curves.png", dpi=300)
    print("Saved survival_curves.png")

def run_experiment(function_type="Spiral"):
    import json
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {"quantum": {}, "training": {}, "data": {}}

    # --- Data parameters ---
    # Derive c (number of controls) from legacy n_samples + default q=0.2
    n_samples  = config.get("data", {}).get("n_samples", 1000)
    n_features = config.get("data", {}).get("n_features", 10)
    q_default  = 0.2   # default treatment ratio
    c = max(1, round(n_samples / (1.0 + q_default)))

    n_epochs = config.get("training", {}).get("n_epochs", 20)
    lr = config.get("training", {}).get("learning_rate", 0.1)
    
    print(f"\n{'='*40}\nRunning Experiment for: {function_type}\n{'='*40}")

    df, _ = generate_synthetic_survival_data(
        c=c,
        q=q_default,
        p=0.25,
        epsilon=0.05,
        n_features=n_features,
        function_type=function_type,
    )
    
    # Calculate True CATE for treatment units (after T_obs sort)
    t_latent = df['t_latent'].values
    time_control   = -np.log(0.02) / (0.1 * np.exp(0.5  * t_latent))
    time_treatment = -np.log(0.3)  / (0.1 * np.exp(0.15 * t_latent))
    df['true_CATE'] = time_treatment - time_control
    
    # Save the generated dataset
    dataset_dir = os.path.join("data", "datasets")
    os.makedirs(dataset_dir, exist_ok=True)
    df.to_csv(os.path.join(dataset_dir, f"{function_type.lower()}_dataset.csv"), index=False)
    
    data_dict = prepare_data(df)
    
    weights, losses = train_model(data_dict, n_epochs=n_epochs, lr=lr)
    
    tau_hat, E_0, E_1 = evaluate_cate(weights, data_dict)
    
    # Treatment group index after sorting (must align with evaluate_cate output)
    df_sorted = df.sort_values(by='T_obs').reset_index(drop=True)
    df_1 = df_sorted[df_sorted['A'] == 1]
    true_cate = df_1['true_CATE'].values

    # Align lengths in case of minor off-by-one
    n_min = min(len(tau_hat), len(true_cate))
    tau_hat_arr   = np.array(tau_hat)[:n_min]
    true_cate_arr = true_cate[:n_min]
    
    mae_cate  = np.mean(np.abs(tau_hat_arr - true_cate_arr))
    mse_cate  = np.mean((tau_hat_arr - true_cate_arr) ** 2)
    rmse_cate = np.sqrt(mse_cate)
    
    print(f"[{function_type}] CATE Evaluation - MAE: {mae_cate:.4f}, MSE: {mse_cate:.4f}, RMSE: {rmse_cate:.4f}")
    
    # plot_survival_curves is heavy — uncomment to enable
    # plot_survival_curves(weights, data_dict)
    
    return rmse_cate

if __name__ == "__main__":
    import json
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        function_type = config.get("data", {}).get("function_type", "Spiral")
    else:
        function_type = "Spiral"
    
    ALL_FUNCTIONS = ["Spiral", "Logarithmic", "Power"]
    
    if function_type.lower() == "all":
        print("Rodando todos os tipos de função disponíveis: " + ", ".join(ALL_FUNCTIONS))
        results = {}
        for ft in ALL_FUNCTIONS:
            results[ft] = run_experiment(ft)
        print("\n=== Resumo Final ===")
        for ft, rmse in results.items():
            print(f"  {ft:15s} | RMSE: {rmse:.4f}")
    else:
        run_experiment(function_type)
