import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
from main import run_experiment

def run_all_benchmarks(force_run=False):
    cache_path = "data/quantum_benk_fidelity_rmses.json"
    functions = ["Spiral", "Logarithmic", "Power"]
    
    if not force_run and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            # Check if all functions are in the cache and none are NaN or null
            if all(f_type in cached for f_type in functions) and all(cached[f_type] is not None and not np.isnan(cached[f_type]) for f_type in functions):
                print(f"Loading Quantum-BENK Fidelity results from cache: {cached}")
                return cached
        except Exception as e:
            print(f"Failed to read cache: {e}")
            
    q_rmses = {}
    for f_type in functions:
        print(f"\nEvaluating Quantum-BENK Fidelity for {f_type}...")
        try:
            # Run the experiment in-process to avoid subprocess issues
            val = run_experiment(f_type)
            print(f"RESULT for {f_type}: {val:.4f}")
            q_rmses[f_type] = float(val)
        except Exception as e:
            print(f"Failed to run experiment for {f_type}: {e}")
            q_rmses[f_type] = np.nan
            
    # Cache results
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        # Convert values to serializable types
        serializable_rmses = {k: (None if np.isnan(v) else v) for k, v in q_rmses.items()}
        with open(cache_path, 'w') as f:
            json.dump(serializable_rmses, f, indent=2)
    except Exception as e:
        print(f"Failed to write cache: {e}")
        
    return q_rmses

def plot_combined_results(q_fidelity_rmses, csv_path="data/baselines.csv"):
    if not os.path.exists(csv_path):
        print(f"Baseline file {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    functions = ["Spiral", "Logarithmic", "Power"]
    
    # Load RBF results
    rbf_path = "../paper_quantum_BENK-main/data/quantum_benk_rmses.json"
    q_rbf_rmses = {"Spiral": 3.2442, "Logarithmic": 3.0069, "Power": 2.1303} # Fallback
    if os.path.exists(rbf_path):
        try:
            with open(rbf_path, 'r') as f:
                q_rbf_rmses = json.load(f)
            print(f"Loaded Q-BENK RBF results from main folder: {q_rbf_rmses}")
        except Exception as e:
            print(f"Could not load Q-BENK RBF cache, using default fallback: {e}")
            
    # Models to plot
    models = df['Model'].unique().tolist()
    
    # We will add "Q-BENK RBF" and "Q-BENK Fidelity"
    if "Q-BENK RBF" not in models:
        models.append("Q-BENK RBF")
    if "Q-BENK Fidelity" not in models:
        models.append("Q-BENK Fidelity")
        
    # Prepare data matrix
    data_matrix = {model: [] for model in models}
    
    for f_type in functions:
        df_func = df[df['Function'] == f_type]
        for model in models:
            if model == "Q-BENK RBF":
                data_matrix[model].append(q_rbf_rmses.get(f_type, np.nan))
            elif model == "Q-BENK Fidelity":
                data_matrix[model].append(q_fidelity_rmses.get(f_type, np.nan))
            else:
                val = df_func[df_func['Model'] == model]['RMSE']
                data_matrix[model].append(val.values[0] if not val.empty else np.nan)
                
    # Plotting
    x = np.arange(len(functions))
    width = 0.85 / len(models)
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Define color scheme
    colors = {
        "Q-BENK RBF": "#1f77b4",       # Nice blue
        "Q-BENK Fidelity": "#228b22",  # Forest green
        "BENK": "#d62728",             # Red
    }
    
    for i, model in enumerate(models):
        offset = (i - len(models)/2) * width + width/2
        color = colors.get(model, None)
        hatch = None
        if model == "Q-BENK Fidelity":
            hatch = "//"
        bars = ax.bar(x + offset, data_matrix[model], width, label=model, color=color, edgecolor='black', linewidth=0.5, hatch=hatch)
        
    ax.set_ylabel('CATE RMSE', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(functions, fontsize=16, fontweight='bold')
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Legend outside, nicely placed
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=14, frameon=True, shadow=True)
    
    # Clean style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/combined_comparison_plot.png", dpi=300)
    print("\nGráfico combinado salvo em 'data/combined_comparison_plot.png'")
    
    # Save the complete table
    records = []
    for f_type in functions:
        for model in models:
            records.append({
                "Model": model,
                "Function": f_type,
                "RMSE": data_matrix[model][functions.index(f_type)]
            })
    df_out = pd.DataFrame(records)
    print("\nTabela Completa Atualizada:")
    print(df_out.to_string(index=False))
    df_out.to_csv("data/updated_baselines.csv", index=False)

if __name__ == "__main__":
    force = "--force" in sys.argv
    q_rmses = run_all_benchmarks(force_run=force)
    plot_combined_results(q_rmses)
