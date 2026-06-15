import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Import the run_experiment function from main.py
from main import run_experiment

def run_all_benchmarks(force_run=False):
    import json
    cache_path = "data/quantum_benk_rmses.json"
    functions = ["Spiral", "Logarithmic", "Power"]
    
    if not force_run and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            # Check if all functions are in the cache
            if all(f_type in cached for f_type in functions):
                print(f"Loading Quantum-BENK results from cache: {cached}")
                return cached
        except Exception as e:
            print(f"Failed to read cache: {e}")
            
    q_rmses = {}
    for f_type in functions:
        print(f"\nEvaluating Quantum-BENK for {f_type}...")
        try:
            import subprocess
            cmd = [
                sys.executable,
                "-c",
                f"from main import run_experiment; val = run_experiment('{f_type}'); print(f'RESULT:{{val}}')"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"Subprocess failed with code {res.returncode}:\n{res.stderr}")
                q_rmses[f_type] = np.nan
            else:
                val = np.nan
                for line in res.stdout.splitlines():
                    if line.startswith("RESULT:"):
                        val = float(line.split(":")[1])
                    else:
                        print(line)
                if res.stderr:
                    print(res.stderr.strip())
                q_rmses[f_type] = val
        except Exception as e:
            print(f"Failed to run experiment for {f_type}: {e}")
            q_rmses[f_type] = np.nan
            
    # Cache results
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        with open(cache_path, 'w') as f:
            json.dump(q_rmses, f)
    except Exception as e:
        print(f"Failed to write cache: {e}")
        
    return q_rmses

def plot_combined_results(q_rmses, csv_path="data/baselines.csv"):
    if not os.path.exists(csv_path):
        print(f"Baseline file {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    functions = ["Spiral", "Logarithmic", "Power"]
    
    # Models to plot
    models = df['Model'].unique().tolist()
    if "Quantum-BENK" not in models:
        models.append("Quantum-BENK")
        
    # Prepare data matrix
    data_matrix = {model: [] for model in models}
    
    for f_type in functions:
        df_func = df[df['Function'] == f_type]
        for model in models:
            if model == "Quantum-BENK":
                data_matrix[model].append(q_rmses.get(f_type, np.nan))
            else:
                val = df_func[df_func['Model'] == model]['RMSE']
                data_matrix[model].append(val.values[0] if not val.empty else np.nan)
                
    # Plotting
    x = np.arange(len(functions))
    width = 0.8 / len(models)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for i, model in enumerate(models):
        offset = (i - len(models)/2) * width + width/2
        color = "#228b22" if model == "Quantum-BENK" else None
        bars = ax.bar(x + offset, data_matrix[model], width, label=model, color=color)
        
    ax.set_ylabel('RMSE', fontsize=16)
    # ax.set_title('Comparação de Performance (CATE RMSE) em Diferentes Conjuntos de Dados', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(functions, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    # Place legend outside
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    
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
