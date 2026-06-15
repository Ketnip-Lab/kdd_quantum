import json

notebook_path = '/home/renatodias/Quantum/kdd/notebooks/02_quantum_algorithms.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # 1. Imports
        if 'import pennylane as qml' in source and 'qiskit_aer.primitives' in source:
            if 'import concurrent.futures' not in source:
                new_source = source.replace('import time', 'import time\nimport concurrent.futures')
                new_source = new_source.replace('from qiskit.primitives import StatevectorEstimator', 'from qiskit.primitives import StatevectorEstimator\nfrom qiskit_aer.primitives import Estimator as AerEstimator')
                cell['source'] = [line + '\n' for line in new_source.split('\n')[:-1]]
                if not new_source.endswith('\n'):
                    cell['source'][-1] = cell['source'][-1].rstrip('\n')
                    
        # 2. QSVM
        if 'qsvm = QSVC(quantum_kernel=quantum_kernel)' in source:
            if 'sampler = AerSampler' not in source:
                new_source = source.replace(
                    'quantum_kernel   = FidelityQuantumKernel(feature_map=feature_map_qsvm)',
                    "sampler = AerSampler(run_options={'max_parallel_threads': 0})\nquantum_kernel   = FidelityQuantumKernel(feature_map=feature_map_qsvm, sampler=sampler)"
                )
                new_source = new_source.replace("print('\\n⏳ Treinando QSVM (pode demorar alguns minutos)...')", "print('\\n⏳ Treinando QSVM com múltiplas threads (pode demorar alguns minutos)...')")
                cell['source'] = [line + '\n' for line in new_source.split('\n')[:-1]]
                if not new_source.endswith('\n'):
                    cell['source'][-1] = cell['source'][-1].rstrip('\n')
                    
        # 3. VQC
        if 'def cost_vqc(params, X, y):' in source:
            if 'ThreadPoolExecutor' not in source:
                new_source = source.replace(
                    "predictions = [vqc_circuit(x, params) for x in X]",
                    "with concurrent.futures.ThreadPoolExecutor() as executor:\n        predictions = list(executor.map(lambda x: vqc_circuit(x, params), X))"
                ).replace(
                    "predictions = pnp.array([vqc_circuit(x, params) for x in X])",
                    "with concurrent.futures.ThreadPoolExecutor() as executor:\n        predictions = list(executor.map(lambda x: vqc_circuit(x, params), X))\n    predictions = pnp.array(predictions)"
                )
                cell['source'] = [line + '\n' for line in new_source.split('\n')[:-1]]
                if not new_source.endswith('\n'):
                    cell['source'][-1] = cell['source'][-1].rstrip('\n')
                    
        # 4. QNN
        if 'estimator  = StatevectorEstimator()' in source:
            if 'AerEstimator' not in source:
                new_source = source.replace(
                    "estimator  = StatevectorEstimator()",
                    "estimator  = AerEstimator(run_options={'max_parallel_threads': 0})"
                )
                cell['source'] = [line + '\n' for line in new_source.split('\n')[:-1]]
                if not new_source.endswith('\n'):
                    cell['source'][-1] = cell['source'][-1].rstrip('\n')


with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)

print("Notebook modificado com sucesso para habilitar paralelismo.")
