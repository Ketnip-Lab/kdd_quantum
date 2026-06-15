import json

notebook_path = '/home/renatodias/Quantum/kdd/notebooks/02_quantum_algorithms.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # 1. Imports
        if 'import pennylane as qml' in source and 'qiskit_aer.primitives' in source:
            if 'print(f\'   PennyLane' not in source:
                cell['source'].append("print(f'   PennyLane version: {qml.__version__}')")
                
        # 2. QSVM
        if 'qsvm = QSVC(quantum_kernel=quantum_kernel)' in source:
            if '}' not in cell['source'][-1] and 'Train Time (s)' in cell['source'][-1]:
                cell['source'].append("}")
                
        # 3. VQC
        if 'def cost_vqc(params, X, y):' in source:
            if 'plt.show()' not in cell['source'][-1]:
                cell['source'].append("plt.show()")
                
        # 4. QNN
        if 'estimator  = AerEstimator' in source:
            if 'plt.show()' not in cell['source'][-1]:
                cell['source'].append("plt.show()")


with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)

print("Notebook corrigido.")
