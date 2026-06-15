import json

notebook_path = '/home/renatodias/Quantum/kdd/notebooks/02_quantum_algorithms.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Fix cell 2
        if "print(f'   PennyLane version" in "".join(cell['source']):
            for i, line in enumerate(cell['source']):
                if "print(f'   Qiskit version: {qiskit.__version__}')" in line and not line.endswith('\n'):
                    cell['source'][i] = line + '\n'
                    break
        
        # Fix cell 14 and 18
        if "plt.show()" in "".join(cell['source']):
            for i, line in enumerate(cell['source']):
                if "plt.tight_layout()" in line and not line.endswith('\n') and i < len(cell['source'])-1 and "plt.show" in cell['source'][i+1]:
                    cell['source'][i] = line + '\n'
                    break
                    
        # Check QSVM cell dict
        if "QUANTUM_RESULTS['QSVM']" in "".join(cell['source']):
            for i, line in enumerate(cell['source']):
                if "'Train Time (s)': qsvm_train_time" in line and not line.endswith('\n') and i < len(cell['source'])-1 and "}" in cell['source'][i+1]:
                    cell['source'][i] = line + '\n'
                    break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)

print("Newlines corrigidas.")
