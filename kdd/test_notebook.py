import json
import ast

notebook_path = '/home/renatodias/Quantum/kdd/notebooks/02_quantum_algorithms.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        try:
            ast.parse(source)
        except Exception as e:
            print(f"Error in cell {i}: {e}")
            print(source)

print("Sintaxe verificada.")
