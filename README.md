# 🔬 NSL-KDD Quantum Analysis

Ambiente de análise do dataset **NSL-KDD** utilizando algoritmos clássicos e quânticos de Machine Learning para detecção de intrusão de rede.

## 📂 Estrutura

```
Quantum/
├── README.md
├── setup_env.sh                    # Script de instalação do ambiente
├── requirements.txt                # Dependências clássicas
├── requirements_quantum.txt        # Dependências quânticas
├── data/
│   └── download_data.sh            # Script para baixar o NSL-KDD
└── notebooks/
    ├── 01_classical_algorithms.ipynb  # Algoritmos Clássicos
    └── 02_quantum_algorithms.ipynb   # Algoritmos Quânticos
```

## 🚀 Instalação

```bash
chmod +x setup_env.sh
./setup_env.sh
```

O script irá:
1. Criar o ambiente virtual `venv_quantum`
2. Instalar todas as dependências
3. Registrar o kernel Jupyter `NSL-KDD Quantum`

## 📥 Dataset NSL-KDD

```bash
chmod +x data/download_data.sh
./data/download_data.sh
```

O dataset NSL-KDD é uma versão refinada do KDD Cup 1999, amplamente usado como benchmark para sistemas de detecção de intrusão (IDS). Contém 41 features de tráfego de rede classificadas em:
- **Normal** — tráfego legítimo
- **DoS** — Denial of Service
- **Probe** — varredura/sondagem
- **R2L** — acesso remoto não autorizado
- **U2R** — escalação de privilégio local

## 📓 Notebooks

| Notebook | Descrição |
|---|---|
| `01_classical_algorithms.ipynb` | Naive Bayes, Decision Tree, KNN, Random Forest, SVM |
| `02_quantum_algorithms.ipynb` | QSVM, VQC, QNN (Qiskit + PennyLane) |

## 🔧 Executar Jupyter

```bash
source venv_quantum/bin/activate
jupyter notebook
```

## 📦 Dependências Principais

**Clássicas:** scikit-learn, pandas, numpy, matplotlib, seaborn

**Quânticas:** Qiskit ≥ 1.0, qiskit-machine-learning, PennyLane, pennylane-qiskit

---

> ⚠️ **Nota:** Os algoritmos quânticos rodam em simuladores locais (não requerem hardware quântico real). A execução pode ser lenta para datasets grandes — os notebooks usam subconjuntos otimizados.
