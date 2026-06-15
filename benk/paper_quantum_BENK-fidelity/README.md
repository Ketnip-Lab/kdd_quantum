# Quantum-BENK Framework

Este repositório contém a implementação do framework **Quantum-BENK**, uma evolução do método descrito no artigo "BENK: the Beran Estimator with Neural Kernels". O objetivo deste projeto é substituir o kernel de rede neural clássico por um Circuito Quântico Variacional (VQC) para estimar o Efeito do Tratamento Condicional (CATE) em dados de sobrevida com censura.

## Estrutura do Projeto

```
paper_quantum_BENK-main/
├── config.json            # Hiperparâmetros globais do experimento
├── data_generator.py      # Gerador de dados sintéticos parametrizável
├── quantum_kernel.py      # Kernel Quântico Variacional (VQC) via PennyLane
├── beran_estimator.py     # Estimador de Beran com pesos quânticos
├── main.py                # Pipeline principal de treino e avaliação
├── compare_results.py     # Benchmark contra modelos clássicos
├── stress_test.py         # Pipeline de testes de estresse (4 cenários)
├── requirements.txt       # Dependências Python
└── data/
    ├── datasets/          # Datasets gerados por main.py e stress_test.py
    └── stress_tests/      # Resultados dos testes de estresse (CSV)
```

### Descrição dos Módulos

- **`config.json`**: Arquivo de configuração onde é possível alterar hiperparâmetros como número de qubits, camadas do ansatz (`n_layers`), taxa de aprendizado, número de épocas, seed e constante gamma do RBF.

- **`data_generator.py`**: Gera datasets sintéticos de sobrevida com parâmetros totalmente configuráveis. Suporta três topologias de covariáveis:
  - **Espiral** — $x_k = t \cdot \sin(t \cdot k),\; x_{k+1} = t \cdot \cos(t \cdot k)$ com $t \sim \mathcal{U}(0, 10)$
  - **Logarítmica** — $x_k = a_i \cdot \ln(t)$ com $t \sim \mathcal{U}(0.5, 5)$, $a_i \sim \mathcal{U}([-4,-1] \cup [1,4])$
  - **Potência** — $x_k = t^{k/\sqrt{d}}$ com $t \sim \mathcal{U}(0, 10)$

  Os tempos de evento baseiam-se em equações não-lineares:
  - **Controle**: $f(t) = \frac{-\ln(0.02)}{0.1 \cdot \exp(0.5 \cdot t)}$
  - **Tratamento**: $h(t) = \frac{-\ln(0.3)}{0.1 \cdot \exp(0.15 \cdot t)}$

  A função aceita os seguintes parâmetros de estresse: `c` (tamanho do grupo controle), `q` (proporção tratamento/controle), `p` (taxa de censura) e `epsilon` (nível de ruído gaussiano).

- **`quantum_kernel.py`**: Implementa o Kernel Quântico (VQC) no PennyLane. As características $x$ entram no `AngleEmbedding`, seguidas por camadas variacionais (`StronglyEntanglingLayers`) com parâmetros $\theta$. O estado quântico gera os valores esperados $\vec{v}(x) = [\langle Z_0 \rangle, \dots, \langle Z_{d-1} \rangle]$. A similaridade entre os pacientes é definida via RBF:
  $$K(z, x_i) = \exp\left(-\gamma \|\vec{v}(z) - \vec{v}(x_i)\|^2\right)$$

- **`beran_estimator.py`**: Contém o cérebro estatístico do framework. Calcula os pesos de Nadaraya-Watson:
  $$W(z, x_i) = \frac{K(z, x_i)}{\sum_j K(z, x_j)}$$
  E constrói a Função de Sobrevivência de Beran (onde $\delta_i=1$ se não-censurado e $0$ se censurado):
  $$S(t|z) = \prod_{T_i \le t} \left\{ 1 - \frac{W(z, x_i)}{1 - \sum_{j=1}^{i-1} W(z, x_j)} \right\}^{\delta_i}$$
  A expectativa de vida (tempo esperado até o evento) é a integral desta curva:
  $$\tilde{E}(z) = \int_{0}^{\infty} S(z+t \mid z) \, dt$$

- **`main.py`**: Pipeline principal. A função de perda treinada no grupo de controle minimiza o RMSE entre a expectativa de vida prevista e o tempo real, apenas nos pacientes **não-censurados**:
  $$\mathcal{L}(\theta) = \sqrt{\frac{1}{N_{u}} \sum_{\delta_i=1} (\tilde{E}(x_i, \theta) - T_i)^2}$$
  Após otimizar o kernel, o CATE ($\tau$) para o tratamento é estimado por:
  $$\tau(z) = \tilde{E}_{\text{tratamento}}(z) - \tilde{E}_{\text{controle}}(z)$$

- **`stress_test.py`**: Pipeline automatizado de testes de estresse. Avalia a robustez do Q-BENK sob quatro cenários, iterando sobre as três topologias. Veja a seção [Testes de Estresse](#testes-de-estresse) abaixo.

---

## Requisitos e Instalação

É recomendável utilizar um ambiente virtual (`.venv`).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*(Pacotes necessários: `pennylane`, `jax`, `jaxlib`, `optax`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `lifelines` e `tqdm`)*

---

## Uso

### 1. Executando um Único Experimento

Para rodar a simulação para a função configurada no `config.json` (`Spiral`, `Logarithmic` ou `Power`):

```bash
python main.py
```

Isso irá:
- Treinar o modelo quântico na topologia escolhida.
- Salvar o dataset gerado em `data/datasets/<funcao>_dataset.csv`.
- Avaliar o CATE no grupo de tratamento.
- Exibir as métricas MAE, MSE e RMSE do CATE estimado.

### 2. Executando o Benchmark Completo

Para comparar o Q-BENK com modelos clássicos (Cox, BENK Clássico etc.) nas 3 topologias:

```bash
python compare_results.py
```

Isso irá:
- Executar o `main.py` em loop sobre `Spiral`, `Logarithmic` e `Power`.
- Gerar e salvar os 3 datasets em `data/datasets/`.
- Criar o gráfico consolidado `data/combined_comparison_plot.png`.
- Gerar a tabela de ranking em `data/updated_baselines.csv`.

### 3. Testes de Estresse

Para avaliar a robustez do Q-BENK em 4 cenários de estresse sistemáticos:

```bash
python stress_test.py
```

---

## Testes de Estresse

O script `stress_test.py` implementa um pipeline automatizado que varia **um parâmetro por vez**, mantendo os demais fixos nos valores de âncora, e calcula o RMSE do CATE estimado para cada combinação parâmetro × topologia.

### Valores de Âncora

| Parâmetro | Símbolo | Valor Fixo |
|-----------|---------|------------|
| Grupo controle | $c$ | 200 |
| Proporção tratamento/controle | $q$ | 0.2 |
| Taxa de censura | $p$ | 0.25 |
| Nível de ruído | $\varepsilon$ | 0.05 |

### Cenários

| # | Nome | Parâmetro Variado | Valores Testados |
|---|------|-------------------|-----------------|
| 1 | Escalabilidade de Amostra | $c$ (controles) | 100, 200, 300, 500, 1000 |
| 2 | Robustez ao Ruído | $\varepsilon$ (ruído) | 0%, 5%, 10%, 15% |
| 3 | Desbalanceamento de Grupos | $q$ (proporção trat.) | 10%, 20%, 30%, 40%, 50% |
| 4 | Sensibilidade à Censura | $p$ (censura) | 10%, 20%, 30%, 40%, 50% |

Cada cenário é testado sobre as **3 topologias** (Espiral, Logarítmica, Potência), totalizando **60 execuções completas** de treino.

### Saídas Geradas

```
data/stress_tests/
├── scenario_1_sample_scalability.csv    # Cenário 1: variação de c
├── scenario_2_noise_robustness.csv      # Cenário 2: variação de ε
├── scenario_3_group_imbalance.csv       # Cenário 3: variação de q
├── scenario_4_censorship_sensitivity.csv # Cenário 4: variação de p
└── stress_test_all_results.csv          # Consolidado de todos os cenários
```

Cada CSV contém as colunas: `topology`, `parameter_name`, `parameter_value`, `rmse`, `elapsed_sec`, `status`.

---

## Configuração

Todas as opções do experimento podem ser ajustadas no `config.json`, sem necessidade de alterar o código-fonte.

```json
{
    "quantum": {
        "n_qubits": 10,
        "n_layers": 2,
        "gamma": 1.0,
        "seed": 42
    },
    "execution": {
        "use_parallel": true,
        "backend": "default.qubit"
    },
    "training": {
        "n_epochs": 20,
        "learning_rate": 0.1
    },
    "data": {
        "n_samples": 1000,
        "n_features": 10,
        "function_type": "all"
    }
}
```

### Parâmetros Disponíveis

| Seção | Parâmetro | Tipo | Descrição |
|-------|-----------|------|-----------|
| `quantum` | `n_qubits` | int | Número de qubits do circuito VQC. Deve ser igual a `n_features`. |
| `quantum` | `n_layers` | int | Número de camadas do Ansatz (`StronglyEntanglingLayers`). Mais camadas = mais expressividade. |
| `quantum` | `gamma` | float | Constante de escala do kernel RBF quântico. Valores maiores = similaridade mais localizada. |
| `quantum` | `seed` | int | Semente para reprodutibilidade dos pesos iniciais do VQC. |
| `execution` | `use_parallel` | bool | Ativa paralelismo em nível de hardware via flags XLA e OpenMP. |
| `execution` | `backend` | string | Backend do simulador quântico. Use `"default.qubit"` (padrão JAX) ou `"lightning.qubit"` (C++ rápido). |
| `training` | `n_epochs` | int | Número de épocas de otimização do kernel. Mais épocas = menor RMSE, mas mais lento. |
| `training` | `learning_rate` | float | Taxa de aprendizado do otimizador Adam. Sugerido: entre `0.01` e `0.3`. |
| `data` | `n_samples` | int | Total de amostras sintéticas geradas (usado em `main.py`; derivado como `c + round(c*q)`). |
| `data` | `n_features` | int | Dimensionalidade dos vetores de covariáveis. |
| `data` | `function_type` | string | Tipo de função geradora: `"Spiral"`, `"Logarithmic"`, `"Power"` ou `"all"` para rodar os três em sequência. |

---

## Características Principais

- **Aceleração JAX**: Todo o pipeline (kernel quântico e estimador de Beran) está vetorizado via `jax.vmap` e compilado com `jax.jit` para treinamento eficiente em CPU/GPU.
- **Sobrevivência com Censura**: A função de perda considera apenas amostras não-censuradas; a Função de Sobrevivência de Beran incorpora toda a informação através da distribuição de pesos ponderada.
- **Testes de Estresse Sistemáticos**: O script `stress_test.py` isola o efeito de cada parâmetro de forma independente, permitindo avaliar escalabilidade, robustez a ruído, desbalanceamento e sensibilidade à censura.
- **Retrocompatibilidade**: O `data_generator.py` aceita o parâmetro legado `n_samples` para não quebrar scripts existentes.
