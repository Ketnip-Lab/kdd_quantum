"""
quantum_kernel.py — Kernel de Fidelidade Quântica para o Q-BENK
================================================================

IMPLEMENTAÇÃO: Kernel de Fidelidade Quântica
--------------------------------------------
Esta versão substitui o kernel RBF sobre features quânticas pela fidelidade
quântica genuína entre estados preparados pelo ansatz parametrizado.

TEORIA
------
Dado um ansatz U(x; θ) = StronglyEntanglingLayers(θ) ∘ AngleEmbedding(x),
o estado quântico associado a x é:

    |ψ(x)⟩ = U(x; θ)|0⟩

O kernel de fidelidade quântica é definido como:

    K_Q(xᵢ, xⱼ) = |⟨ψ(xⱼ)|ψ(xᵢ)⟩|²
                 = |⟨0| U†(xⱼ; θ) U(xᵢ; θ) |0⟩|²

IMPLEMENTAÇÃO VIA "ADJOINT TRICK"
----------------------------------
Em vez de calcular a sobreposição diretamente (o que exigiria memória de
estado completo), usamos um único circuito combinado:

    |0⟩ ─→ U(xᵢ; θ) ─→ U†(xⱼ; θ) ─→ medir P(|0…0⟩)

A probabilidade de medir o estado todo-zero é exatamente K_Q(xᵢ, xⱼ):

    P(|0…0⟩) = |⟨0| U†(xⱼ; θ) U(xᵢ; θ) |0⟩|² = K_Q(xᵢ, xⱼ)

PROPRIEDADES DO KERNEL DE FIDELIDADE
--------------------------------------
1. Simetria:         K_Q(xᵢ, xⱼ) = K_Q(xⱼ, xᵢ)
2. Auto-normalização: K_Q(x, x) = 1  (auto-fidelidade é sempre 1)
3. Range:            K_Q(xᵢ, xⱼ) ∈ [0, 1]
4. PSD por construção: a matriz kernel é positivo semi-definida (garantida)
5. Sem hiperparâmetros: não requer o parâmetro γ do kernel RBF

DIFERENÇA EM RELAÇÃO À IMPLEMENTAÇÃO ANTERIOR
-----------------------------------------------
Implementação anterior (paper_quantum_BENK-main/):
  - Circuito → φ(x) = [⟨Z₀⟩,...,⟨Z₉⟩] (features clássicas)
  - Kernel: exp(−γ ‖φ(xᵢ)−φ(xⱼ)‖²)  (RBF clássico)
  - Execuções do circuito: O(N) para N amostras

Esta implementação (paper_quantum_BENK-fidelity/):
  - Circuito combinado: U(xᵢ;θ) → U†(xⱼ;θ) → P(|0…0⟩)
  - Kernel: K_Q(xᵢ, xⱼ) = P(|0…0⟩) (fidelidade quântica)
  - Execuções do circuito: O(N²) para matriz N×N

NOTA DE PERFORMANCE
--------------------
Para N amostras de treinamento, a construção da matriz kernel requer N²
execuções do circuito. Com jax.vmap duplo, isso é vetorizado automaticamente.
Para datasets grandes (N > 300), recomenda-se reduzir n_samples no config.json.
"""

import jax
import jax.numpy as jnp
import pennylane as qml
import json
import os

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
config_path = os.path.join(os.path.dirname(__file__), "config.json")
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    n_qubits = config.get("quantum", {}).get("n_qubits", 10)
    backend  = config.get("execution", {}).get("backend", "default.qubit")
else:
    n_qubits = 10
    backend  = "default.qubit"

# Dispositivo quântico
dev = qml.device(backend, wires=n_qubits)


# ---------------------------------------------------------------------------
# Ansatz auxiliar — reutilizado dentro e fora do adjoint
# ---------------------------------------------------------------------------
def ansatz(x, weights):
    """
    Aplica o ansatz parametrizado ao estado atual dos qubits.

    Etapas:
      1. AngleEmbedding: codifica cada feature xₖ como rotação RY no qubit k.
         Isso mapeia x ∈ ℝ^n_qubits → espaço de estados de n_qubits qubits.
      2. StronglyEntanglingLayers: aplica L camadas de rotações (RZ-RY-RZ) +
         entanglement cíclico (CNOT), gerando entrelaçamento entre os qubits.

    Parâmetros
    ----------
    x       : vetor de features de entrada, shape (n_qubits,)
    weights : pesos treináveis, shape (n_layers, n_qubits, 3)
    """
    qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))


# ---------------------------------------------------------------------------
# Circuito do kernel de fidelidade (adjoint trick)
# ---------------------------------------------------------------------------
@qml.qnode(dev, interface="jax")
def fidelity_kernel_circuit(x1, x2, weights):
    """
    Circuito quântico que calcula K_Q(x1, x2) = |⟨ψ(x2)|ψ(x1)⟩|².

    Estratégia (adjoint trick):
      |0⟩ ─→ U(x1; θ) ─→ U†(x2; θ) ─→ medir probabilidades

    A probabilidade do estado |0…0⟩ é exatamente a fidelidade ao quadrado
    entre |ψ(x1)⟩ e |ψ(x2)⟩:

        P(|0…0⟩) = |⟨0| U†(x2;θ) U(x1;θ) |0⟩|²
                 = |⟨ψ(x2)|ψ(x1)⟩|²

    Parâmetros
    ----------
    x1, x2  : vetores de features, shape (n_qubits,)
    weights : pesos treináveis, shape (n_layers, n_qubits, 3)

    Retorna
    -------
    Array de probabilidades sobre todos os 2^n_qubits estados de base.
    O índice 0 corresponde ao estado |0…0⟩.
    """
    # Passo 1: prepara |ψ(x1)⟩ = U(x1; θ)|0⟩
    ansatz(x1, weights)

    # Passo 2: aplica o adjunto (inversa unitária) U†(x2; θ)
    # qml.adjoint inverte automaticamente a ordem das portas e
    # conjuga complexo todos os ângulos de rotação.
    qml.adjoint(ansatz)(x2, weights)

    # Passo 3: mede a distribuição de probabilidade sobre todos os estados
    # P(|0…0⟩) = K_Q(x1, x2)
    return qml.probs(wires=range(n_qubits))


# ---------------------------------------------------------------------------
# Função de kernel escalar
# ---------------------------------------------------------------------------
def single_kernel_value(x1, x2, weights):
    """
    Retorna K_Q(x1, x2) = P(|0…0⟩) ∈ [0, 1].

    Este é o valor escalar do kernel de fidelidade para um par (x1, x2).
    Apenas o índice [0] do vetor de probabilidades é necessário,
    pois ele corresponde ao estado |0…0⟩.

    Parâmetros
    ----------
    x1, x2  : vetores de features, shape (n_qubits,)
    weights : pesos treináveis, shape (n_layers, n_qubits, 3)

    Retorna
    -------
    Escalar em [0, 1].
    """
    probs = fidelity_kernel_circuit(x1, x2, weights)
    return probs[0]  # probabilidade do estado |0…0⟩


# ---------------------------------------------------------------------------
# Matriz de kernel (API pública — mesma assinatura do módulo anterior)
# ---------------------------------------------------------------------------
def quantum_kernel_matrix(X1, X2, weights, gamma=None):
    """
    Computa a matriz de kernel de fidelidade quântica K ∈ [0,1]^(N×M).

    Cada entrada K[i,j] = K_Q(X1[i], X2[j]) = |⟨ψ(X2[j])|ψ(X1[i])⟩|²

    Estratégia de vetorização:
      - jax.vmap interno: fixa x1, varre todos os x2 em X2 → linha da matriz
      - jax.vmap externo: varre todos os x1 em X1 → todas as linhas

    O parâmetro `gamma` é aceito por compatibilidade com o beran_estimator.py,
    mas é IGNORADO nesta implementação (o kernel de fidelidade não tem γ).

    Parâmetros
    ----------
    X1     : shape (N, n_features)
    X2     : shape (M, n_features)
    weights: shape (n_layers, n_qubits, 3)
    gamma  : ignorado (mantido para compatibilidade de API)

    Retorna
    -------
    K : jnp.ndarray de shape (N, M), valores em [0, 1]
    """
    # Linha i da matriz: K[i, :] = [K_Q(X1[i], X2[j]) para todo j]
    kernel_row = jax.vmap(single_kernel_value, in_axes=(None, 0, None))

    # Todas as linhas: K[:, :] = [kernel_row(X1[i]) para todo i]
    kernel_matrix = jax.vmap(kernel_row, in_axes=(0, None, None))

    return kernel_matrix(X1, X2, weights)


# ---------------------------------------------------------------------------
# Inicialização dos pesos (idêntico à versão anterior)
# ---------------------------------------------------------------------------
def initialize_weights(n_layers=None, seed=None):
    """
    Inicializa os pesos treináveis do ansatz com valores aleatórios uniformes
    no intervalo [0, 2π].

    Shape dos pesos: (n_layers, n_qubits, 3)
      - n_layers : número de camadas StronglyEntanglingLayers
      - n_qubits : número de qubits
      - 3        : três ângulos por qubit por camada (θ₁, θ₂, θ₃)

    Parâmetros
    ----------
    n_layers : int, padrão lido do config.json (2)
    seed     : int, semente PRNG, padrão lido do config.json (42)

    Retorna
    -------
    jnp.ndarray de shape (n_layers, n_qubits, 3)
    """
    if n_layers is None:
        n_layers = config.get("quantum", {}).get("n_layers", 2) if "config" in globals() else 2
    if seed is None:
        seed = config.get("quantum", {}).get("seed", 42) if "config" in globals() else 42

    key   = jax.random.PRNGKey(seed)
    shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    return jax.random.uniform(key, shape=shape, minval=0.0, maxval=2 * jnp.pi)


# ---------------------------------------------------------------------------
# Verificação de sanidade (executar com: python quantum_kernel.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Verificação do Kernel de Fidelidade Quântica")
    print("=" * 60)

    weights = initialize_weights()
    print(f"Pesos inicializados — shape: {weights.shape}")

    # Teste 1: auto-fidelidade deve ser 1.0
    x = jnp.ones(n_qubits) * 0.5
    k_self = single_kernel_value(x, x, weights)
    print(f"\n[Teste 1] Auto-fidelidade K_Q(x, x) = {k_self:.6f}  (esperado: 1.0)")

    # Teste 2: simetria K(x1, x2) == K(x2, x1)
    x1 = jnp.array([0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.0])
    x2 = jnp.array([0.9, 0.3, 0.1, 0.7, 0.5, 0.8, 0.2, 0.6, 0.4, 1.0])
    k12 = single_kernel_value(x1, x2, weights)
    k21 = single_kernel_value(x2, x1, weights)
    print(f"[Teste 2] Simetria: K(x1,x2) = {k12:.6f}, K(x2,x1) = {k21:.6f}  (esperado: iguais)")

    # Teste 3: matriz kernel 3×3
    X = jnp.array([x1, x2, x])
    K = quantum_kernel_matrix(X, X, weights)
    print(f"\n[Teste 3] Matriz kernel 3×3:")
    print(jnp.round(K, 4))
    print(f"  Diagonal (deve ser 1.0): {jnp.diag(K)}")

    # Teste 4: verificar PSD (autovalores ≥ 0)
    eigenvalues = jnp.linalg.eigvalsh(K)
    print(f"  Autovalores (devem ser ≥ 0): {eigenvalues}")

    print("\nVerificação concluída.")
