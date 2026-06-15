import jax
import jax.numpy as jnp
import pennylane as qml
import json
import os

# Load config
config_path = os.path.join(os.path.dirname(__file__), "config.json")
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    n_qubits = config.get("quantum", {}).get("n_qubits", 10)
    backend = config.get("execution", {}).get("backend", "default.qubit")
    use_parallel = config.get("execution", {}).get("use_parallel", False)
else:
    n_qubits = 10
    backend = "default.qubit"
    use_parallel = False

# Configure the quantum device
dev = qml.device(backend, wires=n_qubits)

# Define the fidelity quantum kernel circuit
@qml.qnode(dev, interface="jax")
def fidelity_kernel_circuit(x1, x2, weights):
    """
    Quantum circuit that computes the transition amplitude |<ψ(x2)|ψ(x1)>|^2.
    It prepares the state |ψ(x1)> using AngleEmbedding and StronglyEntanglingLayers,
    then applies the adjoint (conjugate transpose) operations for x2.
    """
    # Prepares |ψ(x1)>
    qml.AngleEmbedding(x1, wires=range(n_qubits), rotation='Y')
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    # Applies U†(x2; θ)
    qml.adjoint(qml.StronglyEntanglingLayers)(weights, wires=range(n_qubits))
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits), rotation='Y')
    
    # Returns the probability distribution over all computational basis states
    return qml.probs(wires=range(n_qubits))

def single_kernel(x1, x2, weights):
    # The probability of observing |0...0> is the first element of the probs array
    return fidelity_kernel_circuit(x1, x2, weights)[0]

# Define the pairwise quantum kernel matrix using the fidelity kernel
@jax.jit
def quantum_kernel_matrix(X1, X2, weights, gamma=None):
    """
    Compute pairwise quantum fidelity kernel matrix.
    X1: shape (N, n_features)
    X2: shape (M, n_features)
    weights: quantum circuit weights
    gamma: unused parameter (kept for API compatibility)
    """
    # Double vmap over X1 and X2
    kernel_row = jax.vmap(single_kernel, in_axes=(None, 0, None))
    kernel_all = jax.vmap(kernel_row, in_axes=(0, None, None))
    return kernel_all(X1, X2, weights)

def initialize_weights(n_layers=None, seed=None):
    if n_layers is None:
        n_layers = config.get("quantum", {}).get("n_layers", 2) if 'config' in globals() else 2
    if seed is None:
        seed = config.get("quantum", {}).get("seed", 42) if 'config' in globals() else 42
        
    key = jax.random.PRNGKey(seed)
    # Shape for StronglyEntanglingLayers is (n_layers, n_qubits, 3)
    shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    return jax.random.uniform(key, shape=shape, minval=0.0, maxval=2 * jnp.pi)

if __name__ == "__main__":
    weights = initialize_weights()
    x1 = jnp.zeros((2, 10))
    x2 = jnp.ones((3, 10))
    K = quantum_kernel_matrix(x1, x2, weights)
    print("Kernel matrix shape:", K.shape)
    print("Kernel matrix:")
    print(K)
