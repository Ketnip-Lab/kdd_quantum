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

# Define the quantum node that returns expected values
@qml.qnode(dev, interface="jax")
def quantum_features(x, weights):
    """
    Parametrized quantum circuit returning Pauli-Z expectation values.
    x: input feature vector of shape (n_qubits,)
    weights: trainable weights for the strongly entangling layers
    """
    qml.AngleEmbedding(x, wires=range(n_qubits), rotation='Y')
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Vectorized version for batching using JAX's vmap
vmap_quantum_features = jax.vmap(quantum_features, in_axes=(0, None))

# Define the RBF kernel using the quantum features
@jax.jit
def quantum_kernel_matrix(X1, X2, weights, gamma=None):
    """
    Compute pairwise quantum kernel matrix.
    X1: shape (N, n_features)
    X2: shape (M, n_features)
    weights: quantum circuit weights
    """
    if gamma is None:
        gamma = config.get("quantum", {}).get("gamma", 1.0) if 'config' in globals() else 1.0
    # Get quantum feature vectors
    V1 = vmap_quantum_features(X1, weights) # Shape: (n_qubits, N) due to list of expvals or (N, n_qubits)
    V2 = vmap_quantum_features(X2, weights) 
    
    # Pennylane returning a list of expvals via vmap results in a list of arrays.
    # We stack them to get (N, n_qubits)
    if isinstance(V1, (list, tuple)):
        V1 = jnp.stack(V1, axis=-1)
    if isinstance(V2, (list, tuple)):
        V2 = jnp.stack(V2, axis=-1)

    # Compute pairwise squared distances: ||V1_i - V2_j||^2
    # V1 shape: (N, n_qubits), V2 shape: (M, n_qubits)
    V1_sq = jnp.sum(V1**2, axis=1, keepdims=True) # (N, 1)
    V2_sq = jnp.sum(V2**2, axis=1, keepdims=True) # (M, 1)
    dists = V1_sq - 2 * jnp.dot(V1, V2.T) + V2_sq.T # (N, M)
    
    # Apply RBF kernel
    return jnp.exp(-gamma * dists)

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
    print(K)
