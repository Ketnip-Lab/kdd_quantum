import jax
import jax.numpy as jnp
import os
import json
from quantum_kernel import quantum_kernel_matrix

config_path = os.path.join(os.path.dirname(__file__), "config.json")
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    DEFAULT_GAMMA = config.get("quantum", {}).get("gamma", 1.0)
else:
    DEFAULT_GAMMA = 1.0

@jax.jit
def compute_weights(Z, X_ref, weights, gamma=DEFAULT_GAMMA):
    """
    Compute Nadaraya-Watson weights.
    Z: query data (N_z, n_features)
    X_ref: reference data (N_ref, n_features)
    weights: quantum kernel parameters
    Returns: W of shape (N_z, N_ref)
    """
    K = quantum_kernel_matrix(Z, X_ref, weights, gamma) # (N_z, N_ref)
    # Normalize across reference data
    K_sum = jnp.sum(K, axis=1, keepdims=True)
    # Add epsilon to prevent division by zero
    W = K / (K_sum + 1e-8)
    return W

@jax.jit
def beran_survival_function(Z, X_ref, T_ref, delta_ref, weights, gamma=DEFAULT_GAMMA):
    """
    Compute the Beran survival function S(t|Z).
    Assumes X_ref, T_ref, delta_ref are ALREADY SORTED by T_ref in ascending order.
    Z: (N_z, n_features)
    X_ref: (N_ref, n_features)
    T_ref: (N_ref,)
    delta_ref: (N_ref,)
    weights: quantum kernel parameters
    
    Returns: S_t matrix of shape (N_z, N_ref), where S_t[i, j] is S(T_ref[j] | Z[i])
    """
    W = compute_weights(Z, X_ref, weights, gamma) # (N_z, N_ref)
    
    # We need to compute S(t|z) = prod_{T_i <= t} [1 - W(z, x_i) / (1 - sum_{j=1}^{i-1} W(z, x_j))]^delta_i
    
    # Compute cumulative sum of W along axis 1 (across reference points)
    # sum_{j=1}^{i-1} W(z, x_j)
    # jnp.cumsum includes the current element, so we need to shift it
    W_cumsum = jnp.cumsum(W, axis=1)
    
    # We want sum up to i-1. So we shift by 1 and prepend 0
    W_cumsum_prev = jnp.pad(W_cumsum[:, :-1], ((0, 0), (1, 0)), mode='constant')
    
    # Compute the term inside the product:
    # 1 - W_i / (1 - W_cumsum_prev_i)
    denominator = 1.0 - W_cumsum_prev
    denominator = jnp.clip(denominator, min=1e-8)
    
    term = 1.0 - (W / denominator)
    # Clip term to [0, 1] to avoid negative probabilities and log of negative
    term = jnp.clip(term, min=1e-8, max=1.0)
    
    # Apply delta_i (if delta_i = 0, term^0 = 1)
    delta_ref_expanded = jnp.expand_dims(delta_ref, axis=0) # (1, N_ref)
    term_with_delta = jnp.where(delta_ref_expanded == 1, term, 1.0)
    
    # Cumulative product to get S(t|Z)
    S_t = jnp.cumprod(term_with_delta, axis=1)
    
    return S_t

@jax.jit
def expected_survival_time(Z, X_ref, T_ref, delta_ref, weights, gamma=DEFAULT_GAMMA):
    """
    Compute expected survival time: E(Z) = integral S(t|Z) dt
    This is approximately sum_{j=1}^{N_ref} S(T_{j-1}|Z) * (T_j - T_{j-1})
    """
    S_t = beran_survival_function(Z, X_ref, T_ref, delta_ref, weights, gamma) # (N_z, N_ref)
    
    # Prepend T_0 = 0 and S(T_0|Z) = 1
    T_ref_pad = jnp.pad(T_ref, (1, 0), mode='constant') # (N_ref + 1,)
    S_t_pad = jnp.pad(S_t, ((0, 0), (1, 0)), mode='constant', constant_values=1.0) # (N_z, N_ref + 1)
    
    # dt = T_j - T_{j-1}
    dt = T_ref_pad[1:] - T_ref_pad[:-1] # (N_ref,)
    
    # Integrate: S(T_{j-1}) * dt
    E_z = jnp.sum(S_t_pad[:, :-1] * jnp.expand_dims(dt, axis=0), axis=1)
    
    return E_z

@jax.jit
def expected_survival_time_treatment(Z, X_ref_0, T_ref_0, delta_ref_0, X_ref_1, T_ref_1, delta_ref_1, weights, gamma=DEFAULT_GAMMA):
    """
    Compute expected survival using Control (0) reference and Treatment (1) reference
    """
    E_0 = expected_survival_time(Z, X_ref_0, T_ref_0, delta_ref_0, weights, gamma)
    E_1 = expected_survival_time(Z, X_ref_1, T_ref_1, delta_ref_1, weights, gamma)
    return E_0, E_1

@jax.jit
def loss_fn(weights, Z, T_true, delta_true, X_ref, T_ref, delta_ref, gamma=DEFAULT_GAMMA):
    """
    Compute RMSE between expected survival time and true time for UNCENSORED samples.
    """
    E_z = expected_survival_time(Z, X_ref, T_ref, delta_ref, weights, gamma)
    
    # Filter uncensored
    uncensored_mask = (delta_true == 1)
    
    # Calculate MSE on uncensored only
    diff = E_z - T_true
    
    # We do a masked mean
    num_uncensored = jnp.sum(uncensored_mask)
    # prevent division by zero
    num_uncensored = jnp.maximum(num_uncensored, 1.0)
    
    mse = jnp.sum(jnp.where(uncensored_mask, diff**2, 0.0)) / num_uncensored
    
    return jnp.sqrt(mse)
