import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def generate_synthetic_survival_data(
    c=200,
    q=0.2,
    p=0.25,
    epsilon=0.05,
    n_features=10,
    function_type="Spiral",
    seed=42,
    # Legacy compatibility: if n_samples is provided, derive c from it
    n_samples=None,
):
    """
    Generate synthetic survival data for Q-BENK experiments.

    Parameters
    ----------
    c : int
        Number of control units. Total sample size = c + round(c * q).
    q : float
        Ratio of treatment units to control units. E.g. 0.2 means 20% of c.
    p : float
        Right-censoring rate. delta ~ Binomial(1, 1 - p), i.e. p fraction
        of observations are censored.
    epsilon : float
        Standard deviation of Gaussian noise added to each covariate.
    n_features : int
        Dimensionality of the covariate vector X.
    function_type : str
        Topology for generating X. One of "Spiral", "Logarithmic", "Power".
    seed : int
        Random seed for reproducibility.
    n_samples : int or None
        Legacy parameter. If provided, overrides c (sets c = round(n_samples /
        (1 + q))) so that total ≈ n_samples. Emits a deprecation hint.
    """
    # --- Legacy compatibility ---
    if n_samples is not None:
        c = max(1, round(n_samples / (1.0 + q)))

    n_treatment = max(1, round(c * q))
    n_total = c + n_treatment

    np.random.seed(seed)

    X = np.zeros((n_total, n_features))

    if function_type == "Spiral":
        # Spiral: t in [0, 10]
        t = np.random.uniform(0, 10, size=n_total)
        for k in range(n_features // 2):
            idx = k + 1  # 1-indexed pair
            X[:, 2 * k]     = t * np.sin(t * idx)
            X[:, 2 * k + 1] = t * np.cos(t * idx)
        if n_features % 2 != 0:
            idx = int(np.ceil(n_features / 2))
            X[:, -1] = t * np.sin(t * idx)

        # Additive Gaussian noise controlled by epsilon
        if epsilon > 0:
            X += np.random.normal(0, epsilon, size=(n_total, n_features))

    elif function_type == "Logarithmic":
        # Logarithmic: t in [0.5, 5]
        t = np.random.uniform(0.5, 5.0, size=n_total)
        for k in range(n_features):
            a_i = np.where(
                np.random.rand(n_total) > 0.5,
                np.random.uniform(1, 4, size=n_total),
                np.random.uniform(-4, -1, size=n_total),
            )
            X[:, k] = a_i * np.log(t)

        # Additive Gaussian noise controlled by epsilon
        if epsilon > 0:
            X += np.random.normal(0, epsilon, size=(n_total, n_features))

    elif function_type == "Power":
        # Power: t in [0, 10]
        t = np.random.uniform(0, 10, size=n_total)
        for k in range(n_features):
            idx = k + 1
            X[:, k] = t ** (idx / np.sqrt(n_features))

        # Additive Gaussian noise controlled by epsilon
        if epsilon > 0:
            X += np.random.normal(0, epsilon, size=(n_total, n_features))

    else:
        raise ValueError(f"Unknown function_type: {function_type}. "
                         f"Choose from 'Spiral', 'Logarithmic', 'Power'.")

    # --- Normalization to [0, pi] ---
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X)

    # --- Treatment Assignment ---
    # First c units are control (A=0), last n_treatment units are treatment (A=1)
    A = np.zeros(n_total, dtype=int)
    A[c:] = 1

    # --- Event Times (based on latent t) ---
    # Control:   f(t) = -ln(0.02) / (0.1 * exp(0.5 * t))
    # Treatment: h(t) = -ln(0.3)  / (0.1 * exp(0.15 * t))
    time_control   = -np.log(0.02) / (0.1 * np.exp(0.5 * t))
    time_treatment = -np.log(0.3)  / (0.1 * np.exp(0.15 * t))

    T_true = np.where(A == 1, time_treatment, time_control)

    # --- Censoring: rate p ---
    # delta=1 means event observed, delta=0 means censored
    delta = np.random.binomial(1, 1.0 - p, size=n_total)

    # Observed Time: censored units get a uniform draw in (0, T_true)
    C = np.random.uniform(0, T_true)
    T_obs = np.where(delta == 1, T_true, C)

    data = pd.DataFrame(X_scaled, columns=[f'X_{i}' for i in range(n_features)])
    data['A']       = A
    data['T_obs']   = T_obs
    data['delta']   = delta
    data['T_true']  = T_true
    data['t_latent'] = t

    return data, scaler


if __name__ == "__main__":
    df, _ = generate_synthetic_survival_data()

    import os
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/synthetic_survival_dataset.csv", index=False)
    print("Dataset salvo em data/synthetic_survival_dataset.csv")

    print("\nSurvival summary:")
    print(df.groupby('A')[['T_obs', 'delta']].mean())
