"""
generate_data.py

Generates synthetic stock return data from a latent factor model.

Model:
    x_{i,t} = lambda_i' @ f_t + epsilon_{i,t}        (observation equation)
    f_t = A @ f_{t-1} + eta_t,  eta_t ~ N(0, Sigma_f) (factor dynamics, VAR(1))
    epsilon_{i,t} = rho_i * epsilon_{i,t-1} + sigma_i * xi_{i,t} (idiosyncratic AR(1))

Outputs:
    returns_{tag}.npy      shape: (n_timesteps, n_stocks)
    metadata_{tag}.json    all generation params + SNR per stock
"""

import numpy as np
import json
import argparse
import os
from datetime import datetime


def make_stable_matrix(K: int, spectral_radius: float, rng: np.random.Generator) -> np.ndarray:
    """
    Build a K×K matrix with a given spectral radius.
    spectral_radius close to 1 = highly persistent factors (trending)
    spectral_radius close to 0 = fast mean-reverting factors
    """
    A = rng.standard_normal((K, K))
    eigenvalues = np.linalg.eigvals(A)
    current_radius = np.max(np.abs(eigenvalues))
    A = A * (spectral_radius / current_radius)
    return A


def generate_factors(
    n_timesteps: int,
    K: int,
    A: np.ndarray,
    sigma_f: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate factor paths via VAR(1):
        f_t = A @ f_{t-1} + eta_t,  eta_t ~ N(0, sigma_f^2 * I)
    Returns: (n_timesteps, K)
    """
    factors = np.zeros((n_timesteps, K))
    factors[0] = rng.standard_normal(K) * sigma_f
    for t in range(1, n_timesteps):
        shock = rng.standard_normal(K) * sigma_f
        factors[t] = A @ factors[t - 1] + shock
    return factors


def generate_idiosyncratic_noise(
    n_timesteps: int,
    n_stocks: int,
    rho: np.ndarray,
    sigma: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate idiosyncratic noise via AR(1):
        epsilon_{i,t} = rho_i * epsilon_{i,t-1} + sigma_i * xi_{i,t}
    Returns: (n_timesteps, n_stocks)
    """
    eps = np.zeros((n_timesteps, n_stocks))
    eps[0] = rng.standard_normal(n_stocks) * sigma
    for t in range(1, n_timesteps):
        shock = rng.standard_normal(n_stocks) * sigma
        eps[t] = rho * eps[t - 1] + shock
    return eps


def compute_snr(
    loadings: np.ndarray,
    factors: np.ndarray,
    noise: np.ndarray,
) -> np.ndarray:
    """
    Per-stock signal-to-noise ratio:
        SNR_i = Var(lambda_i' f_t) / Var(epsilon_{i,t})
    Returns: (n_stocks,)
    """
    signal = factors @ loadings.T
    signal_var = np.var(signal, axis=0)
    noise_var = np.var(noise, axis=0)
    return signal_var / (noise_var + 1e-10)


def generate(
    n_stocks: int = 50,
    n_timesteps: int = 2000,
    n_factors: int = 3,
    spectral_radius: float = 0.95,
    sigma_f: float = 1.0,
    sigma_eps: float = 1.0,
    rho_range: tuple = (0.0, 0.3),
    seed: int = 42,
) -> dict:
    """Main generation function. Returns a dict with all arrays and metadata."""
    rng = np.random.default_rng(seed)

    A = make_stable_matrix(n_factors, spectral_radius, rng)
    loadings = rng.standard_normal((n_stocks, n_factors))
    rho = rng.uniform(rho_range[0], rho_range[1], size=n_stocks)
    sigma = np.full(n_stocks, sigma_eps)

    factors = generate_factors(n_timesteps, n_factors, A, sigma_f, rng)
    noise = generate_idiosyncratic_noise(n_timesteps, n_stocks, rho, sigma, rng)

    signal = factors @ loadings.T
    returns = signal + noise

    snr = compute_snr(loadings, factors, noise)

    return {
        "returns": returns,
        "factors": factors,
        "loadings": loadings,
        "noise": noise,
        "A": A,
        "rho": rho,
        "snr": snr,
        "params": {
            "n_stocks": n_stocks,
            "n_timesteps": n_timesteps,
            "n_factors": n_factors,
            "spectral_radius": spectral_radius,
            "sigma_f": sigma_f,
            "sigma_eps": sigma_eps,
            "rho_range": list(rho_range),
            "seed": seed,
            "mean_snr": float(np.mean(snr)),
            "median_snr": float(np.median(snr)),
        },
    }


def save(result: dict, output_dir: str, tag: str):
    """Save returns array and metadata to disk."""
    os.makedirs(output_dir, exist_ok=True)

    returns_path = os.path.join(output_dir, f"returns_{tag}.npy")
    np.save(returns_path, result["returns"].astype(np.float32))

    gt_path = os.path.join(output_dir, f"ground_truth_{tag}.npz")
    np.savez(
        gt_path,
        factors=result["factors"],
        loadings=result["loadings"],
        noise=result["noise"],
        A=result["A"],
        rho=result["rho"],
        snr=result["snr"],
    )

    meta_path = os.path.join(output_dir, f"metadata_{tag}.json")
    with open(meta_path, "w") as f:
        json.dump(result["params"], f, indent=2)

    print(f"Saved: {returns_path}")
    print(f"Saved: {gt_path}")
    print(f"Saved: {meta_path}")
    print(f"  mean SNR: {result['params']['mean_snr']:.3f}")
    print(f"  median SNR: {result['params']['median_snr']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic factor model data")
    parser.add_argument("--n_stocks", type=int, default=50)
    parser.add_argument("--n_timesteps", type=int, default=2000)
    parser.add_argument("--n_factors", type=int, default=3)
    parser.add_argument("--spectral_radius", type=float, default=0.95)
    parser.add_argument("--sigma_f", type=float, default=1.0)
    parser.add_argument("--sigma_eps", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    tag = args.tag or f"snr{args.sigma_f / args.sigma_eps:.2f}_seed{args.seed}"

    result = generate(
        n_stocks=args.n_stocks,
        n_timesteps=args.n_timesteps,
        n_factors=args.n_factors,
        spectral_radius=args.spectral_radius,
        sigma_f=args.sigma_f,
        sigma_eps=args.sigma_eps,
        seed=args.seed,
    )

    save(result, args.output_dir, tag)
