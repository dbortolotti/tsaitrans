"""
generate_data.py

Generates synthetic stock return data from a latent factor model.

Model:
    x_{i,t} = lambda_i' @ f_t + epsilon_{i,t}        (observation equation)
    f_t = A @ f_{t-1} + eta_t,  eta_t ~ N(0, I)      (factor dynamics, VAR(1))
    epsilon_{i,t} = rho_i * epsilon_{i,t-1} + xi_{i,t} (idiosyncratic AR(1))

Primary parameters:
    target_vol          — daily volatility (std of cumulative return over one day)
    snr                 — signal variance / noise variance
    factor_half_life    — factor mean-reversion half-life as fraction of a trading day
    noise_half_life_range — (min, max) idiosyncratic half-life as fraction of a trading day
    n_factors           — dimensionality of the shared signal space
    steps_per_day       — number of timesteps in one trading day

Per-step parameters are derived from the half-lives:
    spectral_radius = 0.5 ** (1 / (factor_half_life * steps_per_day))
    rho_i           = 0.5 ** (1 / (noise_half_life_i * steps_per_day))

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
    factor_half_life: float = 0.1,
    noise_half_life_range: tuple = (0.005, 0.025),
    target_vol: float = 0.02,
    snr: float = 0.3,
    steps_per_day: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Main generation function. Returns a dict with all arrays and metadata.

    factor_half_life and noise_half_life_range are expressed as fractions of a
    trading day. Per-step AR parameters are derived as:
        spectral_radius = 0.5 ** (1 / (factor_half_life * steps_per_day))
        rho_i           = 0.5 ** (1 / (noise_half_life_i * steps_per_day))

    target_vol is the DAILY volatility (std of cumulative return over one day).
    Per-step vol = target_vol / sqrt(steps_per_day).
    """
    rng = np.random.default_rng(seed)

    # Derive per-step persistence parameters from half-lives
    spectral_radius = 0.5 ** (1.0 / (factor_half_life * steps_per_day))
    hl_low, hl_high = noise_half_life_range
    rho_low  = 0.5 ** (1.0 / (hl_low  * steps_per_day))
    rho_high = 0.5 ** (1.0 / (hl_high * steps_per_day))

    A = make_stable_matrix(n_factors, spectral_radius, rng)
    loadings = rng.standard_normal((n_stocks, n_factors))
    rho = rng.uniform(rho_low, rho_high, size=n_stocks)

    # Generate at unit scale
    factors = generate_factors(n_timesteps, n_factors, A, sigma_f=1.0, rng=rng)
    noise = generate_idiosyncratic_noise(n_timesteps, n_stocks, rho, sigma=np.ones(n_stocks), rng=rng)
    signal = factors @ loadings.T

    # Step 1: set per-step SNR ratio using i.i.d. approximation.
    # This correctly partitions variance between signal and noise.
    per_step_vol = target_vol / np.sqrt(steps_per_day)
    sigma_signal_target = per_step_vol * np.sqrt(snr / (1.0 + snr))
    sigma_noise_target  = per_step_vol / np.sqrt(1.0 + snr)

    signal_scale = sigma_signal_target / np.std(signal)
    noise_scale  = sigma_noise_target  / np.std(noise)

    signal  = signal  * signal_scale
    factors = factors * signal_scale   # keep factors consistent with signal
    noise   = noise   * noise_scale

    returns = signal + noise

    # Step 2: correct for autocorrelation. The AR factor and noise structure
    # inflate the daily vol (std of sum over one day) well beyond the i.i.d.
    # assumption. Rescale everything uniformly so the empirical daily vol
    # matches target_vol. Using sum as a fast approximation of cumprod-1
    # (valid since per-step returns are O(1e-4)).
    n_days = n_timesteps // steps_per_day
    if n_days > 1:
        # Multiple days: compute daily sums and take std across all (days × stocks)
        usable = n_days * steps_per_day
        daily_sums = returns[:usable].reshape(n_days, steps_per_day, n_stocks).sum(axis=1)
        actual_daily_std = np.std(daily_sums)
    elif n_stocks > 1:
        # Single day, multiple stocks: std across stocks
        actual_daily_std = np.std(np.sum(returns, axis=0))
    else:
        # Single day, single stock: can't estimate empirically, skip correction
        actual_daily_std = target_vol

    daily_correction = target_vol / actual_daily_std
    returns *= daily_correction
    signal  *= daily_correction
    factors *= daily_correction
    noise   *= daily_correction

    empirical_snr = compute_snr(loadings, factors, noise)

    return {
        "returns": returns,
        "factors": factors,
        "loadings": loadings,
        "noise": noise,
        "A": A,
        "rho": rho,
        "snr": empirical_snr,
        "params": {
            "n_stocks": n_stocks,
            "n_timesteps": n_timesteps,
            "n_factors": n_factors,
            "factor_half_life": factor_half_life,
            "noise_half_life_range": list(noise_half_life_range),
            "derived_spectral_radius": spectral_radius,
            "derived_rho_range": [float(rho_low), float(rho_high)],
            "target_vol": target_vol,
            "snr": snr,
            "steps_per_day": steps_per_day,
            "seed": seed,
            "empirical_per_step_vol": float(np.std(returns)),
            "empirical_daily_vol": float(np.std(returns) * np.sqrt(steps_per_day)),
            "mean_snr": float(np.mean(empirical_snr)),
            "median_snr": float(np.median(empirical_snr)),
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
    parser.add_argument("--factor_half_life", type=float, default=0.1)
    parser.add_argument("--noise_half_life_min", type=float, default=0.005)
    parser.add_argument("--noise_half_life_max", type=float, default=0.025)
    parser.add_argument("--target_vol", type=float, default=0.02)
    parser.add_argument("--snr", type=float, default=0.3)
    parser.add_argument("--steps_per_day", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    tag = args.tag or f"vol{args.target_vol}_snr{args.snr}_fhl{args.factor_half_life}_seed{args.seed}"

    result = generate(
        n_stocks=args.n_stocks,
        n_timesteps=args.n_timesteps,
        n_factors=args.n_factors,
        factor_half_life=args.factor_half_life,
        noise_half_life_range=(args.noise_half_life_min, args.noise_half_life_max),
        target_vol=args.target_vol,
        snr=args.snr,
        steps_per_day=args.steps_per_day,
        seed=args.seed,
    )

    save(result, args.output_dir, tag)
