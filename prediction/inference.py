"""
inference.py

Loads a trained checkpoint, runs on test stocks, computes metrics, and plots.

Can be called standalone or imported by run_experiment.py.

Standalone usage:
    python inference.py --checkpoint checkpoints/run1 --data data/returns.npy \
        --test_stocks '[9]'

Outputs:
    <results_dir>/metrics.json
    <results_dir>/predictions.png
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import FactorTransformer, TimeSeriesDataset, get_device


def compute_metrics(pred: np.ndarray, target: np.ndarray, sigma: np.ndarray = None) -> dict:
    """
    pred, target: (n_samples, horizon, 1)
    sigma: (n_samples, horizon, 1) — optional, from probabilistic model
    Returns dict of scalar metrics.
    """
    mse = np.mean((pred - target) ** 2)
    mae = np.mean(np.abs(pred - target))
    rmse = np.sqrt(mse)

    # Directional accuracy
    dir_acc = np.mean(np.sign(pred[:, 0, :]) == np.sign(target[:, 0, :]))

    # Naive baseline: predict zero
    naive_mse = np.mean(target ** 2)
    r2 = 1.0 - mse / (naive_mse + 1e-10)

    metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "directional_accuracy": float(dir_acc),
        "naive_mse": float(naive_mse),
        "r2_vs_naive": float(r2),
    }

    if sigma is not None:
        # NLL (heteroscedastic)
        nll = np.mean((target - pred) ** 2 / (sigma ** 2 + 1e-8) + 2 * np.log(sigma + 1e-8))
        metrics["nll"] = float(nll)
        metrics["mean_sigma"] = float(np.mean(sigma))

        # Z-score signal: aggregate over horizon
        mu_total = pred.sum(axis=1)        # (n_samples, 1)
        sigma_total = np.sqrt((sigma ** 2).sum(axis=1))  # (n_samples, 1)
        z = mu_total / (sigma_total + 1e-8)              # (n_samples, 1)
        metrics["mean_abs_z"] = float(np.mean(np.abs(z)))
        metrics["median_abs_z"] = float(np.median(np.abs(z)))
        metrics["z_gt_1_frac"] = float(np.mean(np.abs(z) > 1.0))
        metrics["z_gt_2_frac"] = float(np.mean(np.abs(z) > 2.0))

    return metrics


def run_inference(
    checkpoint_dir: str,
    returns: np.ndarray,
    test_stocks: list,
    results_dir: str,
    n_plot_stocks: int = 5,
):
    """
    Run inference on test stocks.

    Args:
        checkpoint_dir: path to transformer checkpoint dir
        returns: (T, N_total) full returns array
        test_stocks: list of stock indices to evaluate on
        results_dir: where to save results
    """
    device = get_device()
    print(f"Using device: {device}")

    # Load checkpoint
    ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.5f})")

    # Load normalization stats
    mean = float(np.load(os.path.join(checkpoint_dir, "mean.npy")))
    std = float(np.load(os.path.join(checkpoint_dir, "std.npy")))

    context_len = config.get("context_len", 60)
    horizon = config.get("horizon", 1)

    # Test dataset
    test_ds = TimeSeriesDataset(returns, test_stocks, context_len, horizon, mean, std)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    print(f"Test stocks: {test_stocks} ({len(test_ds)} samples)")

    # Rebuild model
    probabilistic = config.get("probabilistic", False)
    model = FactorTransformer(
        n_stocks=1,
        context_len=context_len,
        horizon=horizon,
        d_model=config.get("d_model", 64),
        n_heads=config.get("n_heads", 4),
        n_layers=config.get("n_layers", 3),
        ffn_dim=config.get("ffn_dim", 256),
        dropout=config.get("dropout", 0.1),
        probabilistic=probabilistic,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Parameters: {model.count_parameters():,}")
    if probabilistic:
        print("Mode: probabilistic")

    # Run predictions
    all_preds = []
    all_targets = []
    all_sigmas = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            if probabilistic:
                mu, log_sigma = model(x)
                all_preds.append(mu.cpu().numpy())
                all_sigmas.append(torch.exp(log_sigma).cpu().numpy())
            else:
                pred = model(x)
                all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())

    preds = np.concatenate(all_preds, axis=0)    # (n_test, horizon, 1)
    targets = np.concatenate(all_targets, axis=0)
    sigmas = np.concatenate(all_sigmas, axis=0) if all_sigmas else None

    # Metrics
    metrics = compute_metrics(preds, targets, sigmas)
    print("\n--- Test Metrics (normalised space) ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    if metrics["r2_vs_naive"] < 0:
        print("\n  WARNING: R² < 0. Model is worse than predicting zero.")
    elif metrics["r2_vs_naive"] < 0.02:
        print("\n  NOTE: R² is near zero. Normal for high-noise returns.")

    if probabilistic:
        print(f"\n--- Z-Score Signal Summary ---")
        print(f"  mean |z|:  {metrics['mean_abs_z']:.4f}")
        print(f"  |z| > 1:   {metrics['z_gt_1_frac']:.1%}")
        print(f"  |z| > 2:   {metrics['z_gt_2_frac']:.1%}")

    # Save
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({**metrics, "test_stocks": test_stocks, "probabilistic": probabilistic}, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")

    # Plot
    plot_path = os.path.join(results_dir, "predictions.png")
    _plot_predictions(preds, targets, test_stocks, n_plot_stocks, plot_path)

    if probabilistic:
        zscore_plot_path = os.path.join(results_dir, "zscore_distribution.png")
        _plot_zscore_distribution(preds, sigmas, test_stocks, zscore_plot_path)

    return metrics


def _plot_predictions(preds, targets, test_stocks, n_plot_stocks, save_path):
    """Plot predicted vs actual for test stocks."""
    # Each test stock contributes (T - context_len - horizon + 1) samples
    # They are concatenated in order in the dataset
    n_stocks = min(n_plot_stocks, len(test_stocks))
    samples_per_stock = len(preds) // len(test_stocks)

    fig, axes = plt.subplots(n_stocks, 2, figsize=(14, 3 * n_stocks))
    if n_stocks == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_stocks):
        start = i * samples_per_stock
        end = start + samples_per_stock
        p = preds[start:end, 0, 0]
        t = targets[start:end, 0, 0]

        # Time series
        ax = axes[i, 0]
        n_show = min(200, len(p))
        ax.plot(t[:n_show], label="Actual", alpha=0.8, linewidth=0.8)
        ax.plot(p[:n_show], label="Predicted", alpha=0.8, linewidth=0.8, linestyle="--")
        ax.set_title(f"Stock {test_stocks[i]} — Time Series (first {n_show} steps)")
        ax.set_ylabel("Normalised return")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Scatter
        ax = axes[i, 1]
        ax.scatter(t, p, alpha=0.2, s=5, rasterized=True)
        lim = max(np.abs(t).max(), np.abs(p).max()) * 1.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="k", linewidth=0.5)
        ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=0.8, label="Perfect")
        corr = np.corrcoef(t, p)[0, 1] if len(t) > 1 else 0.0
        ax.set_title(f"Stock {test_stocks[i]} — Scatter (corr={corr:.3f})")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Test set inference results", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {save_path}")


def _plot_zscore_distribution(preds, sigmas, test_stocks, save_path):
    """Plot z-score distribution and signal calibration."""
    mu_total = preds.sum(axis=1).squeeze(-1)          # (n_samples,)
    sigma_total = np.sqrt((sigmas ** 2).sum(axis=1)).squeeze(-1)
    z = mu_total / (sigma_total + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Z-score histogram
    ax = axes[0]
    ax.hist(z, bins=100, density=True, alpha=0.7, edgecolor="none")
    ax.axvline(1.0, color="r", linestyle="--", alpha=0.5, label="|z|=1")
    ax.axvline(-1.0, color="r", linestyle="--", alpha=0.5)
    ax.axvline(2.0, color="orange", linestyle="--", alpha=0.5, label="|z|=2")
    ax.axvline(-2.0, color="orange", linestyle="--", alpha=0.5)
    ax.set_title("Z-Score Distribution")
    ax.set_xlabel("z = μ_total / σ_total")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Sigma distribution
    ax = axes[1]
    mean_sigma = sigmas.mean(axis=1).squeeze(-1)
    ax.hist(mean_sigma, bins=100, density=True, alpha=0.7, edgecolor="none")
    ax.set_title("Mean σ Distribution")
    ax.set_xlabel("σ (per horizon step)")
    ax.grid(True, alpha=0.3)

    # |mu_total| vs cost threshold
    ax = axes[2]
    abs_mu = np.abs(mu_total)
    ax.hist(abs_mu, bins=100, density=True, alpha=0.7, edgecolor="none", label="|μ_total|")
    ax.set_title("|μ_total| — Signal Magnitude")
    ax.set_xlabel("|μ_total| (normalised)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Z-score plot saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--test_stocks", type=str, required=True,
                        help="JSON list of test stock indices, e.g. '[9]'")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--n_plot_stocks", type=int, default=5)
    args = parser.parse_args()

    returns = np.load(args.data)
    test_stocks = json.loads(args.test_stocks)
    run_inference(args.checkpoint, returns, test_stocks, args.results_dir, args.n_plot_stocks)
