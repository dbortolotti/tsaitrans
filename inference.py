"""
inference.py

Loads a trained checkpoint, runs on the test set, computes metrics,
and plots predictions vs. ground truth.

Usage:
    python inference.py --checkpoint checkpoints/run1 --data data/returns_snr1.00_seed42.npy

Outputs:
    results/<tag>_metrics.json
    results/<tag>_predictions.png
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for all envs)
import matplotlib.pyplot as plt

from model import FactorTransformer, get_device, make_splits


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """
    pred, target: (n_samples, horizon, n_stocks)
    Returns dict of scalar metrics.
    """
    mse  = np.mean((pred - target) ** 2)
    mae  = np.mean(np.abs(pred - target))
    rmse = np.sqrt(mse)

    # Directional accuracy: did we get the sign right?
    # Only meaningful for horizon=1 or first step
    dir_acc = np.mean(np.sign(pred[:, 0, :]) == np.sign(target[:, 0, :]))

    # Naive baseline: predict next = 0 (mean of normalised returns is ~0)
    # This is the right baseline in normalised space
    naive_mse = np.mean(target ** 2)

    # R² vs naive
    r2 = 1.0 - mse / (naive_mse + 1e-10)

    return {
        "mse":          float(mse),
        "rmse":         float(rmse),
        "mae":          float(mae),
        "directional_accuracy": float(dir_acc),
        "naive_mse":    float(naive_mse),
        "r2_vs_naive":  float(r2),
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(checkpoint_dir: str, data_path: str, n_plot_stocks: int = 5):
    device = get_device()
    print(f"Using device: {device}")

    # --- Load checkpoint ---
    ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.5f})")

    # --- Load normalisation stats ---
    mean = np.load(os.path.join(checkpoint_dir, "mean.npy"))
    std  = np.load(os.path.join(checkpoint_dir, "std.npy"))

    # --- Load data ---
    returns = np.load(data_path)
    print(f"Data shape: {returns.shape}")

    # --- Test dataset ---
    _, _, test_ds, _, _ = make_splits(
        returns,
        context_len=config["context_len"],
        horizon=config["horizon"],
        train_frac=config["train_frac"],
        val_frac=config["val_frac"],
    )
    # Override with saved stats to ensure exact match
    test_ds.mean = mean
    test_ds.std  = std

    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_ds)}")

    # --- Rebuild model ---
    n_stocks = returns.shape[1]
    model = FactorTransformer(
        n_stocks=n_stocks,
        context_len=config["context_len"],
        horizon=config["horizon"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        ffn_dim=config["ffn_dim"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Parameters: {model.count_parameters():,}")

    # --- Run predictions ---
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())

    preds   = np.concatenate(all_preds,   axis=0)  # (n_test, horizon, n_stocks)
    targets = np.concatenate(all_targets, axis=0)

    # --- Metrics ---
    metrics = compute_metrics(preds, targets)
    print("\n--- Test Metrics (normalised space) ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Honest note: R² close to 0 is expected and normal for noisy financial returns.
    # R² < 0 means you're worse than predicting zero — that's a red flag.
    if metrics["r2_vs_naive"] < 0:
        print("\n  WARNING: R² < 0. Model is worse than predicting zero. Check training.")
    elif metrics["r2_vs_naive"] < 0.02:
        print("\n  NOTE: R² is near zero. This is normal for high-noise returns.")

    # --- Save metrics ---
    os.makedirs("results", exist_ok=True)
    tag = config.get("tag", "run")
    metrics_path = os.path.join("results", f"{tag}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({**metrics, "config": config}, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")

    # --- Plot predictions vs ground truth ---
    plot_path = os.path.join("results", f"{tag}_predictions.png")
    _plot_predictions(preds, targets, n_plot_stocks, plot_path, tag)

    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_predictions(
    preds: np.ndarray,     # (n_test, horizon, n_stocks)
    targets: np.ndarray,
    n_stocks_to_plot: int,
    save_path: str,
    tag: str,
):
    """
    Plot predicted vs actual for the first n_stocks_to_plot stocks.
    For horizon=1 this is a scatter + time series overlay.
    """
    n_stocks = min(n_stocks_to_plot, preds.shape[2])
    fig, axes = plt.subplots(n_stocks, 2, figsize=(14, 3 * n_stocks))
    if n_stocks == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_stocks):
        p = preds[:, 0, i]      # first horizon step, stock i
        t = targets[:, 0, i]

        # --- Time series (first 200 test steps) ---
        ax = axes[i, 0]
        n_show = min(200, len(p))
        ax.plot(t[:n_show], label="Actual",    alpha=0.8, linewidth=0.8)
        ax.plot(p[:n_show], label="Predicted", alpha=0.8, linewidth=0.8, linestyle="--")
        ax.set_title(f"Stock {i+1} — Time Series (first {n_show} test steps)")
        ax.set_ylabel("Normalised return")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # --- Scatter: predicted vs actual ---
        ax = axes[i, 1]
        ax.scatter(t, p, alpha=0.2, s=5, rasterized=True)
        lim = max(np.abs(t).max(), np.abs(p).max()) * 1.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="k", linewidth=0.5)
        ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=0.8, label="Perfect")
        corr = np.corrcoef(t, p)[0, 1]
        ax.set_title(f"Stock {i+1} — Scatter (corr={corr:.3f})")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Inference results: {tag}", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",      type=str, required=True,
                        help="Path to checkpoint dir, e.g. checkpoints/run1")
    parser.add_argument("--data",            type=str, required=True,
                        help="Path to returns .npy file used for this run")
    parser.add_argument("--n_plot_stocks",   type=int, default=5,
                        help="How many stocks to plot")
    args = parser.parse_args()

    run_inference(
        checkpoint_dir=args.checkpoint,
        data_path=args.data,
        n_plot_stocks=args.n_plot_stocks,
    )
