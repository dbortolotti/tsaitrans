"""
backtest.py

Phase 0: Economic validation of transformer signal.

Tests whether the signal has positive expected value after costs using
simple analytical strategies, before involving RL.

Strategies tested:
    1. Linear:     position = k * signal
    2. Threshold:  position = sign(signal) if |signal| > θ, else 0
    3. Z-score:    position = k * z  if |z| > θ  (probabilistic models only)

Metrics reported: PnL, Sharpe, turnover, PnL per trade, edge per trade.

Standalone usage:
    python backtest.py --checkpoint checkpoints/run1 --data data/returns.npy \
        --test_stocks '[30,31,32]' --half_spread 0.0005

Can also be imported and called from run_experiment.py.
"""

import argparse
import json
import logging
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import FactorTransformer, TimeSeriesDataset, get_device, normalize_horizons
import torch
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


def load_predictions(checkpoint_dir, returns, stock_indices):
    """
    Load transformer and generate predictions for given stocks.

    Returns:
        preds: (n_samples, horizon, 1) — mu predictions
        targets: (n_samples, horizon, 1) — actual returns
        sigmas: (n_samples, horizon, 1) or None — uncertainty (if probabilistic)
        mean, std: normalization stats
    """
    device = get_device()

    ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    mean = float(np.load(os.path.join(checkpoint_dir, "mean.npy")))
    std = float(np.load(os.path.join(checkpoint_dir, "std.npy")))

    context_len = config.get("context_len", 60)
    horizons = normalize_horizons(config)
    probabilistic = config.get("probabilistic", False)

    ds = TimeSeriesDataset(returns, stock_indices, context_len, horizons=horizons, mean=mean, std=std)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    model = FactorTransformer(
        n_stocks=1,
        context_len=context_len,
        horizons=horizons,
        d_model=config.get("d_model", 64),
        n_heads=config.get("n_heads", 4),
        n_layers=config.get("n_layers", 3),
        ffn_dim=config.get("ffn_dim", 256),
        dropout=0.0,
        probabilistic=probabilistic,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds, all_targets, all_sigmas = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if probabilistic:
                mu, log_sigma = model(x)
                all_preds.append(mu.cpu().numpy())
                all_sigmas.append(torch.exp(log_sigma).cpu().numpy())
            else:
                pred = model(x)
                all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    sigmas = np.concatenate(all_sigmas, axis=0) if all_sigmas else None

    return preds, targets, sigmas, mean, std, config


def run_strategy(signal, future_returns, half_spread_norm, strategy, **params):
    """
    Run a simple trading strategy and compute metrics.

    Args:
        signal: (n_samples,) trading signal
        future_returns: (n_samples,) actual next-period return (sum over horizon)
        half_spread_norm: half-spread in normalised units
        strategy: 'linear', 'threshold', or 'zscore'
        params: strategy-specific params (k, theta)

    Returns: dict of metrics
    """
    n = len(signal)
    k = params.get("k", 1.0)
    theta = params.get("theta", 0.0)

    # Compute positions
    if strategy == "linear":
        positions = k * signal
    elif strategy == "threshold":
        positions = np.where(np.abs(signal) > theta, np.sign(signal), 0.0)
    elif strategy == "zscore":
        positions = np.where(np.abs(signal) > theta, k * signal, 0.0)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # PnL: position * future return - transaction cost
    # Cost = half_spread * |change in position| (pay spread on turnover)
    delta_pos = np.diff(positions, prepend=0.0)
    turnover = np.abs(delta_pos)
    costs = half_spread_norm * turnover
    gross_pnl = positions * future_returns
    net_pnl = gross_pnl - costs

    total_pnl = np.sum(net_pnl)
    total_gross = np.sum(gross_pnl)
    total_cost = np.sum(costs)
    total_turnover = np.sum(turnover)
    n_trades = np.sum(turnover > 1e-10)

    # Sharpe (annualised assuming daily)
    pnl_std = np.std(net_pnl) + 1e-10
    sharpe = np.mean(net_pnl) / pnl_std * np.sqrt(n)

    return {
        "total_pnl": float(total_pnl),
        "total_gross_pnl": float(total_gross),
        "total_cost": float(total_cost),
        "mean_pnl_per_step": float(np.mean(net_pnl)),
        "sharpe": float(sharpe),
        "total_turnover": float(total_turnover),
        "n_trades": int(n_trades),
        "pnl_per_trade": float(total_pnl / max(n_trades, 1)),
        "edge_per_trade": float(total_gross / max(n_trades, 1)),
        "mean_position": float(np.mean(np.abs(positions))),
    }


def run_backtest(checkpoint_dir, returns, test_stocks, results_dir, half_spread=0.0005):
    """
    Run Phase 0 analytical backtest.

    Tests linear and threshold strategies across parameter sweeps.
    If the model is probabilistic, also tests z-score strategies.
    """
    logger.info("Loading predictions...")
    preds, targets, sigmas, mean, std, config = load_predictions(
        checkpoint_dir, returns, test_stocks
    )

    horizons = normalize_horizons(config)
    probabilistic = config.get("probabilistic", False)

    # Use longest-horizon cumulative prediction as signal (already cumulative, no sum)
    mu_total = preds[:, -1, 0]          # (n_samples,) — longest horizon prediction
    ret_total = targets[:, -1, 0]       # (n_samples,) — actual return at that horizon

    # Half-spread in normalised units
    half_spread_norm = half_spread / (std + 1e-8)

    logger.info("Samples: %d", len(mu_total))
    logger.info("Horizons: %s (signal from longest: h=%s)", horizons, horizons[-1])
    logger.info("Half-spread (raw): %s", half_spread)
    logger.info("Half-spread (norm): %.6f", half_spread_norm)
    logger.info("Signal std: %.6f", np.std(mu_total))
    logger.info("Signal mean: %.6f", np.mean(mu_total))

    results = {"config": {"half_spread": half_spread, "horizons": horizons,
                           "probabilistic": probabilistic, "test_stocks": test_stocks}}

    # --- Linear strategy: position = k * mu_total ---
    logger.info("--- Linear Strategy: position = k * signal ---")
    k_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    linear_results = []
    for k in k_values:
        r = run_strategy(mu_total, ret_total, half_spread_norm, "linear", k=k)
        linear_results.append({"k": k, **r})
        logger.info(
            "  k=%5.1f | PnL=%8.4f | Sharpe=%7.2f | Edge/trade=%8.6f | Turnover=%8.1f",
            k, r["total_pnl"], r["sharpe"], r["edge_per_trade"], r["total_turnover"],
        )
    results["linear"] = linear_results

    # --- Threshold strategy: position = sign(signal) if |signal| > theta ---
    logger.info("--- Threshold Strategy: position = sign(signal) if |signal| > θ ---")
    signal_std = np.std(mu_total)
    theta_values = [0.0, 0.25 * signal_std, 0.5 * signal_std, signal_std,
                    1.5 * signal_std, 2.0 * signal_std]
    threshold_results = []
    for theta in theta_values:
        r = run_strategy(mu_total, ret_total, half_spread_norm, "threshold", theta=theta)
        threshold_results.append({"theta": float(theta), "theta_sigma": float(theta / signal_std), **r})
        logger.info(
            "  θ=%4.2fσ | PnL=%8.4f | Sharpe=%7.2f | Trades=%5d | PnL/trade=%8.6f",
            theta / signal_std, r["total_pnl"], r["sharpe"], r["n_trades"], r["pnl_per_trade"],
        )
    results["threshold"] = threshold_results

    # --- Z-score strategy (probabilistic only) ---
    if probabilistic and sigmas is not None:
        sigma_long = sigmas[:, -1, 0]  # longest horizon sigma (no aggregation)
        z = mu_total / (sigma_long + 1e-8)

        logger.info("--- Z-Score Signal Stats ---")
        logger.info("  mean |z|: %.4f", np.mean(np.abs(z)))
        logger.info("  |z| > 1: %.1f%%", 100 * np.mean(np.abs(z) > 1.0))
        logger.info("  |z| > 2: %.1f%%", 100 * np.mean(np.abs(z) > 2.0))

        logger.info("--- Z-Score Strategy: position = k * z if |z| > θ ---")
        z_params = [
            (1.0, 0.0), (1.0, 0.5), (1.0, 1.0), (1.0, 1.5), (1.0, 2.0),
            (2.0, 0.5), (2.0, 1.0), (5.0, 1.0), (5.0, 2.0),
        ]
        zscore_results = []
        for k, theta in z_params:
            r = run_strategy(z, ret_total, half_spread_norm, "zscore", k=k, theta=theta)
            zscore_results.append({"k": k, "theta": theta, **r})
            logger.info(
                "  k=%4.1f θ=%4.1f | PnL=%8.4f | Sharpe=%7.2f | Trades=%5d | PnL/trade=%8.6f",
                k, theta, r["total_pnl"], r["sharpe"], r["n_trades"], r["pnl_per_trade"],
            )
        results["zscore"] = zscore_results

    # --- Signal bucketing: E[return | signal_bin] ---
    logger.info("--- Signal Bucketing: E[return | signal_bin] ---")
    n_bins = 10
    bin_edges = np.percentile(mu_total, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_indices = np.digitize(mu_total, bin_edges) - 1

    bucket_results = []
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        mean_signal = np.mean(mu_total[mask])
        mean_return = np.mean(ret_total[mask])
        count = int(mask.sum())
        bucket_results.append({
            "bin": b, "mean_signal": float(mean_signal),
            "mean_return": float(mean_return), "count": count,
            "profitable_vs_cost": float(abs(mean_return) > half_spread_norm),
        })
        marker = " ***" if abs(mean_return) > half_spread_norm else ""
        logger.info(
            "  bin %2d | signal=%8.5f | E[ret]=%8.5f | n=%5d%s",
            b, mean_signal, mean_return, count, marker,
        )
    results["buckets"] = bucket_results

    logger.info("(*** = |E[return]| > half_spread_norm = %.6f)", half_spread_norm)

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "backtest.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved: %s", results_path)

    # Plot
    _plot_backtest(results, results_dir, half_spread_norm)

    return results


def _plot_backtest(results, results_dir, half_spread_norm):
    """Plot backtest summary."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Linear strategy: PnL vs k
    ax = axes[0, 0]
    ks = [r["k"] for r in results["linear"]]
    pnls = [r["total_pnl"] for r in results["linear"]]
    sharpes = [r["sharpe"] for r in results["linear"]]
    ax.bar(range(len(ks)), pnls, tick_label=[str(k) for k in ks], alpha=0.7)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("k")
    ax.set_ylabel("Total PnL (norm)")
    ax.set_title("Linear Strategy: PnL vs k")
    ax.grid(True, alpha=0.3)

    # Threshold strategy: Sharpe vs theta
    ax = axes[0, 1]
    thetas = [r["theta_sigma"] for r in results["threshold"]]
    sharpes_t = [r["sharpe"] for r in results["threshold"]]
    ax.plot(thetas, sharpes_t, "o-")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("θ (in signal σ)")
    ax.set_ylabel("Sharpe")
    ax.set_title("Threshold Strategy: Sharpe vs θ")
    ax.grid(True, alpha=0.3)

    # Signal bucketing
    ax = axes[1, 0]
    if results.get("buckets"):
        signals = [r["mean_signal"] for r in results["buckets"]]
        rets = [r["mean_return"] for r in results["buckets"]]
        ax.bar(range(len(signals)), rets, alpha=0.7)
        ax.axhline(half_spread_norm, color="r", linestyle="--", alpha=0.5, label=f"+cost")
        ax.axhline(-half_spread_norm, color="r", linestyle="--", alpha=0.5, label=f"-cost")
        ax.set_xlabel("Signal bin (low → high)")
        ax.set_ylabel("E[return]")
        ax.set_title("Signal Calibration: E[return | signal bin]")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Z-score strategy (or second linear view)
    ax = axes[1, 1]
    if "zscore" in results:
        # Plot PnL for z-score strategies
        labels = [f"k={r['k']},θ={r['theta']}" for r in results["zscore"]]
        pnls_z = [r["total_pnl"] for r in results["zscore"]]
        ax.barh(range(len(labels)), pnls_z, tick_label=labels, alpha=0.7)
        ax.axvline(0, color="k", linewidth=0.5)
        ax.set_xlabel("Total PnL (norm)")
        ax.set_title("Z-Score Strategy Comparison")
        ax.grid(True, alpha=0.3)
    else:
        # Edge per trade vs k
        edges = [r["edge_per_trade"] for r in results["linear"]]
        ax.bar(range(len(ks)), edges, tick_label=[str(k) for k in ks], alpha=0.7)
        ax.axhline(half_spread_norm, color="r", linestyle="--", label="cost")
        ax.set_xlabel("k")
        ax.set_ylabel("Edge per trade (norm)")
        ax.set_title("Linear Strategy: Edge vs Cost")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Phase 0: Analytical Signal Backtest", fontsize=13)
    plt.tight_layout()
    save_path = os.path.join(results_dir, "backtest.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Backtest plot saved: %s", save_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    parser = argparse.ArgumentParser(description="Phase 0: Analytical signal backtest")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--test_stocks", type=str, required=True,
                        help="JSON list of test stock indices")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--half_spread", type=float, default=0.0005)
    args = parser.parse_args()

    returns = np.load(args.data)
    test_stocks = json.loads(args.test_stocks)
    run_backtest(args.checkpoint, returns, test_stocks, args.results_dir, args.half_spread)
