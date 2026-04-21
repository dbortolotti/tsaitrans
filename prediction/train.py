"""
train.py

Trains the FactorTransformer on returns data using stock-based splits.

Each stock is an independent realization — splits are across stocks, not time.
The model is univariate (processes one stock at a time).

Can be called standalone or imported by run_experiment.py.

Standalone usage:
    python train.py --data data/returns.npy --split '{"transformer_train":[0,1,2],"transformer_val":[3,4],"test":[9]}'

Saves:
    <save_dir>/best_model.pt
    <save_dir>/config.json
    <save_dir>/mean.npy, std.npy
    <save_dir>/train_log.csv
"""

import argparse
import csv
import json
import logging
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import FactorTransformer, get_device, make_splits, normalize_horizons


logger = logging.getLogger(__name__)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay to 0."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(config: dict, returns: np.ndarray, stock_split: dict, save_dir: str):
    """
    Train the transformer.

    Args:
        config: transformer hyperparams (d_model, n_heads, lr, etc.)
        returns: (T, N_total) full returns array
        stock_split: dict with 'transformer_train', 'transformer_val', 'test' keys
        save_dir: where to save checkpoints
    """
    seed_everything(config.get("seed", 42))
    device = get_device()
    logger.info("Using device: %s", device)
    logger.info("Data shape: %s", returns.shape)

    context_len = config.get("context_len", 60)
    horizons = normalize_horizons(config)

    # Stock-based splits
    train_ds, val_ds, test_ds, mean, std = make_splits(
        returns, stock_split, context_len=context_len, horizons=horizons
    )
    logger.info("Train: %d | Val: %d | Test: %d samples", len(train_ds), len(val_ds), len(test_ds))
    logger.info("Horizons: %s (cumulative return targets)", horizons)
    logger.info("Train stocks: %s", stock_split["transformer_train"])
    logger.info("Val stocks:   %s", stock_split["transformer_val"])

    batch_size = config.get("batch_size", 128)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )

    # Model — univariate (n_stocks=1)
    probabilistic = config.get("probabilistic", False)
    model = FactorTransformer(
        n_stocks=1,
        context_len=context_len,
        horizons=horizons,
        d_model=config.get("d_model", 64),
        n_heads=config.get("n_heads", 4),
        n_layers=config.get("n_layers", 3),
        ffn_dim=config.get("ffn_dim", 256),
        dropout=config.get("dropout", 0.1),
        probabilistic=probabilistic,
    ).to(device)
    logger.info("Parameters: %s", f"{model.count_parameters():,}")
    if probabilistic:
        logger.info("Mode: probabilistic (heteroscedastic NLL loss)")

    n_epochs = config.get("n_epochs", 50)
    lr = config.get("lr", 3e-4)
    patience = config.get("patience", 10)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = n_epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)
    scheduler = cosine_with_warmup(optimizer, warmup_steps, total_steps)

    if not probabilistic:
        criterion = nn.MSELoss()

    os.makedirs(save_dir, exist_ok=True)

    # Save config
    full_config = {**config, "n_stocks": 1, "probabilistic": probabilistic,
                   "horizons": horizons, "stock_split": stock_split}
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(full_config, f, indent=2)

    # Save normalization stats (scalars)
    np.save(os.path.join(save_dir, "mean.npy"), np.array(mean))
    np.save(os.path.join(save_dir, "std.npy"), np.array(std))

    # Training
    log_path = os.path.join(save_dir, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "elapsed_s"])

    best_val_loss = float("inf")
    patience_counter = 0
    t_start = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if probabilistic:
                mu, log_sigma = model(x)
                # Heteroscedastic NLL: (y - mu)^2 / sigma^2 + 2*log_sigma
                loss = ((y - mu) ** 2 / torch.exp(2 * log_sigma) + 2 * log_sigma).mean()
            else:
                pred = model(x)
                loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                if probabilistic:
                    mu, log_sigma = model(x)
                    loss = ((y - mu) ** 2 / torch.exp(2 * log_sigma) + 2 * log_sigma).mean()
                    val_loss += loss.item()
                else:
                    pred = model(x)
                    val_loss += criterion(pred, y).item()
        val_loss /= len(val_loader)

        elapsed = time.time() - t_start
        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            "Epoch %03d/%d | train=%.5f | val=%.5f | lr=%.2e | %.0fs",
            epoch, n_epochs, train_loss, val_loss, current_lr, elapsed,
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, current_lr, elapsed])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "config": full_config,
                },
                os.path.join(save_dir, "best_model.pt"),
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                break

    logger.info("Best val loss: %.5f", best_val_loss)
    logger.info("Checkpoint: %s/best_model.pt", save_dir)
    return best_val_loss


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to returns .npy")
    parser.add_argument("--split", type=str, required=True,
                        help='JSON string with stock split, e.g. \'{"transformer_train":[0,1,2],"transformer_val":[3,4],"test":[9]}\'')
    parser.add_argument("--save_dir", type=str, default="checkpoints/run1")
    parser.add_argument("--context_len", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=1,
                        help="(legacy) Dense horizon count. Use --horizons instead.")
    parser.add_argument("--horizons", type=str, default=None,
                        help="JSON list of sparse horizons, e.g. '[1,2,4,8,16]'")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--ffn_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probabilistic", action="store_true",
                        help="Enable probabilistic mode (heteroscedastic NLL)")
    args = parser.parse_args()

    returns = np.load(args.data)
    stock_split = json.loads(args.split)
    config = {k: v for k, v in vars(args).items() if k not in ("data", "split", "save_dir")}
    # Parse --horizons JSON if provided
    if config.get("horizons") is not None:
        config["horizons"] = json.loads(config["horizons"])

    train(config, returns, stock_split, args.save_dir)
