"""
train.py

Trains the FactorTransformer on returns data.

Usage:
    python train.py --data data/returns_snr1.00_seed42.npy --tag run1

Saves:
    checkpoints/<tag>/best_model.pt   -- best val loss checkpoint
    checkpoints/<tag>/train_log.csv   -- per-epoch loss history
"""

import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import FactorTransformer, get_device, make_splits


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# ---------------------------------------------------------------------------
# LR Schedule: cosine with linear warmup
# ---------------------------------------------------------------------------

def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """
    Linear warmup for warmup_steps, then cosine decay to 0.
    Returns a LambdaLR scheduler.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: dict):
    seed_everything(config["seed"])
    device = get_device()
    print(f"Using device: {device}")

    # --- Load data ---
    returns = np.load(config["data_path"])  # (n_timesteps, n_stocks)
    print(f"Data shape: {returns.shape}")
    n_timesteps, n_stocks = returns.shape

    # --- Datasets ---
    train_ds, val_ds, test_ds, mean, std = make_splits(
        returns,
        context_len=config["context_len"],
        horizon=config["horizon"],
        train_frac=config["train_frac"],
        val_frac=config["val_frac"],
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=False)

    # --- Model ---
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
    print(f"Parameters: {model.count_parameters():,}")

    # --- Optimiser + schedule ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    total_steps = config["n_epochs"] * len(train_loader)
    warmup_steps = int(0.05 * total_steps)
    scheduler = cosine_with_warmup(optimizer, warmup_steps, total_steps)

    criterion = nn.MSELoss()

    # --- Checkpoint dir ---
    ckpt_dir = os.path.join("checkpoints", config["tag"])
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save config
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save normalisation stats (needed at inference time)
    np.save(os.path.join(ckpt_dir, "mean.npy"), mean)
    np.save(os.path.join(ckpt_dir, "std.npy"),  std)

    # --- Training ---
    log_path = os.path.join(ckpt_dir, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "elapsed_s"])

    best_val_loss = float("inf")
    patience_counter = 0
    t_start = time.time()

    for epoch in range(1, config["n_epochs"] + 1):

        # ---- Train ----
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()
        val_loss /= len(val_loader)

        elapsed = time.time() - t_start
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:03d}/{config['n_epochs']} | "
            f"train={train_loss:.5f} | val={val_loss:.5f} | "
            f"lr={current_lr:.2e} | {elapsed:.0f}s"
        )

        # Log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, current_lr, elapsed])

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "config": config,
            }, os.path.join(ckpt_dir, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"Early stopping at epoch {epoch} (patience={config['patience']})")
                break

    print(f"\nBest val loss: {best_val_loss:.5f}")
    print(f"Checkpoint: checkpoints/{config['tag']}/best_model.pt")
    return best_val_loss


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",           type=str,   required=True, help="Path to returns .npy file")
    parser.add_argument("--tag",            type=str,   default=None)
    parser.add_argument("--context_len",    type=int,   default=60)
    parser.add_argument("--horizon",        type=int,   default=1)
    parser.add_argument("--d_model",        type=int,   default=64)
    parser.add_argument("--n_heads",        type=int,   default=4)
    parser.add_argument("--n_layers",       type=int,   default=3)
    parser.add_argument("--ffn_dim",        type=int,   default=256)
    parser.add_argument("--dropout",        type=float, default=0.1)
    parser.add_argument("--batch_size",     type=int,   default=128)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--n_epochs",       type=int,   default=50)
    parser.add_argument("--patience",       type=int,   default=10)
    parser.add_argument("--train_frac",     type=float, default=0.70)
    parser.add_argument("--val_frac",       type=float, default=0.15)
    parser.add_argument("--seed",           type=int,   default=42)
    args = parser.parse_args()

    config = vars(args)
    config["data_path"] = config.pop("data")
    config["tag"] = config["tag"] or os.path.splitext(os.path.basename(config["data_path"]))[0]

    train(config)
