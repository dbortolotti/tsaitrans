"""
run_experiment.py

Runs the full pipeline: generate data → train transformer → inference → train RL.

Usage:
    python run_experiment.py path/to/experiment.json

All outputs go to output/<experiment_name>/:
    data/           — generated returns + metadata
    checkpoints/    — transformer checkpoint
    results/        — transformer inference metrics + plots
    checkpoints_rl/ — RL policy checkpoint
    <name>.json     — copy of the experiment config
"""

import json
import os
import shutil
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Defaults — anything not specified in the experiment JSON uses these
# ---------------------------------------------------------------------------

DEFAULTS = {
    "data": {
        "target_vol": 0.02,
        "snr": 0.3,
        "factor_half_life": 0.1,
        "noise_half_life_range": [0.005, 0.025],
        "n_factors": 3,
        "steps_per_day": 2000,
        "n_steps": 2000,
        "seed": 42,
        "stocks_transformer_train": 20,
        "stocks_transformer_val": 5,
        "stocks_rl_train": 10,
        "stocks_rl_val": 3,
        "stocks_test": 5,
    },
    "transformer": {
        "context_len": 200,
        "horizon": 1,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 3,
        "ffn_dim": 256,
        "dropout": 0.1,
        "batch_size": 128,
        "lr": 3e-4,
        "n_epochs": 100,
        "patience": 15,
        "seed": 42,
    },
    "rl": {
        "predictor": "transformer",
        "n_envs": 32,
        "n_iterations": 200,
        "rollout_steps": 512,
        "lr": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "lambda_inv": 0.01,
        "kappa_spread": 0.0005,
        "max_position": 10,
    },
}


def merge_config(defaults: dict, overrides: dict) -> dict:
    """Deep merge: overrides win, defaults fill gaps."""
    result = {}
    for key in set(list(defaults.keys()) + list(overrides.keys())):
        if key in overrides and key in defaults:
            if isinstance(defaults[key], dict) and isinstance(overrides[key], dict):
                result[key] = merge_config(defaults[key], overrides[key])
            else:
                result[key] = overrides[key]
        elif key in overrides:
            result[key] = overrides[key]
        else:
            result[key] = defaults[key]
    return result


def compute_stock_split(data_config: dict) -> dict:
    """Compute stock index ranges from the split sizes in data config."""
    n_tt = data_config["stocks_transformer_train"]
    n_tv = data_config["stocks_transformer_val"]
    n_rt = data_config["stocks_rl_train"]
    n_rv = data_config["stocks_rl_val"]
    n_te = data_config["stocks_test"]

    idx = 0
    split = {}
    for name, count in [
        ("transformer_train", n_tt),
        ("transformer_val", n_tv),
        ("rl_train", n_rt),
        ("rl_val", n_rv),
        ("test", n_te),
    ]:
        split[name] = list(range(idx, idx + count))
        idx += count

    return split


def main(config_path: str):
    # --- Load and merge config ---
    with open(config_path) as f:
        user_config = json.load(f)
    config = merge_config(DEFAULTS, user_config)

    name = os.path.splitext(os.path.basename(config_path))[0]
    output_dir = os.path.join("output", name)
    os.makedirs(output_dir, exist_ok=True)

    # Copy config into output dir
    shutil.copy(config_path, os.path.join(output_dir, f"{name}.json"))

    # Save resolved config (with defaults filled in)
    with open(os.path.join(output_dir, "resolved_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Experiment: {name}")
    print(f"Output dir: {output_dir}")

    # --- Compute stock split ---
    data_cfg = config["data"]
    stock_split = compute_stock_split(data_cfg)
    n_stocks = sum(len(v) for v in stock_split.values())

    print(f"\nStock split ({n_stocks} total):")
    for group, indices in stock_split.items():
        print(f"  {group}: {indices}")

    # === Stage 1: Generate data ===
    print("\n" + "=" * 60)
    print("STAGE 1: Generate data")
    print("=" * 60)

    # Add prediction/ to path for imports
    repo_root = os.path.dirname(os.path.abspath(__file__))
    pred_dir = os.path.join(repo_root, "prediction")
    placing_dir = os.path.join(repo_root, "placing")
    for d in [pred_dir, placing_dir]:
        if d not in sys.path:
            sys.path.insert(0, d)

    from generate_data import generate, save

    n_steps = data_cfg["n_steps"]
    result = generate(
        n_stocks=n_stocks,
        n_timesteps=n_steps,
        n_factors=data_cfg["n_factors"],
        factor_half_life=data_cfg["factor_half_life"],
        noise_half_life_range=tuple(data_cfg["noise_half_life_range"]),
        target_vol=data_cfg["target_vol"],
        snr=data_cfg["snr"],
        steps_per_day=data_cfg["steps_per_day"],
        seed=data_cfg["seed"],
    )

    data_dir = os.path.join(output_dir, "data")
    save(result, data_dir, name)
    returns = result["returns"].astype(np.float32)

    # === Stage 2: Train transformer ===
    print("\n" + "=" * 60)
    print("STAGE 2: Train transformer")
    print("=" * 60)

    from train import train as train_transformer

    transformer_cfg = config["transformer"]
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    train_transformer(transformer_cfg, returns, stock_split, checkpoint_dir)

    # === Stage 3: Transformer inference on test stocks ===
    print("\n" + "=" * 60)
    print("STAGE 3: Transformer inference (test stocks)")
    print("=" * 60)

    from inference import run_inference

    results_dir = os.path.join(output_dir, "results")
    run_inference(checkpoint_dir, returns, stock_split["test"], results_dir)

    # === Stage 4: Train RL ===
    print("\n" + "=" * 60)
    print("STAGE 4: Train RL agent")
    print("=" * 60)

    from train_rl import train as train_rl

    rl_cfg = config["rl"]
    rl_dir = os.path.join(output_dir, "checkpoints_rl")
    train_rl(rl_cfg, returns, stock_split, rl_dir, transformer_checkpoint=checkpoint_dir)

    # === Done ===
    print("\n" + "=" * 60)
    print(f"Experiment '{name}' complete.")
    print(f"All outputs in: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_experiment.py <experiment.json>")
        sys.exit(1)
    main(sys.argv[1])
