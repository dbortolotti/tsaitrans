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
import logging
import os
import shutil
import sys

import numpy as np


logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


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
        "horizons": [1, 2, 4, 8, 16],
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
        "probabilistic": True,
    },
    "rl": {
        "predictor": "transformer",
        "reward_mode": "signal_exposure",
        "n_envs": 4,
        "n_iterations": 500,
        "rollout_steps": 2048,
        "actor_lr": 3e-4,
        "critic_lr": 1e-3,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 5e-4,
        "ent_coef_end": 1e-5,
        "ent_coef_anneal": True,
        "target_kl": 0.02,
        "normalize_rewards": True,
        "log_std_init": -0.5,
        "log_std_min": -3.0,
        "log_std_max": 1.0,
        # Phase 1 experiment knobs
        "ablate_penalties": False,
        "disable_reward_norm": False,
        "action_space": "continuous",
        "use_uncertainty": False,
        "reward_horizon": 8,
        # signal_exposure params
        "target_horizon": 16,
        "reward_scale": 10.0,
        "alpha_pos": 0.01,
        "beta_trade": 0.005,
        "max_position": 1.0,
        "baseline_k": 1.0,
        # full_market_making params
        "half_spread": 0.0005,
        "target_vol": 0.02,
        "n_sigma": 2.0,
        "tau": 20,
        "lambda2": 1.5,
        "alignment_coef": 0.05,
        "alignment_clip": 3.0,
        "trade_penalty": 0.0,
        "max_width": 3.0,
        "max_skew": 3.0,
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


def main(config_path: str, skip_data: bool = False, skip_transformer: bool = False,
         skip_backtest: bool = False, skip_rl: bool = False):
    # --- Load and merge config ---
    with open(config_path) as f:
        user_config = json.load(f)
    config = merge_config(DEFAULTS, user_config)
    user_data_overrides = user_config.get("data", {})

    name = os.path.splitext(os.path.basename(config_path))[0]
    output_dir = os.path.join("output", name)

    if os.path.exists(output_dir):
        if sys.stdin.isatty():
            ans = input(f"Output folder '{output_dir}' already exists. Proceed and overwrite? [Y/n] ").strip().lower()
            if ans == "n":
                logger.info("Aborting.")
                sys.exit(0)
        else:
            logger.info("Output folder '%s' already exists. Proceeding (non-interactive).", output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Copy config into output dir
    shutil.copy(config_path, os.path.join(output_dir, f"{name}.json"))

    # Save resolved config (with defaults filled in)
    with open(os.path.join(output_dir, "resolved_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logger.info("Experiment: %s", name)
    logger.info("Output dir: %s", output_dir)

    # --- Resolve base_experiment (reuse data + transformer from another run) ---
    base_experiment = config.get("base_experiment", None)
    if base_experiment:
        base_dir = os.path.join("output", base_experiment)
        if not os.path.isdir(base_dir):
            logger.error("base_experiment '%s' not found at %s", base_experiment, base_dir)
            sys.exit(1)
        logger.info("Reusing data + transformer from base experiment: %s", base_experiment)

        # Load the base experiment's resolved config to get its data section for stock splits
        base_resolved = os.path.join(base_dir, "resolved_config.json")
        if os.path.exists(base_resolved):
            with open(base_resolved) as f:
                base_config = json.load(f)
            # Reuse the base experiment's DGP by default, but preserve any
            # explicit data overrides from the current config such as smaller
            # RL train/val stock counts for sweeps.
            config["data"] = merge_config(base_config["data"], user_data_overrides)

    # --- Compute stock split ---
    data_cfg = config["data"]
    stock_split = compute_stock_split(data_cfg)
    n_stocks = sum(len(v) for v in stock_split.values())

    logger.info("Stock split (%d total):", n_stocks)
    for group, indices in stock_split.items():
        logger.info("  %s: %s", group, indices)

    # Add prediction/ and placing/ to path for imports
    repo_root = os.path.dirname(os.path.abspath(__file__))
    pred_dir = os.path.join(repo_root, "prediction")
    placing_dir = os.path.join(repo_root, "placing")
    for d in [pred_dir, placing_dir]:
        if d not in sys.path:
            sys.path.insert(0, d)

    if base_experiment:
        # Point data and checkpoint dirs at the base experiment
        data_dir = os.path.join(base_dir, "data")
        checkpoint_dir = os.path.join(base_dir, "checkpoints")
        # Results and RL output go to this experiment's own dir
        results_dir = os.path.join(output_dir, "results")
        rl_dir = os.path.join(output_dir, "checkpoints_rl")

        # Find the data file (named after the base experiment)
        data_files = [f for f in os.listdir(data_dir) if f.startswith("returns_") and f.endswith(".npy")]
        if not data_files:
            logger.error("No returns file found in %s", data_dir)
            sys.exit(1)
        data_file = os.path.join(data_dir, data_files[0])
        returns = np.load(data_file).astype(np.float32)
        logger.info("Loaded data from %s - shape %s", data_file, returns.shape)

        checkpoint_file = os.path.join(checkpoint_dir, "best_model.pt")
        if not os.path.exists(checkpoint_file):
            logger.error("No transformer checkpoint found at %s", checkpoint_file)
            sys.exit(1)
        logger.info("Using transformer checkpoint from %s", checkpoint_dir)

        # Skip stages 1–4
        skip_data = True
        skip_transformer = True
        skip_backtest = True
    else:
        data_dir = os.path.join(output_dir, "data")
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        results_dir = os.path.join(output_dir, "results")
        rl_dir = os.path.join(output_dir, "checkpoints_rl")

        # === Stage 1: Generate data ===
        data_file = os.path.join(data_dir, f"returns_{name}.npy")
        if skip_data and os.path.exists(data_file):
            logger.info("Stage 1 skipped: loading existing data")
            returns = np.load(data_file).astype(np.float32)
        else:
            if skip_data:
                ans = input("\n[WARN] --skip-data: no existing data found. Generate now? [Y/n] ").strip().lower()
                if ans == "n":
                    logger.info("Aborting.")
                    sys.exit(1)
            logger.info("%s", "=" * 60)
            logger.info("STAGE 1: Generate data")
            logger.info("%s", "=" * 60)

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

            save(result, data_dir, name)
            returns = result["returns"].astype(np.float32)

        # === Stage 2: Train transformer ===
        checkpoint_file = os.path.join(checkpoint_dir, "best_model.pt")
        if skip_transformer and os.path.exists(checkpoint_file):
            logger.info("Stage 2 skipped: reusing existing transformer checkpoint")
        else:
            if skip_transformer:
                ans = input("[WARN] --skip-transformer: no checkpoint found. Train now? [Y/n] ").strip().lower()
                if ans == "n":
                    logger.info("Aborting.")
                    sys.exit(1)
            logger.info("%s", "=" * 60)
            logger.info("STAGE 2: Train transformer")
            logger.info("%s", "=" * 60)

            from train import train as train_transformer

            transformer_cfg = config["transformer"]
            train_transformer(transformer_cfg, returns, stock_split, checkpoint_dir)

        # === Stage 3: Transformer inference on test stocks ===
        if skip_transformer and os.path.exists(checkpoint_file):
            logger.info("Stage 3 skipped: skipping inference (transformer was skipped)")
        else:
            logger.info("%s", "=" * 60)
            logger.info("STAGE 3: Transformer inference (test stocks)")
            logger.info("%s", "=" * 60)

            from inference import run_inference

            run_inference(checkpoint_dir, returns, stock_split["test"], results_dir)

        # === Stage 4: Phase 0 analytical backtest ===
        backtest_dir = os.path.join(output_dir, "backtest")
        if skip_backtest or skip_transformer:
            logger.info("Stage 4 skipped: backtest skipped")
        else:
            logger.info("%s", "=" * 60)
            logger.info("STAGE 4: Phase 0 analytical backtest")
            logger.info("%s", "=" * 60)

            from backtest import run_backtest

            rl_half_spread = config["rl"].get("half_spread", 0.0005)
            # Test on RL train stocks (same stocks RL will use)
            run_backtest(checkpoint_dir, returns, stock_split["rl_train"],
                         backtest_dir, half_spread=rl_half_spread)

    # === Stage 5: Train RL ===
    if skip_rl:
        logger.info("Stage 5 skipped: RL training skipped")
    else:
        logger.info("%s", "=" * 60)
        logger.info("STAGE 5: Train RL agent")
        logger.info("%s", "=" * 60)

        from train_rl import train as train_rl

        rl_cfg = config["rl"]
        train_rl(rl_cfg, returns, stock_split, rl_dir, transformer_checkpoint=checkpoint_dir)

    # === Done ===
    logger.info("%s", "=" * 60)
    logger.info("Experiment '%s' complete.", name)
    logger.info("All outputs in: %s/", output_dir)
    logger.info("%s", "=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a full experiment pipeline.")
    parser.add_argument("config", help="Path to experiment JSON config")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip stage 1 (data generation) and reuse existing data")
    parser.add_argument("--skip-transformer", action="store_true",
                        help="Skip stages 2-3 (transformer training + inference) and reuse existing checkpoint")
    parser.add_argument("--skip-backtest", action="store_true",
                        help="Skip stage 4 (analytical backtest)")
    parser.add_argument("--skip-rl", action="store_true",
                        help="Skip stage 5 (RL training)")
    args = parser.parse_args()

    configure_logging()
    main(args.config, skip_data=args.skip_data, skip_transformer=args.skip_transformer,
         skip_backtest=args.skip_backtest, skip_rl=args.skip_rl)
