"""
grid_search.py

Config-driven RL grid search for the current market-making setup.

The grid search works by generating experiment JSON files that reuse a base
experiment's data and transformer checkpoint via `base_experiment`, then
optionally running them sequentially through `run_experiment.py`.

Usage:
    python grid_search.py --config experiments/mm_reward_grid.json --generate-only
    python -u grid_search.py --config experiments/mm_reward_grid.json 2>&1 | tee log_mm_reward_grid.txt
"""

import argparse
import itertools
import json
import logging
import os
import subprocess
import sys
from copy import deepcopy


logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def slugify_value(value):
    if isinstance(value, float):
        text = f"{value:.8g}"
    else:
        text = str(value)
    return text.replace("-", "m").replace(".", "p")


def deep_merge(base: dict, updates: dict) -> dict:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_train_summary(run_dir: str) -> dict:
    log_path = os.path.join(run_dir, "checkpoints_rl", "train_log.json")
    if not os.path.exists(log_path):
        return {"status": "missing_train_log"}

    rows = load_json(log_path)
    if not rows:
        return {"status": "empty_train_log"}

    best_cum_pnl = max(row.get("mean_cumulative_pnl", float("-inf")) for row in rows)
    best_sharpe = max(row.get("sharpe", float("-inf")) for row in rows)
    best_corr = max(row.get("action_signal_corr", float("-inf")) for row in rows)
    last = rows[-1]
    tail = rows[-min(20, len(rows)) :]

    return {
        "status": "ok",
        "n_iters": len(rows),
        "last_mean_reward": last.get("mean_reward"),
        "last_cum_pnl": last.get("mean_cumulative_pnl"),
        "last_sharpe": last.get("sharpe"),
        "last_corr": last.get("action_signal_corr"),
        "last_abs_pos": last.get("mean_abs_position"),
        "best_cum_pnl": best_cum_pnl,
        "best_sharpe": best_sharpe,
        "best_corr": best_corr,
        "tail_mean_cum_pnl": sum(r.get("mean_cumulative_pnl", 0.0) for r in tail) / len(tail),
        "tail_mean_sharpe": sum(r.get("sharpe", 0.0) for r in tail) / len(tail),
        "tail_mean_corr": sum(r.get("action_signal_corr", 0.0) for r in tail) / len(tail),
        "tail_mean_abs_pos": sum(r.get("mean_abs_position", 0.0) for r in tail) / len(tail),
    }


def build_experiment_config(base_config: dict, base_experiment: str, rl_updates: dict) -> dict:
    config = deepcopy(base_config)
    config["base_experiment"] = base_experiment
    config["rl"] = deep_merge(config.get("rl", {}), rl_updates)
    return config


def main(config_path: str, generate_only: bool = False):
    spec = load_json(config_path)

    base_experiment = spec["base_experiment"]
    output_prefix = spec["output_prefix"]
    base_config_path = spec.get("base_config", f"experiments/{base_experiment}.json")
    generated_dir = spec.get("generated_dir", os.path.join("experiments", "generated", output_prefix))
    grid = spec["grid"]
    fixed_rl = spec.get("fixed_rl", {})
    fixed_top_level = spec.get("fixed_top_level", {})
    run_args = spec.get("run_args", ["--skip-backtest"])

    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    base_config = load_json(base_config_path)
    os.makedirs(generated_dir, exist_ok=True)

    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[key] for key in keys)))

    logger.info("Grid search spec: %s", config_path)
    logger.info("Base experiment: %s", base_experiment)
    logger.info("Base config: %s", base_config_path)
    logger.info("Output prefix: %s", output_prefix)
    logger.info("Generated configs: %s", generated_dir)
    logger.info("Grid keys: %s", keys)
    logger.info("Combinations: %d", len(combos))

    summary_rows = []

    for idx, values in enumerate(combos, start=1):
        rl_updates = deepcopy(fixed_rl)
        label_parts = []
        for key, value in zip(keys, values):
            rl_updates[key] = value
            label_parts.append(f"{key}{slugify_value(value)}")

        run_name = f"{output_prefix}__" + "__".join(label_parts)
        config_data = build_experiment_config(base_config, base_experiment, rl_updates)
        config_data = deep_merge(config_data, fixed_top_level)

        config_file = os.path.join(generated_dir, f"{run_name}.json")
        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)

        row = {
            "run_name": run_name,
            "config_path": config_file,
            **{key: value for key, value in zip(keys, values)},
        }

        logger.info("[%d/%d] Prepared %s", idx, len(combos), run_name)

        if generate_only:
            row["status"] = "generated_only"
            summary_rows.append(row)
            continue

        cmd = [sys.executable, "-u", "run_experiment.py", config_file, *run_args]
        logger.info("Running: %s", " ".join(cmd))
        run_dir = os.path.join("output", run_name)
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "grid_run.log")
        row["log_path"] = log_path

        with open(log_path, "w") as log_file:
            completed = subprocess.run(
                cmd,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        row["exit_code"] = int(completed.returncode)

        row.update(read_train_summary(run_dir))
        summary_rows.append(row)

        if completed.returncode != 0:
            logger.warning("Run failed: %s", run_name)
        else:
            logger.info(
                "Finished %s | tail_cum_pnl=%.4f tail_sharpe=%.2f tail_corr=%.4f tail_|pos|=%.4f",
                run_name,
                row.get("tail_mean_cum_pnl", float("nan")),
                row.get("tail_mean_sharpe", float("nan")),
                row.get("tail_mean_corr", float("nan")),
                row.get("tail_mean_abs_pos", float("nan")),
            )

    summary_path = os.path.join(generated_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)

    logger.info("Saved summary to %s", summary_path)

    successful = [row for row in summary_rows if row.get("status") == "ok"]
    if successful:
        ranked = sorted(
            successful,
            key=lambda row: (
                row.get("tail_mean_cum_pnl", float("-inf")),
                row.get("tail_mean_sharpe", float("-inf")),
                row.get("tail_mean_corr", float("-inf")),
            ),
            reverse=True,
        )
        best = ranked[0]
        logger.info(
            "Best run so far: %s | tail_cum_pnl=%.4f tail_sharpe=%.2f tail_corr=%.4f tail_|pos|=%.4f",
            best["run_name"],
            best.get("tail_mean_cum_pnl", float("nan")),
            best.get("tail_mean_sharpe", float("nan")),
            best.get("tail_mean_corr", float("nan")),
            best.get("tail_mean_abs_pos", float("nan")),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config-driven RL grid search")
    parser.add_argument(
        "--config",
        required=True,
        help="Grid-search spec JSON, e.g. experiments/mm_reward_grid.json",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only create the experiment JSON files, do not run them",
    )
    args = parser.parse_args()

    configure_logging()
    main(args.config, generate_only=args.generate_only)
