"""
grid_search.py

Grid search over RL parameters using an existing experiment's data and transformer.

Usage:
    python grid_search.py -i output/experiment2 -o grid3
    python grid_search.py -i output/experiment2 -o grid3 --config experiments/grid.json

Creates output/<name>_ns{n_sigma}_l2{lambda2}/ for each combination,
symlinks data + transformer checkpoints, runs RL training + 10 simulations.
"""

import json
import os
import shutil
import sys
import itertools
import time


# --- Grid ---
N_SIGMA_VALUES = [0.5, 1.0, 2.0, 3.0]
LAMBDA2_VALUES = [1.0, 3.0, 5.0]

N_SIMS = 10


def main(base_experiment: str, output_name: str, config_path: str):
    # Load config from JSON file
    if not os.path.exists(config_path):
        print(f"[ERROR] {config_path} not found")
        sys.exit(1)

    with open(config_path) as f:
        base_config = json.load(f)

    # Check required dirs exist
    data_dir = os.path.join(base_experiment, "data")
    checkpoint_dir = os.path.join(base_experiment, "checkpoints")
    results_dir = os.path.join(base_experiment, "results")
    for d in [data_dir, checkpoint_dir, results_dir]:
        if not os.path.exists(d):
            print(f"[ERROR] {d} not found")
            sys.exit(1)

    combos = list(itertools.product(N_SIGMA_VALUES, LAMBDA2_VALUES))
    print(f"Grid search: {len(combos)} combinations x {N_SIMS} sims each")
    print(f"  n_sigma:  {N_SIGMA_VALUES}")
    print(f"  lambda2:  {LAMBDA2_VALUES}")
    print(f"  Base:     {base_experiment}")
    print(f"  Output:   output/{output_name}_*/")
    print("=" * 60)

    # Add module dirs to path
    repo_root = os.path.dirname(os.path.abspath(__file__))
    for d in ["prediction", "placing"]:
        p = os.path.join(repo_root, d)
        if p not in sys.path:
            sys.path.insert(0, p)

    from train_rl import train as train_rl
    from simulate import main as simulate

    import numpy as np

    # Load returns once
    data_files = [f for f in os.listdir(data_dir) if f.startswith("returns_") and f.endswith(".npy")]
    if not data_files:
        print(f"[ERROR] No returns file found in {data_dir}")
        sys.exit(1)
    returns = np.load(os.path.join(data_dir, data_files[0]))

    # Compute stock split
    data_cfg = base_config["data"]
    idx = 0
    stock_split = {}
    for name, key in [
        ("transformer_train", "stocks_transformer_train"),
        ("transformer_val", "stocks_transformer_val"),
        ("rl_train", "stocks_rl_train"),
        ("rl_val", "stocks_rl_val"),
        ("test", "stocks_test"),
    ]:
        count = data_cfg[key]
        stock_split[name] = list(range(idx, idx + count))
        idx += count

    t_total = time.time()

    for i, (n_sigma, lambda2) in enumerate(combos):
        run_name = f"{output_name}_ns{n_sigma}_l2{lambda2}"
        run_dir = os.path.join("output", run_name)

        print(f"\n{'=' * 60}")
        print(f"[{i+1}/{len(combos)}] n_sigma={n_sigma}, lambda2={lambda2}")
        print(f"  Output: {run_dir}")
        print("=" * 60)

        os.makedirs(run_dir, exist_ok=True)

        # Symlink data and checkpoints (avoid copying large files)
        for subdir in ["data", "checkpoints", "results"]:
            link = os.path.join(run_dir, subdir)
            target = os.path.abspath(os.path.join(base_experiment, subdir))
            if os.path.exists(link):
                if os.path.islink(link):
                    os.remove(link)
                else:
                    shutil.rmtree(link)
            os.symlink(target, link)

        # Build RL config
        rl_cfg = {
            "predictor": base_config.get("rl", {}).get("predictor", "transformer"),
            "n_envs": base_config.get("rl", {}).get("n_envs", 32),
            "n_iterations": base_config.get("rl", {}).get("n_iterations", 200),
            "rollout_steps": base_config.get("rl", {}).get("rollout_steps", 512),
            "lr": base_config.get("rl", {}).get("lr", 3e-4),
            "gamma": base_config.get("rl", {}).get("gamma", 0.99),
            "gae_lambda": base_config.get("rl", {}).get("gae_lambda", 0.95),
            "half_spread": 0.0005,
            "target_vol": data_cfg.get("target_vol", 0.02),
            "tau": 20,
            "max_width": 3.0,
            "max_skew": 3.0,
            "n_sigma": n_sigma,
            "lambda2": lambda2,
        }

        # Save config for this run
        run_config = dict(base_config)
        run_config["rl"] = rl_cfg
        with open(os.path.join(run_dir, "resolved_config.json"), "w") as f:
            json.dump(run_config, f, indent=2)

        # Train RL
        rl_dir = os.path.join(run_dir, "checkpoints_rl")
        t_start = time.time()
        try:
            train_rl(
                rl_cfg,
                returns,
                stock_split,
                rl_dir,
                transformer_checkpoint=os.path.join(run_dir, "checkpoints"),
            )
        except Exception as e:
            print(f"[ERROR] RL training failed: {e}")
            continue

        # Run N_SIMS simulations with different seeds
        sim_results = []
        base_seed = data_cfg.get("seed", 42) + 1000
        for s in range(N_SIMS):
            seed = base_seed + s
            try:
                result = simulate(run_dir, seed=seed, deterministic=True)
                sim_results.append(result["summary"])
                print(f"  sim {s+1}/{N_SIMS}: seed={seed}  PnL={result['summary']['total_pnl']:.4f}  "
                      f"Fills={result['summary']['n_total_fills']}  Avg|Pos|={result['summary']['avg_abs_position']:.2f}")
            except Exception as e:
                print(f"  sim {s+1}/{N_SIMS}: [ERROR] {e}")

        # Save aggregated sim summary
        if sim_results:
            pnls = [r["total_pnl"] for r in sim_results]
            positions = [r["avg_abs_position"] for r in sim_results]
            fills = [r["n_total_fills"] for r in sim_results]
            summary = {
                "n_sims": len(sim_results),
                "pnl_mean": float(np.mean(pnls)),
                "pnl_std": float(np.std(pnls)),
                "pnl_median": float(np.median(pnls)),
                "avg_position_mean": float(np.mean(positions)),
                "avg_fills": float(np.mean(fills)),
                "individual": sim_results,
            }
            with open(os.path.join(rl_dir, "sim_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  Summary: PnL={summary['pnl_mean']:.4f} +/- {summary['pnl_std']:.4f}  "
                  f"Avg|Pos|={summary['avg_position_mean']:.2f}  Fills={summary['avg_fills']:.0f}")

        elapsed = time.time() - t_start
        print(f"[DONE] {run_name} in {elapsed:.1f}s")

    total = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"Grid search complete in {total:.1f}s")
    print(f"Results in: output/{output_name}_*/")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Grid search over RL parameters")
    parser.add_argument("-i", "--input", required=True,
                        help="Path to base experiment output (e.g. output/experiment2)")
    parser.add_argument("-o", "--output", required=True,
                        help="Output name prefix (results go to output/<name>_ns*_l2*/)")
    parser.add_argument("--config", default="experiments/grid.json",
                        help="JSON config for base params (default: experiments/grid.json)")
    args = parser.parse_args()
    main(args.input, args.output, args.config)
