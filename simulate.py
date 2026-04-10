"""
simulate.py

Generates a fresh stock from the experiment's DGP, runs the trained transformer
and RL policy on it, and exports sim_results.json for the visualizer.

Usage:
    python simulate.py output/example
    python simulate.py output/example --seed 123    # different realization
    python simulate.py output/example --deterministic  # no policy noise

Opens trading_visualizer.html automatically if available.
"""

import json
import os
import sys

import numpy as np
import torch


def main(experiment_dir: str, seed: int = None, deterministic: bool = False):
    # --- Load resolved config ---
    config_path = os.path.join(experiment_dir, "resolved_config.json")
    if not os.path.exists(config_path):
        # Fall back to the experiment JSON
        name = os.path.basename(experiment_dir)
        config_path = os.path.join(experiment_dir, f"{name}.json")
    with open(config_path) as f:
        config = json.load(f)

    data_cfg = config["data"]
    rl_cfg = config.get("rl", {})

    # Use a different seed from training so we get a fresh realization
    if seed is None:
        seed = data_cfg.get("seed", 42) + 1000

    print(f"[INFO] Experiment: {experiment_dir}")
    print(f"[INFO] Simulation seed: {seed}")

    # --- Add module dirs to path ---
    repo_root = os.path.dirname(os.path.abspath(__file__))
    for d in ["prediction", "placing"]:
        p = os.path.join(repo_root, d)
        if p not in sys.path:
            sys.path.insert(0, p)

    # --- Generate a single fresh stock ---
    from generate_data import generate

    result = generate(
        n_stocks=1,  # just one stock for simulation
        n_timesteps=data_cfg.get("n_steps", 2000),
        n_factors=data_cfg.get("n_factors", 3),
        factor_half_life=data_cfg.get("factor_half_life", 0.1),
        noise_half_life_range=tuple(data_cfg.get("noise_half_life_range", [0.005, 0.025])),
        target_vol=data_cfg.get("target_vol", 0.02),
        snr=data_cfg.get("snr", 0.3),
        steps_per_day=data_cfg.get("steps_per_day", 2000),
        seed=seed,
    )
    returns_raw = result["returns"][:, 0].astype(np.float32)  # (T,)
    T = len(returns_raw)
    print(f"[INFO] Generated {T} timesteps, SNR={result['snr'][0]:.2f}")

    # --- Load normalization stats ---
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    norm_mean = float(np.load(os.path.join(checkpoint_dir, "mean.npy")))
    norm_std = float(np.load(os.path.join(checkpoint_dir, "std.npy")))

    # --- Normalize returns (same as RL training) ---
    returns_normed = (returns_raw - norm_mean) / norm_std

    # --- Load transformer and generate predictions ---
    from model import FactorTransformer

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        tf_config = json.load(f)

    context_len = tf_config.get("context_len", 60)

    model = FactorTransformer(
        n_stocks=1,
        context_len=context_len,
        horizon=tf_config.get("horizon", 1),
        d_model=tf_config.get("d_model", 64),
        n_heads=tf_config.get("n_heads", 4),
        n_layers=tf_config.get("n_layers", 3),
        ffn_dim=tf_config.get("ffn_dim", 256),
        dropout=0.0,
    ).to(device)

    ckpt = torch.load(
        os.path.join(checkpoint_dir, "best_model.pt"),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Generate predictions in normalized space
    predictions = np.zeros(T, dtype=np.float32)
    normed_series = (returns_raw - norm_mean) / norm_std

    with torch.no_grad():
        for t in range(context_len, T):
            window = normed_series[t - context_len : t].reshape(1, context_len, 1)
            x = torch.tensor(window, dtype=torch.float32, device=device)
            pred = model(x)
            predictions[t] = pred[0, 0, 0].item()

    print(f"[INFO] Transformer predictions generated")

    # --- Load RL policy ---
    from policy import ActorCritic

    rl_dir = os.path.join(experiment_dir, "checkpoints_rl")
    policy = ActorCritic(obs_dim=4, act_dim=2, hidden=64).to(device)

    policy_path = os.path.join(rl_dir, "best_policy.pt")
    if not os.path.exists(policy_path):
        policy_path = os.path.join(rl_dir, "final_policy.pt")
    policy.load_state_dict(torch.load(policy_path, map_location=device, weights_only=True))
    policy.eval()
    print(f"[INFO] Loaded policy from {policy_path}")

    # --- Run simulation using MarketMakingEnv ---
    from market_env import MarketMakingEnv

    env = MarketMakingEnv(
        returns=returns_raw,
        predictions=predictions,
        half_spread=rl_cfg.get("half_spread", 0.001),
        target_vol=data_cfg.get("target_vol", 0.02),
        r_squared=rl_cfg.get("r_squared", 0.01),
        n_sigma=rl_cfg.get("n_sigma", 1.0),
        tau=rl_cfg.get("tau", 20),
        lambda2=rl_cfg.get("lambda2", 1.5),
        max_width=rl_cfg.get("max_width", 3.0),
        max_skew=rl_cfg.get("max_skew", 3.0),
    )

    obs, _ = env.reset()
    total_reward = 0.0

    while True:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        actions, _, _ = policy.get_action(obs_t, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(actions[0])
        total_reward += reward
        if terminated or truncated:
            break

    log = env.get_log()
    n_steps = len(log["times"])

    cum_pnl = log.get("cumulative_pnls", np.cumsum(log["pnls"]).tolist())

    # Build sim_results.json
    sim_results = {
        "summary": {
            "total_pnl": float(cum_pnl[-1]) if cum_pnl else 0.0,
            "total_reward": float(total_reward),
            "n_buys": len(log["fills_buy_t"]),
            "n_sells": len(log["fills_sell_t"]),
            "n_total_fills": len(log["fills_buy_t"]) + len(log["fills_sell_t"]),
            "avg_abs_position": float(np.mean(np.abs(log["positions"]))) if log["positions"] else 0.0,
            "n_steps": n_steps,
            "seed": seed,
        },
        "times": log["times"],
        "mid_prices": [round(p, 4) for p in log["mid_prices"]],
        "bids": [round(p, 4) for p in log["bids"]],
        "asks": [round(p, 4) for p in log["asks"]],
        "positions": log["positions"],
        "cum_pnl": [round(p, 4) for p in cum_pnl],
        "fills_buy_t": log["fills_buy_t"],
        "fills_buy_type": log.get("fills_buy_type", []),
        "fills_sell_t": log["fills_sell_t"],
        "fills_sell_type": log.get("fills_sell_type", []),
    }

    # Save
    results_dir = os.path.join(experiment_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "sim_results.json")
    with open(out_path, "w") as f:
        json.dump(sim_results, f)

    print(f"\n[INFO] Simulation complete:")
    print(f"  Steps:      {n_steps}")
    print(f"  Fills:      {sim_results['summary']['n_buys']}B / {sim_results['summary']['n_sells']}S")
    print(f"  Total PnL:  {sim_results['summary']['total_pnl']:.4f}")
    print(f"  Total Rew:  {sim_results['summary']['total_reward']:.4f}")
    print(f"  Avg |Pos|:  {sim_results['summary']['avg_abs_position']:.2f}")
    print(f"\nSaved: {out_path}")

    return sim_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simulate trained RL policy on fresh data")
    parser.add_argument("experiment", type=str, help="Path to experiment output dir (e.g. output/example)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for data generation (default: training seed + 1000)")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy (no exploration noise)")
    args = parser.parse_args()

    main(args.experiment, seed=args.seed, deterministic=args.deterministic)
