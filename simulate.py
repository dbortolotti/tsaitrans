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
    base_experiment = config.get("base_experiment")

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
    if not os.path.exists(checkpoint_dir) and base_experiment:
        checkpoint_dir = os.path.join("output", base_experiment, "checkpoints")
    norm_mean = float(np.load(os.path.join(checkpoint_dir, "mean.npy")))
    norm_std = float(np.load(os.path.join(checkpoint_dir, "std.npy")))

    # --- Load transformer and generate predictions ---
    from model import FactorTransformer, normalize_horizons

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        tf_config = json.load(f)

    context_len = tf_config.get("context_len", 60)
    probabilistic = tf_config.get("probabilistic", False)
    horizons = normalize_horizons(tf_config)
    H = len(horizons)

    model = FactorTransformer(
        n_stocks=1,
        context_len=context_len,
        horizons=horizons,
        d_model=tf_config.get("d_model", 64),
        n_heads=tf_config.get("n_heads", 4),
        n_layers=tf_config.get("n_layers", 3),
        ffn_dim=tf_config.get("ffn_dim", 256),
        dropout=0.0,
        probabilistic=probabilistic,
    ).to(device)

    ckpt = torch.load(
        os.path.join(checkpoint_dir, "best_model.pt"),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Generate per-horizon predictions (no aggregation)
    mu_predictions = np.zeros((T, H), dtype=np.float32)
    sigma_predictions = np.ones((T, H), dtype=np.float32)
    normed_series = (returns_raw - norm_mean) / norm_std

    with torch.no_grad():
        for t in range(context_len, T):
            window = normed_series[t - context_len : t].reshape(1, context_len, 1)
            x = torch.tensor(window, dtype=torch.float32, device=device)
            if probabilistic:
                mu, log_sigma = model(x)
                mu_predictions[t, :] = mu[0, :, 0].cpu().numpy()
                sigma_predictions[t, :] = torch.exp(log_sigma[0, :, 0]).cpu().numpy()
            else:
                pred = model(x)
                mu_predictions[t, :] = pred[0, :, 0].cpu().numpy()

    mode = "probabilistic" if probabilistic else "deterministic"
    print(f"[INFO] Transformer predictions generated [{mode}, horizons={horizons}]")

    # --- Load RL policy ---
    from policy import ActorCritic

    obs_dim = 6 + 2 * H
    rl_dir = os.path.join(experiment_dir, "checkpoints_rl")
    policy = ActorCritic(
        obs_dim=obs_dim, act_dim=5, hidden=64,
        log_std_init=rl_cfg.get("log_std_init", -0.5),
        log_std_min=rl_cfg.get("log_std_min", -3.0),
        log_std_max=rl_cfg.get("log_std_max", 1.0),
    ).to(device)

    policy_path = os.path.join(rl_dir, "best_policy.pt")
    norm_path = os.path.join(rl_dir, "best_normalizer.npz")
    if not os.path.exists(policy_path):
        policy_path = os.path.join(rl_dir, "final_policy.pt")
        norm_path = os.path.join(rl_dir, "final_normalizer.npz")
    policy.load_state_dict(torch.load(policy_path, map_location=device, weights_only=True))
    policy.eval()
    print(f"[INFO] Loaded policy from {policy_path}")

    # Load observation normalizer (if available — older checkpoints won't have it)
    obs_mean = np.zeros(obs_dim, dtype=np.float64)
    obs_var = np.ones(obs_dim, dtype=np.float64)
    if os.path.exists(norm_path):
        norm_data = np.load(norm_path)
        obs_mean = norm_data["obs_mean"]
        obs_var = norm_data["obs_var"]
        print(f"[INFO] Loaded obs normalizer from {norm_path}")
    obs_std = np.sqrt(obs_var + 1e-8)

    # --- Run simulation using MarketMakingEnv ---
    from market_env import MarketMakingEnv

    env = MarketMakingEnv(
        returns=returns_raw,
        mu_predictions=mu_predictions,
        sigma_predictions=sigma_predictions,
        half_spread=rl_cfg.get("half_spread", 0.001),
        max_position=rl_cfg.get("max_position", 5),
        k_max=rl_cfg.get("k_max", 3.0),
        kappa_base=rl_cfg.get("kappa_base", 1e-4),
        kappa_close=rl_cfg.get("kappa_close", 5e-4),
        lambda2=rl_cfg.get("lambda2", 1.5),
        alignment_coef=rl_cfg.get("alignment_coef", 0.05),
        alignment_clip=rl_cfg.get("alignment_clip", 3.0),
        trade_penalty=rl_cfg.get("trade_penalty", 0.0),
    )

    obs, _ = env.reset()
    total_reward = 0.0

    while True:
        obs_norm = (obs - obs_mean) / obs_std
        obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
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
        "bids": [round(p, 4) if np.isfinite(p) else None for p in log["bid_prices"]],
        "asks": [round(p, 4) if np.isfinite(p) else None for p in log["offer_prices"]],
        "positions": log["positions"],
        "cum_pnl": [round(p, 4) for p in cum_pnl],
        "fills_buy_t": log["fills_buy_t"],
        "fills_buy_type": log.get("fills_buy_type", []),
        "fills_sell_t": log["fills_sell_t"],
        "fills_sell_type": log.get("fills_sell_type", []),
    }

    # Save alongside the RL checkpoint (results/ may be a shared symlink in grid runs)
    os.makedirs(rl_dir, exist_ok=True)
    out_path = os.path.join(rl_dir, f"sim_results_{seed}.json")
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
