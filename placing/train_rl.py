"""
train_rl.py

Trains the market-making RL agent using PPO.

Uses stock-based splits: RL trains on stocks that are out-of-sample for the
transformer, so the policy learns from realistic (not overfit) predictions.

Can be called standalone or imported by run_experiment.py.

Standalone usage:
    python train_rl.py --data data/returns.npy \
        --transformer checkpoints/run1 \
        --split '{"rl_train":[5,6],"rl_val":[7,8],"test":[9]}' \
        --save_dir checkpoints_rl/run1
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

try:
    from .market_env import MarketMakingEnv, VectorizedMarketEnv
    from .policy import ActorCritic, RolloutBuffer, RunningMeanStd, ppo_update
except ImportError:
    from market_env import MarketMakingEnv, VectorizedMarketEnv
    from policy import ActorCritic, RolloutBuffer, RunningMeanStd, ppo_update


logger = logging.getLogger(__name__)


def load_transformer_predictions(returns, stock_indices, checkpoint_dir):
    """
    Load trained univariate transformer and generate per-horizon predictions.

    Returns:
        mu_predictions: (T, n_stocks, H) predicted cumulative returns per horizon
        sigma_predictions: (T, n_stocks, H) uncertainty per horizon (ones if deterministic)
        mean, std: normalization stats
        horizons: list of horizon values

    No aggregation is performed — the full multi-horizon structure is preserved.
    """
    try:
        pred_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prediction")
        if pred_dir not in sys.path:
            sys.path.insert(0, pred_dir)

        from model import FactorTransformer, normalize_horizons

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        with open(os.path.join(checkpoint_dir, "config.json")) as f:
            config = json.load(f)

        mean = float(np.load(os.path.join(checkpoint_dir, "mean.npy")))
        std = float(np.load(os.path.join(checkpoint_dir, "std.npy")))

        probabilistic = config.get("probabilistic", False)
        horizons = normalize_horizons(config)
        H = len(horizons)

        model = FactorTransformer(
            n_stocks=1,
            context_len=config.get("context_len", 60),
            horizons=horizons,
            d_model=config.get("d_model", 64),
            n_heads=config.get("n_heads", 4),
            n_layers=config.get("n_layers", 3),
            ffn_dim=config.get("ffn_dim", 256),
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

        T = returns.shape[0]
        n_stocks = len(stock_indices)
        mu_predictions = np.zeros((T, n_stocks, H))
        sigma_predictions = np.ones((T, n_stocks, H))  # ones default for deterministic
        ctx_len = config.get("context_len", 60)
        logger.info(
            "Generating transformer predictions for %d RL stocks with T=%d, context_len=%d",
            n_stocks, T, ctx_len,
        )

        with torch.no_grad():
            for si, stock_idx in enumerate(stock_indices):
                series = returns[:, stock_idx]
                normed = (series - mean) / std
                for t in range(ctx_len, T):
                    window = normed[t - ctx_len : t].reshape(1, ctx_len, 1)
                    x = torch.tensor(window, dtype=torch.float32, device=device)

                    if probabilistic:
                        mu, log_sigma = model(x)  # each (1, H, 1)
                        mu_predictions[t, si, :] = mu[0, :, 0].cpu().numpy()
                        sigma_predictions[t, si, :] = torch.exp(log_sigma[0, :, 0]).cpu().numpy()
                    else:
                        pred = model(x)  # (1, H, 1)
                        mu_predictions[t, si, :] = pred[0, :, 0].cpu().numpy()

                if (si + 1) % max(1, n_stocks // 10) == 0 or si + 1 == n_stocks:
                    logger.info("Prediction loading progress: %d/%d RL stocks", si + 1, n_stocks)

        mode = "probabilistic" if probabilistic else "deterministic"
        logger.info("Loaded transformer predictions from %s [%s, horizons=%s]", checkpoint_dir, mode, horizons)
        return mu_predictions, sigma_predictions, mean, std, horizons

    except Exception as e:
        logger.warning("Could not load transformer: %s", e)
        logger.warning("Falling back to momentum predictor")
        return None


def momentum_predictor(returns, stock_indices, mean, std, horizons, lookback=10):
    """
    Simple momentum predictor in normalized space.
    Returns (mu, sigma) each of shape (T, len(stock_indices), H).
    Sigma is constant ones (no uncertainty estimate from momentum).
    """
    T = returns.shape[0]
    H = len(horizons)
    n_stocks = len(stock_indices)
    mu_predictions = np.zeros((T, n_stocks, H))
    sigma_predictions = np.ones((T, n_stocks, H))
    for si, stock_idx in enumerate(stock_indices):
        normed = (returns[:, stock_idx] - mean) / std
        for t in range(lookback, T):
            mom = normed[t - lookback : t].mean()
            # Scale momentum signal by horizon (longer horizon = larger expected move)
            for hi, h in enumerate(horizons):
                mu_predictions[t, si, hi] = mom * h
    return mu_predictions, sigma_predictions


def compute_rollout_diagnostics(actions_buf, obs_buf, horizons):
    """
    Compute per-iteration diagnostics from rollout data.

    actions_buf: list of (n_envs, act_dim) arrays
    obs_buf: list of (n_envs, obs_dim) arrays (pre-normalization)
    """
    actions = np.concatenate(actions_buf, axis=0)  # (steps*envs, act_dim)
    obs = np.concatenate(obs_buf, axis=0)

    H = len(horizons)

    # Observation layout:
    # [position_norm, time_remaining, bid_active, live_k_bid,
    #  offer_active, live_k_offer, mu_1..mu_H, sigma_1..sigma_H]
    positions = obs[:, 0]
    mu_long = obs[:, 6 + H - 1]
    sigma_long = obs[:, 6 + 2 * H - 1]
    z_long = mu_long / (sigma_long + 1e-8)

    mean_abs_action = float(np.mean(np.abs(actions)))
    mean_abs_position = float(np.mean(np.abs(positions)))

    # Turnover: abs change in position across consecutive steps (approximate)
    turnover = float(np.mean(np.abs(np.diff(positions))))

    # Fraction near-zero actions (|action| < 0.05)
    frac_near_zero = float(np.mean(np.abs(actions) < 0.05))

    # Use the aggressive-decision channel as a simple directional proxy.
    action_flat = actions[:, 0]
    if len(action_flat) > 1 and np.std(action_flat) > 1e-12 and np.std(z_long) > 1e-12:
        corr = float(np.corrcoef(action_flat, z_long)[0, 1])
    else:
        corr = 0.0

    return {
        "mean_abs_action": mean_abs_action,
        "mean_abs_position": mean_abs_position,
        "turnover": turnover,
        "frac_near_zero": frac_near_zero,
        "action_signal_corr": corr if np.isfinite(corr) else 0.0,
    }


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(
    config: dict,
    returns: np.ndarray,
    stock_split: dict,
    save_dir: str,
    transformer_checkpoint: str = None,
):
    """
    Train the RL market-making agent.

    Args:
        config: RL hyperparams
        returns: (T, N_total) full returns array
        stock_split: dict with 'rl_train', 'rl_val', 'test' keys
        save_dir: where to save RL checkpoints
        transformer_checkpoint: path to transformer checkpoint dir (optional)
    """
    device = get_device()
    logger.info("Using device: %s", device)
    logger.info("Data shape: %s", returns.shape)

    rl_train_stocks = stock_split["rl_train"]
    rl_val_stocks = stock_split["rl_val"]
    logger.info("RL train stocks: %s", rl_train_stocks)
    logger.info("RL val stocks:   %s", rl_val_stocks)

    # Get predictions and normalization stats
    predictor = config.get("predictor", "transformer")
    norm_mean, norm_std = None, None
    mu_predictions, sigma_predictions, horizons = None, None, None

    if predictor == "transformer" and transformer_checkpoint:
        result = load_transformer_predictions(
            returns, rl_train_stocks, transformer_checkpoint
        )
        if result is not None:
            mu_predictions, sigma_predictions, norm_mean, norm_std, horizons = result

    # Fallback: compute normalization from transformer checkpoint or RL train stocks
    if norm_mean is None or norm_std is None:
        if transformer_checkpoint:
            norm_mean = float(np.load(os.path.join(transformer_checkpoint, "mean.npy")))
            norm_std = float(np.load(os.path.join(transformer_checkpoint, "std.npy")))
        else:
            train_data = returns[:, rl_train_stocks]
            norm_mean = float(train_data.mean())
            norm_std = float(train_data.std()) + 1e-8

    # Default horizons if not loaded from transformer
    if horizons is None:
        horizons = config.get("horizons", [1, 2, 4, 8, 16])
    H = len(horizons)

    if mu_predictions is None:
        mu_predictions, sigma_predictions = momentum_predictor(
            returns, rl_train_stocks, norm_mean, norm_std, horizons
        )
        logger.info("Using momentum predictor")

    logger.info("Normalization: mean=%.4f, std=%.4f", norm_mean, norm_std)
    logger.info("Horizons: %s (H=%d, obs_dim=%d)", horizons, H, 6 + 2 * H)

    # Extract config
    n_envs = config.get("n_envs", 4)
    n_iterations = config.get("n_iterations", 500)
    rollout_steps = config.get("rollout_steps", 2048)
    actor_lr = config.get("actor_lr", 3e-4)
    critic_lr = config.get("critic_lr", 1e-3)
    gamma = config.get("gamma", 0.99)
    lam = config.get("gae_lambda", 0.95)
    ent_coef_start = config.get("ent_coef", 5e-4)
    ent_coef_end = config.get("ent_coef_end", 1e-5)
    target_kl = config.get("target_kl", 0.02)
    normalize_rewards = config.get("normalize_rewards", True)

    ent_coef_anneal = config.get("ent_coef_anneal", True)
    disable_reward_norm = config.get("disable_reward_norm", False)
    use_uncertainty = config.get("use_uncertainty", False)
    half_spread = config.get("half_spread", 0.001)
    max_position = config.get("max_position", 5)
    k_max = config.get("k_max", 3.0)
    kappa_base = config.get("kappa_base", 1e-4)
    kappa_close = config.get("kappa_close", 5e-4)
    lambda2 = config.get("lambda2", 1.5)

    T = returns.shape[0]
    n_rl_stocks = len(rl_train_stocks)

    act_dim = 5

    logger.info("=== RL Config ===")
    logger.info("gamma               = %s", gamma)
    logger.info("disable_reward_norm = %s", disable_reward_norm)
    logger.info("use_uncertainty     = %s", use_uncertainty)
    logger.info("ent_coef_anneal     = %s", ent_coef_anneal)
    logger.info("half_spread         = %s", half_spread)
    logger.info("max_position        = %s", max_position)
    logger.info("k_max               = %s", k_max)
    logger.info("kappa_base          = %s", kappa_base)
    logger.info("kappa_close         = %s", kappa_close)
    logger.info("lambda2             = %s", lambda2)

    def make_env_data(n):
        """Create n episodes, each a full trading day from a random RL train stock."""
        ret_list, mu_list, sigma_list = [], [], []
        rng = np.random.default_rng()
        for _ in range(n):
            si = rng.integers(0, n_rl_stocks)
            stock_idx = rl_train_stocks[si]
            ret_list.append(returns[:, stock_idx])
            mu = mu_predictions[:, si, :]       # (T, H)
            sigma = sigma_predictions[:, si, :] # (T, H)
            if use_uncertainty:
                # Replace mu with mu/sigma (z-score) — obs dim unchanged
                mu = mu / (sigma + 1e-8)
            mu_list.append(mu)
            sigma_list.append(sigma)
        return ret_list, mu_list, sigma_list

    def make_vec_env(ret_list, mu_list, sigma_list):
        return VectorizedMarketEnv(
            ret_list, mu_list, sigma_list,
            half_spread=half_spread,
            max_position=max_position,
            k_max=k_max,
            kappa_base=kappa_base,
            kappa_close=kappa_close,
            lambda2=lambda2,
        )

    # Policy
    obs_dim = 6 + 2 * H
    log_std_init = config.get("log_std_init", -0.5)
    log_std_min = config.get("log_std_min", -3.0)
    log_std_max = config.get("log_std_max", 1.0)
    policy = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden=64,
                         log_std_init=log_std_init,
                         log_std_min=log_std_min,
                         log_std_max=log_std_max).to(device)
    optimizer = torch.optim.Adam([
        {"params": policy.actor_params(), "lr": actor_lr},
        {"params": policy.critic_params(), "lr": critic_lr},
    ], eps=1e-5)

    # Reward and observation normalization for stable PPO training
    reward_rms = RunningMeanStd(shape=())
    obs_rms = RunningMeanStd(shape=(obs_dim,))

    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info("Policy parameters: %s", f"{n_params:,}")
    logger.info("Action dim: %d", act_dim)

    os.makedirs(save_dir, exist_ok=True)

    best_reward = -np.inf
    log_rows = []

    # Collapse detection state
    low_action_streak = 0
    COLLAPSE_WARN_THRESHOLD = 10  # warn after this many consecutive low-action iters

    baseline_rew_val = 0.0

    logger.info(
        "%5s %10s %10s %8s %10s %10s %10s %10s %8s %8s %8s %8s %10s %10s %7s",
        "Iter", "MeanRew", "CumPnL", "Sharpe", "PolLoss", "VLoss", "Entropy", "KL",
        "|Act|", "|Pos|", "Corr", "LogStd", "RawRewStd", "RewStd", "Time",
    )
    logger.info("%s", "-" * 135)

    t_start = time.time()

    for iteration in range(n_iterations):
        iter_start = time.time()

        ret_list, mu_list, sigma_list = make_env_data(n_envs)
        vec_env = make_vec_env(ret_list, mu_list, sigma_list)

        buffer = RolloutBuffer()
        obs = vec_env.reset()
        obs_rms.update(obs)
        episode_rewards = []
        raw_rewards_buf = []
        raw_pnl_buf = []
        raw_actions_buf = []
        raw_obs_buf = []
        rollout_cum_pnl = None

        for step in range(rollout_steps):
            obs_norm = (obs - obs_rms.mean) / obs_rms.std
            obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device)
            actions, log_probs, values = policy.get_action(obs_t)

            next_obs, rewards, dones, infos = vec_env.step(actions)

            obs_rms.update(next_obs)
            raw_rewards_buf.append(rewards.copy())
            raw_pnl_buf.append(
                np.array([info.get("step_pnl", 0.0) for info in infos], dtype=np.float64)
            )
            rollout_cum_pnl = np.array(
                [info.get("cumulative_pnl", 0.0) for info in infos], dtype=np.float64
            )
            if disable_reward_norm:
                rewards_norm = rewards
            else:
                reward_rms.update(rewards)
                rewards_norm = rewards / reward_rms.std

            buffer.add(obs_norm, actions, log_probs, rewards_norm, values, dones)
            raw_actions_buf.append(actions.copy())
            raw_obs_buf.append(obs.copy())
            obs = next_obs
            episode_rewards.append(rewards.mean())  # log raw rewards

        with torch.no_grad():
            obs_norm = (obs - obs_rms.mean) / obs_rms.std
            obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device)
            _, _, last_values = policy.forward(obs_t)
            last_values = last_values.cpu().numpy()

        buffer.compute_returns_and_advantages(last_values, gamma, lam)
        # Entropy coefficient: anneal or hold constant
        if ent_coef_anneal:
            frac = iteration / max(n_iterations - 1, 1)
            ent_coef = ent_coef_start + (ent_coef_end - ent_coef_start) * frac
        else:
            ent_coef = ent_coef_start
        ppo_metrics = ppo_update(policy, optimizer, buffer, device,
                                 ent_coef=ent_coef, target_kl=target_kl,
                                 disable_reward_norm=disable_reward_norm)

        # NaN detection — abort gracefully instead of crashing
        has_nan = any(torch.isnan(p).any() for p in policy.parameters())
        if has_nan or np.isnan(ppo_metrics["policy_loss"]):
            logger.warning("NaN detected in policy at iteration %d - aborting training early", iteration)
            break

        # Diagnostics
        diag = compute_rollout_diagnostics(raw_actions_buf, raw_obs_buf, horizons)

        mean_reward = np.mean(episode_rewards)
        raw_rew_std = float(np.std(np.concatenate(raw_rewards_buf)))  # RewScaleObserved
        pnl_series = np.concatenate(raw_pnl_buf) if raw_pnl_buf else np.array([0.0])
        mean_cum_pnl = float(np.mean(rollout_cum_pnl)) if rollout_cum_pnl is not None else 0.0
        pnl_std = float(np.std(pnl_series))
        sharpe = float(np.mean(pnl_series) / (pnl_std + 1e-10) * np.sqrt(len(pnl_series)))
        log_std_val = policy.actor_log_std.data.mean().item()
        kl_flag = " KL!" if ppo_metrics.get("early_stopped", False) else ""
        elapsed = time.time() - iter_start

        # Collapse detection
        if diag["mean_abs_action"] < 0.05:
            low_action_streak += 1
        else:
            low_action_streak = 0
        collapse_flag = ""
        if low_action_streak >= COLLAPSE_WARN_THRESHOLD:
            collapse_flag = " ⚠ COLLAPSE"
        if ppo_metrics["entropy"] < 0.1:
            collapse_flag += " ⚠ LOW-ENT"

        logger.info(
            "%5d %10.4f %10.4f %8.2f %10.4f %10.4f %10.4f %10.4f %8.4f %8.4f %8.4f %8.4f %10.6f %10.2f %6.1fs%s%s",
            iteration, mean_reward, mean_cum_pnl, sharpe, ppo_metrics["policy_loss"],
            ppo_metrics["value_loss"], ppo_metrics["entropy"], ppo_metrics["approx_kl"],
            diag["mean_abs_action"], diag["mean_abs_position"], diag["action_signal_corr"],
            log_std_val, raw_rew_std, float(reward_rms.std), elapsed, kl_flag, collapse_flag,
        )

        log_rows.append(
            {
                "iteration": iteration,
                "mean_reward": float(mean_reward),
                "mean_cumulative_pnl": mean_cum_pnl,
                "sharpe": sharpe,
                "reward_std": float(np.std(episode_rewards)),
                "raw_reward_std": raw_rew_std,
                **ppo_metrics,
                **diag,
                "ent_coef": ent_coef,
                "log_std": log_std_val,
                "reward_rms_std": float(reward_rms.std),
                "obs_mean": obs_rms.mean.tolist(),
                "obs_std": obs_rms.std.tolist(),
                "elapsed": elapsed,
            }
        )

        if mean_reward > best_reward:
            best_reward = mean_reward
            torch.save(policy.state_dict(), os.path.join(save_dir, "best_policy.pt"))
            np.savez(os.path.join(save_dir, "best_normalizer.npz"),
                     obs_mean=obs_rms.mean, obs_var=obs_rms.var, obs_count=obs_rms.count)

    total_time = time.time() - t_start
    logger.info("Training complete in %.1fs", total_time)
    logger.info("Best mean reward: %.4f", best_reward)

    # Final summary + success gate
    success_gate = False
    if len(log_rows) > 20:
        last20 = log_rows[-20:]
        avg_rew = np.mean([r["mean_reward"] for r in last20])
        avg_pos = np.mean([r["mean_abs_position"] for r in last20])
        avg_act = np.mean([r["mean_abs_action"] for r in last20])
        avg_corr = np.mean([r["action_signal_corr"] for r in last20])
        logger.info(
            "Last 20 iters: mean_reward=%.4f, |pos|=%.4f, |action|=%.4f, signal_corr=%.4f",
            avg_rew, avg_pos, avg_act, avg_corr,
        )

        # Simple market-making success gate: non-trivial inventory usage and
        # non-degenerate action magnitude in at least one recent iteration.
        for row in last20:
            if row["mean_abs_position"] >= 0.1 and row["mean_abs_action"] >= 0.05:
                success_gate = True
                break

        if success_gate:
            logger.info("SUCCESS GATE PASSED (non-trivial position and action usage)")
        else:
            logger.info("Policy still looks close to degenerate")
    elif len(log_rows) > 0:
        last = log_rows[-min(10, len(log_rows)):]
        avg_rew = np.mean([r["mean_reward"] for r in last])
        avg_act = np.mean([r["mean_abs_action"] for r in last])
        avg_corr = np.mean([r["action_signal_corr"] for r in last])
        logger.info(
            "Last iters: mean_reward=%.4f, |action|=%.4f, signal_corr=%.4f",
            avg_rew, avg_act, avg_corr,
        )

    torch.save(policy.state_dict(), os.path.join(save_dir, "final_policy.pt"))
    np.savez(os.path.join(save_dir, "final_normalizer.npz"),
             obs_mean=obs_rms.mean, obs_var=obs_rms.var, obs_count=obs_rms.count)

    with open(os.path.join(save_dir, "train_log.json"), "w") as f:
        json.dump(log_rows, f, indent=2)

    save_config = {
        **config,
        "stock_split": stock_split,
        "transformer_checkpoint": transformer_checkpoint,
        "total_time": total_time,
        "success_gate": success_gate,
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(save_config, f, indent=2)

    logger.info("Saved to %s/", save_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    parser = argparse.ArgumentParser(description="Train RL market-making agent")
    parser.add_argument("--data", type=str, required=True, help="Path to returns .npy")
    parser.add_argument("--transformer", type=str, default=None,
                        help="Path to transformer checkpoint dir (optional)")
    parser.add_argument("--split", type=str, required=True,
                        help='JSON string with stock split')
    parser.add_argument("--save_dir", type=str, default="checkpoints_rl/run1")
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--n_iterations", type=int, default=500)
    parser.add_argument("--rollout_steps", type=int, default=2048)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--half_spread", type=float, default=0.001)
    parser.add_argument("--max_position", type=int, default=5)
    parser.add_argument("--k_max", type=float, default=3.0)
    parser.add_argument("--kappa_base", type=float, default=1e-4)
    parser.add_argument("--kappa_close", type=float, default=5e-4)
    parser.add_argument("--lambda2", type=float, default=1.5)
    args = parser.parse_args()

    returns = np.load(args.data)
    stock_split = json.loads(args.split)
    config = {k: v for k, v in vars(args).items()
              if k not in ("data", "transformer", "split", "save_dir")}

    train(config, returns, stock_split, args.save_dir,
          transformer_checkpoint=args.transformer)
