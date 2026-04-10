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
import os
import sys
import time

import numpy as np
import torch

from market_env import MarketMakingEnv, VectorizedMarketEnv
from policy import ActorCritic, RolloutBuffer, RunningMeanStd, ppo_update


def load_transformer_predictions(returns, stock_indices, checkpoint_dir):
    """
    Load trained univariate transformer and generate predictions for specified stocks.
    Returns (T, len(stock_indices)) array of predicted next-step returns in normalized space.
    Also returns (mean, std) used for normalization.
    """
    try:
        pred_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prediction")
        if pred_dir not in sys.path:
            sys.path.insert(0, pred_dir)

        from model import FactorTransformer

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        with open(os.path.join(checkpoint_dir, "config.json")) as f:
            config = json.load(f)

        mean = float(np.load(os.path.join(checkpoint_dir, "mean.npy")))
        std = float(np.load(os.path.join(checkpoint_dir, "std.npy")))

        model = FactorTransformer(
            n_stocks=1,
            context_len=config.get("context_len", 60),
            horizon=config.get("horizon", 1),
            d_model=config.get("d_model", 64),
            n_heads=config.get("n_heads", 4),
            n_layers=config.get("n_layers", 3),
            ffn_dim=config.get("ffn_dim", 256),
            dropout=0.0,
        ).to(device)

        ckpt = torch.load(
            os.path.join(checkpoint_dir, "best_model.pt"),
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        T = returns.shape[0]
        predictions = np.zeros((T, len(stock_indices)))
        ctx_len = config.get("context_len", 60)

        # Predictions stay in normalized space (O(1) magnitude)
        with torch.no_grad():
            for si, stock_idx in enumerate(stock_indices):
                series = returns[:, stock_idx]
                normed = (series - mean) / std
                for t in range(ctx_len, T):
                    window = normed[t - ctx_len : t].reshape(1, ctx_len, 1)
                    x = torch.tensor(window, dtype=torch.float32, device=device)
                    pred = model(x)  # (1, 1, 1)
                    predictions[t, si] = pred[0, 0, 0].item()

        print(f"[INFO] Loaded transformer predictions from {checkpoint_dir}")
        return predictions, mean, std

    except Exception as e:
        print(f"[WARN] Could not load transformer: {e}")
        print("[WARN] Falling back to momentum predictor")
        return None


def momentum_predictor(returns, stock_indices, mean, std, lookback=10):
    """
    Simple momentum predictor in normalized space.
    Returns (T, len(stock_indices)) of normalized predictions.
    """
    T = returns.shape[0]
    predictions = np.zeros((T, len(stock_indices)))
    for si, stock_idx in enumerate(stock_indices):
        normed = (returns[:, stock_idx] - mean) / std
        for t in range(lookback, T):
            predictions[t, si] = normed[t - lookback : t].mean()
    return predictions


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
        config: RL hyperparams (half_spread, n_sigma, etc.)
        returns: (T, N_total) full returns array
        stock_split: dict with 'rl_train', 'rl_val', 'test' keys
        save_dir: where to save RL checkpoints
        transformer_checkpoint: path to transformer checkpoint dir (optional)
    """
    device = get_device()
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Data shape: {returns.shape}")

    rl_train_stocks = stock_split["rl_train"]
    rl_val_stocks = stock_split["rl_val"]
    print(f"[INFO] RL train stocks: {rl_train_stocks}")
    print(f"[INFO] RL val stocks:   {rl_val_stocks}")

    # Get predictions and normalization stats
    predictor = config.get("predictor", "transformer")
    norm_mean, norm_std = None, None

    if predictor == "transformer" and transformer_checkpoint:
        result = load_transformer_predictions(
            returns, rl_train_stocks, transformer_checkpoint
        )
        if result is not None:
            predictions, norm_mean, norm_std = result
        else:
            predictions = None

    # Fallback: compute normalization from transformer checkpoint or RL train stocks
    if norm_mean is None or norm_std is None:
        if transformer_checkpoint:
            norm_mean = float(np.load(os.path.join(transformer_checkpoint, "mean.npy")))
            norm_std = float(np.load(os.path.join(transformer_checkpoint, "std.npy")))
        else:
            train_data = returns[:, rl_train_stocks]
            norm_mean = float(train_data.mean())
            norm_std = float(train_data.std()) + 1e-8

    if predictions is None:
        predictions = momentum_predictor(returns, rl_train_stocks, norm_mean, norm_std)
        print("[INFO] Using momentum predictor")

    print(f"[INFO] Normalization: mean={norm_mean:.4f}, std={norm_std:.4f}")

    # Extract config
    n_envs = config.get("n_envs", 32)
    n_iterations = config.get("n_iterations", 200)
    rollout_steps = config.get("rollout_steps", 512)
    lr = config.get("lr", 3e-4)
    gamma = config.get("gamma", 0.99)
    lam = config.get("gae_lambda", 0.95)

    # New env parameters
    half_spread = config.get("half_spread", 0.001)
    target_vol = config.get("target_vol", 0.02)

    # Auto-load R² from transformer metrics if not explicitly set
    r_squared = config.get("r_squared", None)
    if r_squared is None and transformer_checkpoint:
        # metrics.json lives in results/ sibling to checkpoints/
        experiment_dir = os.path.dirname(transformer_checkpoint)
        metrics_path = os.path.join(experiment_dir, "results", "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            r_squared = max(metrics.get("r2_vs_naive", 0.01), 1e-4)
            print(f"[INFO] Auto-loaded R²={r_squared:.4f} from transformer metrics")
    if r_squared is None:
        r_squared = 0.01
    n_sigma = config.get("n_sigma", 1.0)
    tau = config.get("tau", 20)
    lambda2 = config.get("lambda2", 1.5)
    max_width = config.get("max_width", 3.0)
    max_skew = config.get("max_skew", 3.0)

    T = returns.shape[0]
    n_rl_stocks = len(rl_train_stocks)

    def make_env_data(n):
        """Create n episodes, each a full trading day from a random RL train stock."""
        ret_list, pred_list = [], []
        rng = np.random.default_rng()
        for _ in range(n):
            si = rng.integers(0, n_rl_stocks)
            stock_idx = rl_train_stocks[si]
            ret_list.append(returns[:, stock_idx])
            pred_list.append(predictions[:, si])
        return ret_list, pred_list

    # Policy
    policy = ActorCritic(obs_dim=4, act_dim=2, hidden=64).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

    # Reward and observation normalization for stable PPO training
    reward_rms = RunningMeanStd(shape=())
    obs_rms = RunningMeanStd(shape=(4,))

    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[INFO] Policy parameters: {n_params:,}")
    print(f"[INFO] Env params: half_spread={half_spread}, target_vol={target_vol}, "
          f"R²={r_squared}, n_sigma={n_sigma}, tau={tau}, lambda2={lambda2}")

    os.makedirs(save_dir, exist_ok=True)

    best_reward = -np.inf
    log_rows = []

    print(
        f"\n{'Iter':>5} {'MeanRew':>10} {'PolLoss':>10} {'VLoss':>10} "
        f"{'Entropy':>10} {'RewStd':>10} {'Time':>8}"
    )
    print("-" * 72)

    t_start = time.time()

    for iteration in range(n_iterations):
        iter_start = time.time()

        ret_list, pred_list = make_env_data(n_envs)
        vec_env = VectorizedMarketEnv(
            ret_list,
            pred_list,
            half_spread=half_spread,
            target_vol=target_vol,
            r_squared=r_squared,
            n_sigma=n_sigma,
            tau=tau,
            lambda2=lambda2,
            max_width=max_width,
            max_skew=max_skew,
        )

        buffer = RolloutBuffer()
        obs = vec_env.reset()
        obs_rms.update(obs)
        episode_rewards = []

        for step in range(rollout_steps):
            obs_norm = (obs - obs_rms.mean) / obs_rms.std
            obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device)
            actions, log_probs, values = policy.get_action(obs_t)

            next_obs, rewards, dones = vec_env.step(actions)

            reward_rms.update(rewards)
            obs_rms.update(next_obs)
            rewards_norm = rewards / reward_rms.std

            buffer.add(obs_norm, actions, log_probs, rewards_norm, values, dones)
            obs = next_obs
            episode_rewards.append(rewards.mean())  # log raw rewards

        with torch.no_grad():
            obs_norm = (obs - obs_rms.mean) / obs_rms.std
            obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device)
            _, _, last_values = policy.forward(obs_t)
            last_values = last_values.cpu().numpy()

        buffer.compute_returns_and_advantages(last_values, gamma, lam)
        metrics = ppo_update(policy, optimizer, buffer, device)

        mean_reward = np.mean(episode_rewards)
        elapsed = time.time() - iter_start
        print(
            f"{iteration:5d} {mean_reward:10.4f} {metrics['policy_loss']:10.4f} "
            f"{metrics['value_loss']:10.4f} {metrics['entropy']:10.4f} "
            f"{float(reward_rms.std):10.2f} {elapsed:7.1f}s"
        )

        log_rows.append(
            {
                "iteration": iteration,
                "mean_reward": float(mean_reward),
                **metrics,
                "reward_std": float(reward_rms.std),
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
    print(f"\n[INFO] Training complete in {total_time:.1f}s")
    print(f"[INFO] Best mean reward: {best_reward:.4f}")

    torch.save(policy.state_dict(), os.path.join(save_dir, "final_policy.pt"))
    np.savez(os.path.join(save_dir, "final_normalizer.npz"),
             obs_mean=obs_rms.mean, obs_var=obs_rms.var, obs_count=obs_rms.count)

    with open(os.path.join(save_dir, "train_log.json"), "w") as f:
        json.dump(log_rows, f, indent=2)

    rl_config = {
        **config,
        "stock_split": stock_split,
        "transformer_checkpoint": transformer_checkpoint,
        "total_time": total_time,
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(rl_config, f, indent=2)

    print(f"[INFO] Saved to {save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL market-making agent")
    parser.add_argument("--data", type=str, required=True, help="Path to returns .npy")
    parser.add_argument("--transformer", type=str, default=None,
                        help="Path to transformer checkpoint dir (optional)")
    parser.add_argument("--split", type=str, required=True,
                        help='JSON string with stock split')
    parser.add_argument("--save_dir", type=str, default="checkpoints_rl/run1")
    parser.add_argument("--n_envs", type=int, default=32)
    parser.add_argument("--n_iterations", type=int, default=200)
    parser.add_argument("--rollout_steps", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--half_spread", type=float, default=0.001)
    parser.add_argument("--target_vol", type=float, default=0.02)
    parser.add_argument("--r_squared", type=float, default=0.01)
    parser.add_argument("--n_sigma", type=float, default=1.0)
    parser.add_argument("--tau", type=int, default=20)
    parser.add_argument("--lambda2", type=float, default=1.5)
    args = parser.parse_args()

    returns = np.load(args.data)
    stock_split = json.loads(args.split)
    config = {k: v for k, v in vars(args).items()
              if k not in ("data", "transformer", "split", "save_dir")}

    train(config, returns, stock_split, args.save_dir,
          transformer_checkpoint=args.transformer)
