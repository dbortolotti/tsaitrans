"""
market_env.py

Gymnasium environment for a single-stock market-making agent.

The agent quotes a bid and offer each step, parameterized as (width, skew)
in multiples of half_spread. Fills happen via two mechanisms:
  - Aggressive: agent's current quote crosses the current market → fill at market price
  - Passive: agent's previous resting quote crossed by current market → fill at agent's price

See MARKET_ENV.md for full fill mechanics and RL_DESIGN.md for reward/obs design.

Observation space (3 + 2*H dimensional):
    [position_risk, time_remaining, cumulative_pnl,
     mu_1, ..., mu_H,           -- per-horizon predicted cumulative returns
     sigma_1, ..., sigma_H]     -- per-horizon uncertainty

Action space (2D continuous):
    [width, skew] in [-1, 1], rescaled to [0, max_width] and [-max_skew, max_skew]

Reward:
    r_t = mark_to_market_pnl - (n * position / (R² * sqrt(max(T-t, tau))))²
    terminal: -lambda2 * |position| * half_spread
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MarketMakingEnv(gym.Env):
    """Single-stock market-making environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        returns: np.ndarray,              # (T,) single-stock return series
        mu_predictions: np.ndarray,       # (T, H) predicted cumulative returns per horizon
        sigma_predictions: np.ndarray,    # (T, H) uncertainty per horizon
        half_spread: float = 0.001,       # fixed market half-spread
        target_vol: float = 0.02,         # known DGP daily vol (normalization constant)
        r_squared: float = 0.01,          # transformer R² (measured, not tuned)
        n_sigma: float = 1.0,             # risk tolerance in sigmas (tuning knob)
        tau: int = 20,                    # penalty floor in steps (prevents blow-up near EOD)
        lambda2: float = 1.5,             # EOD liquidation impact multiplier (>= 1)
        max_width: float = 3.0,           # max quote width in half_spread multiples
        max_skew: float = 3.0,            # max quote skew in half_spread multiples
        initial_price: float = 100.0,
    ):
        super().__init__()

        assert len(returns) == len(mu_predictions), "returns and mu_predictions must be same length"
        assert mu_predictions.shape == sigma_predictions.shape, "mu and sigma shape mismatch"

        self.returns = returns.astype(np.float64)
        self.mu_predictions = mu_predictions.astype(np.float64)
        self.sigma_predictions = sigma_predictions.astype(np.float64)
        self.T = len(returns)
        self.H = mu_predictions.shape[1]  # number of horizons

        # Market microstructure
        self.half_spread = half_spread
        self.target_vol = target_vol

        # Reward parameters
        self.r_squared = max(r_squared, 1e-8)  # floor to avoid division by zero
        self.n_sigma = n_sigma
        self.tau = tau
        self.lambda2 = lambda2

        # Action scaling
        self.max_width = max_width
        self.max_skew = max_skew
        self.initial_price = initial_price

        # Spaces: 3 core + H mu + H sigma
        obs_dim = 3 + 2 * self.H
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )
        # Action: [width_raw, skew_raw] in [-1, 1], rescaled internally
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float64
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.position = 0
        self.cumulative_pnl = 0.0

        # Reconstruct price series from returns, scaled by initial price
        self.prices = np.zeros(self.T + 1)
        self.prices[0] = self.initial_price
        for i in range(self.T):
            self.prices[i + 1] = self.prices[i] * (1.0 + self.returns[i])
        # Scaled prices (start at 1.0)
        self.scaled_prices = self.prices / self.initial_price

        # Previous step's quotes (for passive fill checks). None on first step.
        self.prev_bid = None
        self.prev_offer = None

        # Trading log for visualization
        self.log = {
            "times": [], "mid_prices": [], "bids": [], "asks": [],
            "positions": [], "pnls": [], "rewards": [], "cumulative_pnls": [],
            "fills_buy_t": [], "fills_buy_p": [], "fills_buy_type": [],
            "fills_sell_t": [], "fills_sell_p": [], "fills_sell_type": [],
        }

        return self._get_obs(), {}

    def _get_obs(self):
        price = self.scaled_prices[self.t]
        position_risk = self.position * self.target_vol * price  # dollars (scaled)
        time_remaining = (self.T - self.t) / self.T              # [0, 1]

        if self.t < self.T:
            mu = self.mu_predictions[self.t]       # (H,)
            sigma = self.sigma_predictions[self.t]  # (H,)
        else:
            mu = np.zeros(self.H)
            sigma = np.ones(self.H)

        return np.concatenate([
            [position_risk, time_remaining, self.cumulative_pnl],
            mu,
            sigma,
        ]).astype(np.float64)

    def _rescale_action(self, action):
        """Map [-1, 1] -> width in [0, max_width], skew in [-max_skew, max_skew]."""
        width = (action[0] + 1.0) / 2.0 * self.max_width       # [0, max_width]
        skew = action[1] * self.max_skew                         # [-max_skew, max_skew]
        return width, skew

    def _compute_quotes(self, mid, width, skew):
        """Compute bid and offer from mid, width, skew (all in half_spread units)."""
        bid = mid - width * self.half_spread + skew * self.half_spread
        offer = mid + width * self.half_spread + skew * self.half_spread
        return bid, offer

    def _inventory_penalty(self):
        """Time-varying quadratic inventory penalty: (n * p / (R² * sqrt(max(T-t, tau))))²"""
        time_left = max(self.T - self.t, self.tau)
        l_t = 1.0 / (self.r_squared * np.sqrt(time_left))
        return (self.n_sigma * l_t * self.position) ** 2

    def step(self, action):
        if self.t >= self.T - 1:
            return self._get_obs(), 0.0, True, False, {}

        mid = self.scaled_prices[self.t]
        next_mid = self.scaled_prices[self.t + 1]
        delta_mid = next_mid - mid

        # Market quotes
        bm = mid - self.half_spread
        om = mid + self.half_spread
        next_bm = next_mid - self.half_spread
        next_om = next_mid + self.half_spread

        # Agent quotes
        width, skew = self._rescale_action(action)
        bid, offer = self._compute_quotes(mid, width, skew)

        # Save pre-fill position
        position_before = self.position
        pnl_step = 0.0

        # --- Aggressive fills: current quote vs current market ---
        if bid >= om:
            # Agent's bid crosses market offer → aggressive buy at om
            self.position += 1
            pnl_step += (next_mid - om)  # immediate mark-to-market
            self.log["fills_buy_t"].append(self.t)
            self.log["fills_buy_p"].append(om * self.initial_price)
            self.log["fills_buy_type"].append("aggressive")

        if offer <= bm:
            # Agent's offer crosses market bid → aggressive sell at bm
            self.position -= 1
            pnl_step += (bm - next_mid)  # immediate mark-to-market
            self.log["fills_sell_t"].append(self.t)
            self.log["fills_sell_p"].append(bm * self.initial_price)
            self.log["fills_sell_type"].append("aggressive")

        # --- Passive fills: previous quote vs current market ---
        if self.prev_bid is not None:
            if next_om <= self.prev_bid:
                # Market offer moved through our resting bid → passive buy at prev_bid
                self.position += 1
                pnl_step += (next_mid - self.prev_bid)  # mark from fill price to current mid
                self.log["fills_buy_t"].append(self.t)
                self.log["fills_buy_p"].append(self.prev_bid * self.initial_price)
                self.log["fills_buy_type"].append("passive")

            if next_bm >= self.prev_offer:
                # Market bid moved through our resting offer → passive sell at prev_offer
                self.position -= 1
                pnl_step += (self.prev_offer - next_mid)  # mark from fill price to current mid
                self.log["fills_sell_t"].append(self.t)
                self.log["fills_sell_p"].append(self.prev_offer * self.initial_price)
                self.log["fills_sell_type"].append("passive")

        # Unrealized PnL on pre-fill position
        pnl_step += position_before * delta_mid

        # Update cumulative PnL
        self.cumulative_pnl += pnl_step

        # Reward
        terminated = self.t >= self.T - 2  # next step would be T-1

        reward = pnl_step - self._inventory_penalty()
        if terminated:
            # Terminal penalty: forced EOD liquidation cost
            eod_cost = self.lambda2 * abs(self.position) * self.half_spread
            reward -= eod_cost
            self.cumulative_pnl -= eod_cost

        # Logging
        self.log["times"].append(self.t)
        self.log["mid_prices"].append(mid * self.initial_price)
        self.log["bids"].append(bid * self.initial_price)
        self.log["asks"].append(offer * self.initial_price)
        self.log["positions"].append(self.position)
        self.log["pnls"].append(pnl_step)
        self.log["rewards"].append(reward)
        self.log["cumulative_pnls"].append(self.cumulative_pnl)

        # Save current quotes for next step's passive fill check
        self.prev_bid = bid
        self.prev_offer = offer

        # Advance
        self.t += 1

        return self._get_obs(), reward, terminated, False, {}

    def get_log(self):
        """Return the trading log as a dict of lists."""
        return {k: list(v) for k, v in self.log.items()}


class SignalExposureEnv(gym.Env):
    """
    Simplified environment for Phase 1 RL: learn to map predictive signal to
    directional exposure.

    Action: 1D continuous [-1, 1], mapped to [-max_position, max_position].
    Reward: reward_scale * position * target_return
            - alpha_pos * position^2
            - beta_trade * (position - prev_position)^2

    No fills, no spread mechanics — just position management against future returns.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        returns: np.ndarray,              # (T,) single-stock return series
        mu_predictions: np.ndarray,       # (T, H) predicted cumulative returns per horizon
        sigma_predictions: np.ndarray,    # (T, H) uncertainty per horizon
        target_horizon: int = 16,
        reward_scale: float = 10.0,
        alpha_pos: float = 0.01,
        beta_trade: float = 0.005,
        max_position: float = 1.0,
        ablate_penalties: bool = False,
        reward_mode: str = "signal_exposure",
        reward_horizon: int = 8,
        action_space_type: str = "continuous",
        **kwargs,                         # absorb unused market-making params
    ):
        super().__init__()

        self.returns = returns.astype(np.float64)
        self.mu_predictions = mu_predictions.astype(np.float64)
        self.sigma_predictions = sigma_predictions.astype(np.float64)
        self.T = len(returns)
        self.H = mu_predictions.shape[1]

        self.target_horizon = target_horizon
        self.reward_scale = reward_scale
        self.max_position = max_position
        self.reward_mode = reward_mode
        self.reward_horizon = reward_horizon
        self.action_space_type = action_space_type

        # Penalty ablation: zero out penalties when flag is set
        if ablate_penalties:
            self.alpha_pos = 0.0
            self.beta_trade = 0.0
        else:
            self.alpha_pos = alpha_pos
            self.beta_trade = beta_trade

        # Precompute cumulative prices for target return calculation
        self.cum_returns = np.cumsum(self.returns)  # cumulative sum of returns

        # Observation: [position_normalized, time_remaining, cumulative_reward,
        #               mu_1..mu_H, sigma_1..sigma_H]
        obs_dim = 3 + 2 * self.H
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        # Action space: continuous (1D) or discrete3 ({-1, 0, +1})
        if action_space_type == "discrete3":
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float64
            )

        self.reset()

    def _target_return(self, t):
        """Realized future return over target_horizon steps from time t."""
        end = min(t + self.target_horizon, self.T)
        if end <= t:
            return 0.0
        return float(self.returns[t:end].sum())

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.position = 0.0
        self.cumulative_reward = 0.0

        # Logging for diagnostics
        self.log = {
            "positions": [], "actions": [], "rewards": [],
            "target_returns": [], "turnover": [],
        }

        return self._get_obs(), {}

    def _get_obs(self):
        position_norm = self.position / self.max_position  # [-1, 1]
        time_remaining = (self.T - self.t) / self.T

        if self.t < self.T:
            mu = self.mu_predictions[self.t]
            sigma = self.sigma_predictions[self.t]
        else:
            mu = np.zeros(self.H)
            sigma = np.ones(self.H)

        return np.concatenate([
            [position_norm, time_remaining, self.cumulative_reward],
            mu,
            sigma,
        ]).astype(np.float64)

    def step(self, action):
        if self.t >= self.T - 1:
            return self._get_obs(), 0.0, True, False, {}

        # Map action to target position
        if self.action_space_type == "discrete3":
            # action is int in {0, 1, 2} -> position in {-1, 0, +1} * max_position
            new_position = float(action - 1) * self.max_position
        else:
            new_position = float(np.clip(action[0], -1.0, 1.0)) * self.max_position
        prev_position = self.position

        # Reward computation depends on reward_mode
        if self.reward_mode == "horizon_aligned":
            # Horizon-aligned reward: position * future return over reward_horizon
            end = min(self.t + self.reward_horizon, self.T)
            if end <= self.t:
                target_ret = 0.0
            else:
                target_ret = float(self.returns[self.t:end].sum())
            reward = (
                self.reward_scale * new_position * target_ret
                - self.alpha_pos * new_position ** 2
                - self.beta_trade * (new_position - prev_position) ** 2
            )
        else:
            # Default signal_exposure reward
            target_ret = self._target_return(self.t)
            reward = (
                self.reward_scale * new_position * target_ret
                - self.alpha_pos * new_position ** 2
                - self.beta_trade * (new_position - prev_position) ** 2
            )

        self.position = new_position
        self.cumulative_reward += reward

        # Logging
        self.log["positions"].append(new_position)
        self.log["actions"].append(float(action - 1) if self.action_space_type == "discrete3" else float(action[0]))
        self.log["rewards"].append(reward)
        self.log["target_returns"].append(target_ret)
        self.log["turnover"].append(abs(new_position - prev_position))

        self.t += 1
        terminated = self.t >= self.T - 1

        return self._get_obs(), reward, terminated, False, {}

    def get_log(self):
        return {k: list(v) for k, v in self.log.items()}


class VectorizedSignalExposureEnv:
    """Runs N independent SignalExposureEnv instances in parallel."""

    def __init__(self, returns_list, mu_list, sigma_list, **kwargs):
        self.envs = [
            SignalExposureEnv(r, mu, sigma, **kwargs)
            for r, mu, sigma in zip(returns_list, mu_list, sigma_list)
        ]
        self.n_envs = len(self.envs)
        self.action_space_type = kwargs.get("action_space_type", "continuous")

    def reset(self):
        obs = np.stack([env.reset()[0] for env in self.envs])
        return obs

    def step(self, actions):
        """actions: (n_envs, 1) for continuous or (n_envs,) int for discrete3"""
        obs_list, rew_list, done_list = [], [], []
        for i, env in enumerate(self.envs):
            if self.action_space_type == "discrete3":
                act = int(actions[i])
            else:
                act = actions[i]
            obs, rew, term, trunc, info = env.step(act)
            done = term or trunc
            if done:
                obs, _ = env.reset()
            obs_list.append(obs)
            rew_list.append(rew)
            done_list.append(done)
        return np.stack(obs_list), np.array(rew_list), np.array(done_list)


class VectorizedMarketEnv:
    """
    Runs N independent MarketMakingEnv instances in parallel (numpy-vectorized).
    Compatible with a manual training loop (not gymnasium's VecEnv API).
    """

    def __init__(self, returns_list, mu_list, sigma_list, **kwargs):
        """
        returns_list: list of (T,) arrays, one per env
        mu_list: list of (T, H) arrays — per-horizon mu predictions
        sigma_list: list of (T, H) arrays — per-horizon sigma predictions
        """
        self.envs = [
            MarketMakingEnv(r, mu, sigma, **kwargs)
            for r, mu, sigma in zip(returns_list, mu_list, sigma_list)
        ]
        self.n_envs = len(self.envs)

    def reset(self):
        obs = np.stack([env.reset()[0] for env in self.envs])
        return obs

    def step(self, actions):
        """actions: (n_envs, 2)"""
        obs_list, rew_list, done_list = [], [], []
        for i, env in enumerate(self.envs):
            obs, rew, term, trunc, info = env.step(actions[i])
            done = term or trunc
            if done:
                obs, _ = env.reset()
            obs_list.append(obs)
            rew_list.append(rew)
            done_list.append(done)
        return np.stack(obs_list), np.array(rew_list), np.array(done_list)
