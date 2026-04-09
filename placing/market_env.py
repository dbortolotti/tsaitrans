"""
market_env.py

Gymnasium environment for a single-stock market-making agent.

The agent places a bid and an offer at each timestep. If the price moves
through the bid, the agent buys; if through the offer, the agent sells.

Observation space:
    [predicted_return, current_position, realised_vol, inventory_age, mid_price_normed]

Action space (continuous, 2D):
    [bid_offset, ask_offset]   both in units of recent volatility
    Offsets are always positive (distance from mid).  The network outputs
    raw values in [-1, 1] which we rescale to [min_offset, max_offset].

Reward:
    r_t = mark_to_market_pnl
          - lambda_inventory * position^2
          - kappa_spread * (bid_offset + ask_offset)

This gives a clean signal: earn the spread, but don't accumulate inventory.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MarketMakingEnv(gym.Env):
    """Single-stock market-making environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        returns: np.ndarray,          # (T,) single-stock return series
        predictions: np.ndarray,      # (T,) predicted returns from transformer
        lambda_inventory: float = 0.01,
        kappa_spread: float = 0.0005,
        max_position: int = 10,       # hard position limit
        min_offset: float = 0.2,      # min offset in vol units
        max_offset: float = 3.0,      # max offset in vol units
        vol_lookback: int = 20,       # window for realised vol
        initial_price: float = 100.0,
    ):
        super().__init__()

        assert len(returns) == len(predictions), "returns and predictions must be same length"

        self.returns = returns.astype(np.float64)
        self.predictions = predictions.astype(np.float64)
        self.T = len(returns)

        # Params
        self.lambda_inv = lambda_inventory
        self.kappa_spread = kappa_spread
        self.max_position = max_position
        self.min_offset = min_offset
        self.max_offset = max_offset
        self.vol_lookback = vol_lookback
        self.initial_price = initial_price

        # Spaces
        # Observation: [prediction, position_normed, vol, inventory_age_normed, price_return]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float64
        )
        # Action: [bid_offset_raw, ask_offset_raw] in [-1, 1], rescaled internally
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float64
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.vol_lookback  # start after enough history for vol
        self.position = 0
        self.cash = 0.0
        self.price = self.initial_price
        self.inventory_age = 0  # how many steps since last flat

        # Reconstruct price series from returns for this episode
        self.prices = np.zeros(self.T + 1)
        self.prices[0] = self.initial_price
        for i in range(self.T):
            self.prices[i + 1] = self.prices[i] * (1.0 + self.returns[i])

        self.price = self.prices[self.t]

        # Trading log for visualization
        self.log = {
            "times": [], "mid_prices": [], "bids": [], "asks": [],
            "positions": [], "pnls": [], "rewards": [],
            "fills_buy_t": [], "fills_buy_p": [],
            "fills_sell_t": [], "fills_sell_p": [],
        }

        return self._get_obs(), {}

    def _get_obs(self):
        pred = self.predictions[self.t] if self.t < self.T else 0.0
        pos_normed = self.position / self.max_position
        vol = np.std(self.returns[max(0, self.t - self.vol_lookback):self.t]) + 1e-8
        inv_age_normed = min(self.inventory_age / 50.0, 1.0)
        price_ret = self.returns[self.t - 1] if self.t > 0 else 0.0
        return np.array([pred, pos_normed, vol, inv_age_normed, price_ret], dtype=np.float64)

    def _rescale_offset(self, raw, vol):
        """Map [-1, 1] -> [min_offset, max_offset] in vol units, then to price."""
        frac = (raw + 1.0) / 2.0  # [0, 1]
        offset_vol_units = self.min_offset + frac * (self.max_offset - self.min_offset)
        return offset_vol_units * vol * self.price

    def step(self, action):
        if self.t >= self.T - 1:
            return self._get_obs(), 0.0, True, False, {}

        vol = np.std(self.returns[max(0, self.t - self.vol_lookback):self.t]) + 1e-8
        mid = self.prices[self.t]

        # Compute bid and ask prices
        bid_offset = self._rescale_offset(action[0], vol)
        ask_offset = self._rescale_offset(action[1], vol)
        bid_price = mid - bid_offset
        ask_price = mid + ask_offset

        # Next price
        next_price = self.prices[self.t + 1]
        price_move = next_price - mid

        # Determine fills: if next price goes through our level, we get filled
        bought = False
        sold = False

        if next_price <= bid_price and self.position < self.max_position:
            # Our bid gets hit — we buy
            self.position += 1
            self.cash -= bid_price
            bought = True
            self.log["fills_buy_t"].append(self.t)
            self.log["fills_buy_p"].append(bid_price)

        if next_price >= ask_price and self.position > -self.max_position:
            # Our offer gets lifted — we sell
            self.position -= 1
            self.cash += ask_price
            sold = True
            self.log["fills_sell_t"].append(self.t)
            self.log["fills_sell_p"].append(ask_price)

        # Mark-to-market PnL change
        mtm_before = self.position * mid + self.cash  # before fill adjustments already done
        # After fills, recalculate with new position and cash at next price
        mtm_after = self.position * next_price + self.cash
        pnl_step = mtm_after - (self.position * mid + self.cash) + \
                   (self.cash - (self.cash))  # simplify
        # Actually let's just track total PnL properly
        pnl_step = self.position * price_move  # unrealised P&L on existing position
        if bought:
            pnl_step += (next_price - bid_price)  # immediate mark on the fill
        if sold:
            pnl_step += (ask_price - next_price)  # immediate mark on the fill

        # Track inventory age
        if self.position == 0:
            self.inventory_age = 0
        else:
            self.inventory_age += 1

        # Reward
        reward = (
            pnl_step
            - self.lambda_inv * self.position ** 2
            - self.kappa_spread * (bid_offset + ask_offset)
        )

        # Logging
        self.log["times"].append(self.t)
        self.log["mid_prices"].append(mid)
        self.log["bids"].append(bid_price)
        self.log["asks"].append(ask_price)
        self.log["positions"].append(self.position)
        self.log["pnls"].append(pnl_step)
        self.log["rewards"].append(reward)

        # Advance
        self.t += 1
        self.price = next_price

        terminated = self.t >= self.T - 1
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

    def get_log(self):
        """Return the trading log as a dict of lists."""
        return {k: list(v) for k, v in self.log.items()}


class VectorizedMarketEnv:
    """
    Runs N independent MarketMakingEnv instances in parallel (numpy-vectorized).
    Compatible with a manual training loop (not gymnasium's VecEnv API).
    """

    def __init__(self, returns_list, predictions_list, **kwargs):
        """
        returns_list: list of (T,) arrays, one per env
        predictions_list: list of (T,) arrays
        """
        self.envs = [
            MarketMakingEnv(r, p, **kwargs)
            for r, p in zip(returns_list, predictions_list)
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
