"""
market_env.py

Gymnasium environment for a single-stock market-making agent.

The environment uses:
  - one optional aggressive action per step: buy, sell, or none
  - optional passive bid / offer placement, each with a learned quote distance
  - one-tick passive order lifetime

See MARKET_ENV.md for fill mechanics and RL_DESIGN.md for reward/observation
design.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MarketMakingEnv(gym.Env):
    """Single-stock market-making environment with one-tick passive orders."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        returns: np.ndarray,              # (T,) single-stock return series
        mu_predictions: np.ndarray,       # (T, H) predicted cumulative returns per horizon
        sigma_predictions: np.ndarray,    # (T, H) uncertainty per horizon
        half_spread: float = 0.001,       # fixed market half-spread
        max_position: int = 5,            # symmetric hard inventory cap
        k_max: float = 3.0,               # max passive quote distance in half-spread units
        kappa_base: float = 1e-4,         # base quadratic inventory cost
        kappa_close: float = 5e-4,        # extra inventory cost near the close
        lambda2: float = 1.5,             # terminal liquidation multiplier
        initial_price: float = 100.0,
    ):
        super().__init__()

        assert len(returns) == len(mu_predictions), "returns and mu_predictions must be same length"
        assert mu_predictions.shape == sigma_predictions.shape, "mu and sigma shape mismatch"
        assert max_position >= 1, "max_position must be >= 1"

        self.returns = returns.astype(np.float64)
        self.mu_predictions = mu_predictions.astype(np.float64)
        self.sigma_predictions = sigma_predictions.astype(np.float64)
        self.T = len(returns)
        self.H = mu_predictions.shape[1]

        self.half_spread = float(half_spread)
        self.max_position = int(max_position)
        self.k_max = float(k_max)
        self.kappa_base = float(kappa_base)
        self.kappa_close = float(kappa_close)
        self.lambda2 = float(lambda2)
        self.initial_price = float(initial_price)
        self._k_epsilon = 1e-6

        # Raw continuous action decoded into:
        # [aggressive_side, place_bid, k_bid, place_offer, k_offer]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float64
        )

        # Observation:
        # [position_norm, time_remaining, bid_active, live_k_bid,
        #  offer_active, live_k_offer, mu_1..mu_H, sigma_1..sigma_H]
        obs_dim = 6 + 2 * self.H
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.position = 0
        self.cumulative_pnl = 0.0

        self.prices = np.zeros(self.T + 1)
        self.prices[0] = self.initial_price
        for i in range(self.T):
            self.prices[i + 1] = self.prices[i] * (1.0 + self.returns[i])
        self.scaled_prices = self.prices / self.initial_price

        # Orders currently live for this step's passive fill checks.
        self.live_bid_active = False
        self.live_offer_active = False
        self.live_k_bid = 0.0
        self.live_k_offer = 0.0
        self.live_bid_price = 0.0
        self.live_offer_price = 0.0

        self.log = {
            "times": [],
            "mid_prices": [],
            "positions": [],
            "rewards": [],
            "pnls": [],
            "inventory_pnls": [],
            "fill_pnls": [],
            "inventory_penalties": [],
            "terminal_costs": [],
            "cumulative_pnls": [],
            "bid_active": [],
            "offer_active": [],
            "bid_prices": [],
            "offer_prices": [],
            "fills_buy_t": [],
            "fills_buy_p": [],
            "fills_buy_type": [],
            "fills_sell_t": [],
            "fills_sell_p": [],
            "fills_sell_type": [],
        }

        return self._get_obs(), {}

    def _get_obs(self):
        time_remaining = (self.T - self.t) / self.T

        if self.t < self.T:
            mu = self.mu_predictions[self.t]
            sigma = self.sigma_predictions[self.t]
        else:
            mu = np.zeros(self.H)
            sigma = np.ones(self.H)

        return np.concatenate([
            [
                self.position / self.max_position,
                time_remaining,
                float(self.live_bid_active),
                self.live_k_bid if self.live_bid_active else 0.0,
                float(self.live_offer_active),
                self.live_k_offer if self.live_offer_active else 0.0,
            ],
            mu,
            sigma,
        ]).astype(np.float64)

    def _decode_aggressive(self, raw: float) -> str:
        if raw <= -1.0 / 3.0:
            return "sell"
        if raw >= 1.0 / 3.0:
            return "buy"
        return "none"

    def _decode_toggle(self, raw: float) -> bool:
        return raw > 0.0

    def _decode_k(self, raw: float) -> float:
        k_min = -1.0 + self._k_epsilon
        return k_min + (raw + 1.0) * 0.5 * (self.k_max - k_min)

    def _inventory_penalty(self, position_after: int) -> float:
        progress = self.t / max(self.T - 1, 1)
        kappa_t = self.kappa_base + self.kappa_close * (progress ** 2)
        return kappa_t * (position_after ** 2)

    def _can_buy(self) -> bool:
        return self.position + 1 <= self.max_position

    def _can_sell(self) -> bool:
        return self.position - 1 >= -self.max_position

    def step(self, action):
        if self.t >= self.T - 1:
            return self._get_obs(), 0.0, True, False, {
                "step_pnl": 0.0,
                "cumulative_pnl": self.cumulative_pnl,
            }

        raw = np.asarray(action, dtype=np.float64)
        if raw.shape != (5,):
            raise ValueError(f"expected action shape (5,), got {raw.shape}")

        mid = self.scaled_prices[self.t]
        next_mid = self.scaled_prices[self.t + 1]
        delta_mid = next_mid - mid

        bm = mid - self.half_spread
        om = mid + self.half_spread
        next_bm = next_mid - self.half_spread
        next_om = next_mid + self.half_spread

        aggressive = self._decode_aggressive(raw[0])
        place_bid = self._decode_toggle(raw[1])
        k_bid = self._decode_k(raw[2])
        place_offer = self._decode_toggle(raw[3])
        k_offer = self._decode_k(raw[4])

        bid_price_next = mid - k_bid * self.half_spread
        offer_price_next = mid + k_offer * self.half_spread

        position_before = self.position
        fill_pnl = 0.0
        inventory_pnl = position_before * delta_mid

        # Aggressive fill at the current touch.
        if aggressive == "buy" and self._can_buy():
            self.position += 1
            fill_pnl += next_mid - om
            self.log["fills_buy_t"].append(self.t)
            self.log["fills_buy_p"].append(om * self.initial_price)
            self.log["fills_buy_type"].append("aggressive")
        elif aggressive == "sell" and self._can_sell():
            self.position -= 1
            fill_pnl += bm - next_mid
            self.log["fills_sell_t"].append(self.t)
            self.log["fills_sell_p"].append(bm * self.initial_price)
            self.log["fills_sell_type"].append("aggressive")

        # Passive fills from the one-tick-old live orders.
        if self.live_bid_active and next_om <= self.live_bid_price and self._can_buy():
            self.position += 1
            fill_pnl += next_mid - self.live_bid_price
            self.log["fills_buy_t"].append(self.t)
            self.log["fills_buy_p"].append(self.live_bid_price * self.initial_price)
            self.log["fills_buy_type"].append("passive")

        if self.live_offer_active and next_bm >= self.live_offer_price and self._can_sell():
            self.position -= 1
            fill_pnl += self.live_offer_price - next_mid
            self.log["fills_sell_t"].append(self.t)
            self.log["fills_sell_p"].append(self.live_offer_price * self.initial_price)
            self.log["fills_sell_type"].append("passive")

        pnl_step = inventory_pnl + fill_pnl
        self.cumulative_pnl += pnl_step

        inventory_penalty = self._inventory_penalty(self.position)
        reward = pnl_step - inventory_penalty

        terminated = self.t >= self.T - 2
        terminal_cost = 0.0
        if terminated:
            terminal_cost = self.lambda2 * abs(self.position) * self.half_spread
            reward -= terminal_cost
            self.cumulative_pnl -= terminal_cost

        # Passive orders last exactly one future step. Only place an order if a
        # fill on that side would still respect the hard inventory cap.
        next_bid_active = place_bid and (self.position + 1 <= self.max_position)
        next_offer_active = place_offer and (self.position - 1 >= -self.max_position)

        self.log["times"].append(self.t)
        self.log["mid_prices"].append(mid * self.initial_price)
        self.log["positions"].append(self.position)
        self.log["rewards"].append(reward)
        self.log["pnls"].append(pnl_step)
        self.log["inventory_pnls"].append(inventory_pnl)
        self.log["fill_pnls"].append(fill_pnl)
        self.log["inventory_penalties"].append(inventory_penalty)
        self.log["terminal_costs"].append(terminal_cost)
        self.log["cumulative_pnls"].append(self.cumulative_pnl)
        self.log["bid_active"].append(float(next_bid_active))
        self.log["offer_active"].append(float(next_offer_active))
        self.log["bid_prices"].append(
            bid_price_next * self.initial_price if next_bid_active else np.nan
        )
        self.log["offer_prices"].append(
            offer_price_next * self.initial_price if next_offer_active else np.nan
        )

        self.live_bid_active = next_bid_active
        self.live_offer_active = next_offer_active
        self.live_k_bid = k_bid if next_bid_active else 0.0
        self.live_k_offer = k_offer if next_offer_active else 0.0
        self.live_bid_price = bid_price_next if next_bid_active else 0.0
        self.live_offer_price = offer_price_next if next_offer_active else 0.0

        self.t += 1

        return self._get_obs(), reward, terminated, False, {
            "step_pnl": pnl_step,
            "cumulative_pnl": self.cumulative_pnl,
        }

    def get_log(self):
        """Return the trading log as a dict of lists."""
        return {k: list(v) for k, v in self.log.items()}


class VectorizedMarketEnv:
    """Runs N independent MarketMakingEnv instances in parallel."""

    def __init__(self, returns_list, mu_list, sigma_list, **kwargs):
        self.envs = [
            MarketMakingEnv(r, mu, sigma, **kwargs)
            for r, mu, sigma in zip(returns_list, mu_list, sigma_list)
        ]
        self.n_envs = len(self.envs)

    def reset(self):
        obs = np.stack([env.reset()[0] for env in self.envs])
        return obs

    def step(self, actions):
        """actions: (n_envs, 5)"""
        obs_list, rew_list, done_list, info_list = [], [], [], []
        for i, env in enumerate(self.envs):
            obs, rew, term, trunc, info = env.step(actions[i])
            done = term or trunc
            info_list.append(info)
            if done:
                obs, _ = env.reset()
            obs_list.append(obs)
            rew_list.append(rew)
            done_list.append(done)
        return np.stack(obs_list), np.array(rew_list), np.array(done_list), info_list
