# RL Design: State and Reward

This document captures the RL design choices for the simplified
market-making environment.

`MARKET_ENV.md` describes the fill mechanics. This file describes:

- the observation/state seen by the policy
- the reward function
- the terminal liquidation rule

---

## Observation Space

At time `t`, the observation is:

```text
obs_t = [
    position_norm,
    time_remaining,
    bid_active,
    live_k_bid,
    offer_active,
    live_k_offer,
    mu_1, ..., mu_H,
    sigma_1, ..., sigma_H,
]
```

### Definitions

- `position_norm = position / max_position`
- `time_remaining = (T - t) / T`
- `bid_active in {0, 1}` indicates whether a passive bid from the previous step
  is currently live and eligible to fill on this transition
- `live_k_bid` is the live passive bid quote distance in half-spread units, or
  `0` if no live bid exists
- `offer_active in {0, 1}` indicates whether a passive offer from the previous
  step is currently live
- `live_k_offer` is the live passive offer quote distance in half-spread units,
  or `0` if no live offer exists
- `mu_h` is the transformer prediction for cumulative return to horizon `h`
- `sigma_h` is the forecast uncertainty for horizon `h`

### Design rationale

- **Inventory matters directly**: the policy needs current position to manage
  risk and respect the hard cap.
- **Time-to-close matters directly**: the agent should learn to flatten more as
  the episode nears the end.
- **Live passive orders matter directly**: passive fills at time `t` depend on
  the orders submitted at `t-1`, so the policy needs to know what is still live.
- **Multi-horizon forecasts stay in the state**: the transformer predicts
  cumulative returns at multiple horizons, and the policy can decide how to use
  short vs long horizon signals.
- **No cumulative PnL in the state**: it is not needed for the control problem
  and adds extra noise.
- **No absolute price level in the state**: passive quote distances are defined
  in half-spread units, so the problem is already scale-stable.

---

## Reward Function

Per-step reward is the economic outcome of the transition from `t` to `t+1`:

```text
reward_t = inventory_pnl_t + fill_pnl_t - inventory_penalty_t
```

with terminal liquidation cost applied on the last step:

```text
reward_terminal -= terminal_cost
```

### Inventory PnL

```text
inventory_pnl_t = position_before_t * (mid(t+1) - mid(t))
```

This is the mark-to-market change on inventory carried into the step.

### Fill PnL

`fill_pnl_t` is the sum of the PnL contribution from every fill in the step.

Aggressive fills:

```text
aggressive buy  -> mid(t+1) - om(t)
aggressive sell -> bm(t) - mid(t+1)
```

Passive fills:

```text
passive buy  -> mid(t+1) - bid_t
passive sell -> offer_t - mid(t+1)
```

This means:

- aggressive trading pays the spread
- passive fills can earn spread
- neither aggressive nor passive fills are guaranteed to be profitable after the
  next price move

### Inventory penalty

```text
inventory_penalty_t = kappa_t * position_after_t^2
```

where:

```text
kappa_t = kappa_base + kappa_close * progress(t)^2
progress(t) = t / (T - 1)
```

This is a simple time-varying quadratic inventory cost:

- **early in the episode**: inventory is penalized lightly
- **near the close**: inventory is penalized more heavily

The penalty is intentionally simple. It does not depend directly on transformer
`R^2`; the transformer horizon structure is already available through
`mu_1..mu_H` and `sigma_1..sigma_H` in the state.

---

## Terminal Liquidation

Any residual inventory at the final step is penalized via:

```text
terminal_cost = lambda2 * abs(position_T) * half_spread
```

Interpretation:

- `abs(position_T) * half_spread` models paying the spread to flatten
- `lambda2 >= 1` can add extra urgency / liquidation cost

This gives the policy a direct incentive to avoid carrying large inventory into
the end of the episode.

---

## Summary

The policy is trained to:

- earn spread when passive fills are attractive
- use aggressive trades selectively
- manage inventory through the day
- finish close to flat

The reward is intentionally economic and local. The transformer predictions
inform the policy through the observation, not through a forecast-shaped reward.
